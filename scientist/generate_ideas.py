# %%

import os
from logging import getLogger
from typing import Annotated, Literal

import httpx
from agentlens.message import Message
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from scientist.config import ai
from scientist.datasets import ExperimentDataset, ExperimentRow
from scientist.models import Decision, Experiment, Idea, Think

logger = getLogger(__name__)

S2_API_KEY = os.getenv("S2_API_KEY")


@ai.task()
async def generate_novel_ideas(
    experiment: Experiment,
    max_num_generations: int = 20,
    num_refinements: int = 5,
    max_num_iterations: int = 10,
) -> list[Idea]:
    ideas = await generate_ideas(
        experiment=experiment,
        max_num_generations=max_num_generations,
        num_refinements=num_refinements,
    )
    return await select_novel_ideas(
        experiment=experiment,
        ideas=ideas,
        max_num_iterations=max_num_iterations,
    )


@ai.task()
async def generate_ideas(
    experiment: Experiment,
    max_num_generations: int = 20,
    num_refinements: int = 5,
) -> list[Idea]:
    ideas: list[Idea] = []

    for _ in tqdm(range(max_num_generations), desc="Generating idea"):
        idea = await generate_idea(
            init_code=experiment.init_code,
            task_description=experiment.task_description,
            idea_archive=[*experiment.seed_ideas, *ideas],
            num_refinements=num_refinements,
        )
        ideas.append(idea)

    return ideas


@ai.task()
async def generate_idea(
    init_code: str,
    task_description: str,
    idea_archive: list[Idea],
    model: str = "openai:gpt-4o-mini",
    num_refinements: int = 5,
) -> Idea:
    num_rounds = num_refinements + 1  # +1 for the initial idea
    idea_archive_str = ",".join([idea.model_dump_json() for idea in idea_archive])

    # initialize message list to iteratively build up the idea
    messages = [
        ai.message.system(
            "You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field."
        ),
        ai.message.user(f"""\
            {task_description}
            ```python
            # experiment.py
            {init_code}
            ```

            Here are the ideas that you have already generated:
            ```json
            [{idea_archive_str}]
            ```

            Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
            Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

            Notes:
            - You will not have access to any additional resources or datasets.
            - Be cautious and realistic on your ratings.    
            - Only signal "all_done" if it's pretty clear there's nothing more to improve on.
            - You will have {num_rounds} rounds to iterate on the idea, but do not need to use them all.
        """),
    ]

    def append_idea(idea: Idea) -> None:
        new_message = ai.message.assistant(idea.model_dump_json())
        messages.append(new_message)

    for i in tqdm(range(num_rounds), desc="Ideating"):
        if i == 0:
            idea = await generate_initial_idea(
                messages=messages,
                model=model,
            )
        else:
            maybe_idea = await refine_idea(
                messages=messages,
                model=model,
                current_round=i,
                num_rounds=num_rounds,
            )
            if maybe_idea is None:
                break
            idea = maybe_idea

        append_idea(idea)

    return idea


@ai.task()
async def generate_initial_idea(
    *,
    messages: list[Message],
    model: str = "openai:gpt-4o-mini",
) -> Idea:
    idea = await ai.generate_object(
        model=model,
        messages=messages,
        type=Idea,
    )
    score_idea(idea)
    return idea


@ai.task()
async def refine_idea(
    *,
    messages: list[Message],
    current_round: int,
    num_rounds: int,
    model: str = "openai:gpt-4o-mini",
) -> Idea | None:
    result = await ai.generate_object(
        model=model,
        messages=[
            *messages,
            ai.message.user(f"""\
                Round {current_round}/{num_rounds}.
                In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
                Include any other factors that you think are important in evaluating the idea.
                Ensure the idea is clear and concise, and the JSON is the correct format.
                Do not make things overly complicated.
                In the next attempt, try and refine and improve your idea.
                Stick to the spirit of the original idea unless there are glaring issues.
            """),
        ],
        type=Think[Idea | Decision[bool]],
    )
    if isinstance(result.action, Idea):
        score_idea(result.action)
        return result.action
    else:
        return None


def score_idea(idea: Idea) -> None:
    ai.score(
        name="interestingness",
        value=idea.interestingness,
    )
    ai.score(
        name="feasibility",
        value=idea.feasibility,
    )
    ai.score(
        name="novelty",
        value=idea.novelty,
    )


async def select_novel_ideas(
    experiment: Experiment,
    ideas: list[Idea],
    max_num_iterations: int = 10,
) -> list[Idea]:
    novel_ideas = []
    for idea in tqdm(ideas, desc="Checking novelty"):
        is_novel = await check_idea_novelty(
            experiment=experiment,
            idea=idea,
            max_num_iterations=max_num_iterations,
        )
        if is_novel:
            novel_ideas.append(idea)

    return novel_ideas


class PaperMetadata(BaseModel):
    title: str
    authors: str
    venue: str
    year: int
    citation_count: int
    abstract: str


@ai.task()
async def check_idea_novelty(
    experiment: Experiment,
    idea: Idea,
    max_num_iterations: int = 10,
) -> bool:
    messages = [
        ai.message.system(f"""\
            You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
            You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
            Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
            You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
            The top 10 results for any search query will be presented to you with the abstracts.

            You will be given {max_num_iterations} to decide on the paper, but you do not need to use them all.
            At any round, you may exit early and decide on the novelty of the idea.
            Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
            Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

            {experiment.task_description}
            ```python
            # experiment.py
            {experiment.init_code}
            ```
            
            You have this idea:
            ```json
            {idea.model_dump_json()}
            ```
                        
            Notes:
            - Be sure to first reason through the idea and identify any query that could help you make your decision.
            - If your decision is already clear, please write it out, and the process will be terminated early. 
            - A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
        """),
    ]
    last_query_results: list[PaperMetadata] = []

    for j in range(max_num_iterations):
        result, messages = await determine_novelty_or_generate_search_query(
            last_query_results=last_query_results,
            messages=messages,
            current_round=j,
            max_num_iterations=max_num_iterations,
        )
        if isinstance(result, Decision):
            return result.content
        else:
            last_query_results = await search_for_papers(result.query, result_limit=10)

    return False


class LiteratureSearch(BaseModel):
    type: Literal["search"]
    query: str = Field(
        description="A search query to search the literature (e.g. attention is all you need)."
    )


@ai.task()
async def determine_novelty_or_generate_search_query(
    last_query_results: list[PaperMetadata],
    messages: list[Message],
    current_round: int,
    max_num_iterations: int,
    model: str = "openai:gpt-4o-mini",
) -> tuple[LiteratureSearch | Decision[bool], list[Message]]:
    if not last_query_results:
        papers_str = "No papers found."
    else:
        papers_str = "\n\n".join(paper.model_dump_json() for paper in last_query_results)

    new_messages = [
        *messages,
        ai.message.user(f"""\
            Round {current_round}/{max_num_iterations}.
            
            The results of the last query are (empty on first round):
            ```json
            {papers_str}
            ```
        """),
    ]

    thought = await ai.generate_object(
        model=model,
        messages=new_messages,
        type=Think[
            LiteratureSearch
            | Annotated[
                Decision[Annotated[bool, "Whether the idea is novel or not."]],
                "Terminates the process, rendering a final determination of novelty.",
            ]
        ],
    )

    new_messages.append(ai.message.assistant(thought.model_dump_json()))
    return thought.action, new_messages


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(httpx.RequestError),
)
@ai.task(cache=True)
async def search_for_papers(
    query: str,
    result_limit: int = 10,
) -> list[PaperMetadata]:
    if not S2_API_KEY:
        raise ValueError("S2_API_KEY is not set.")
    async with httpx.AsyncClient() as client:
        rsp = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": S2_API_KEY},
            params={
                "query": query,
                "limit": result_limit,
                "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
            },
        )
        rsp.raise_for_status()
        results = rsp.json()
        if not results["total"]:
            return []
        return [PaperMetadata(**paper) for paper in results["data"]]


# %%

dataset = ExperimentDataset("grokking")
len(dataset)

# %%

message_stacks = []


@ai.hook(refine_idea)
def hook_refine_idea(row: ExperimentRow, output, **kwargs):
    message_stacks.append(kwargs["messages"])


novel_ideas = ai.run(
    dataset=dataset,
    main=lambda row: generate_novel_ideas(
        experiment=row.experiment,
        max_num_iterations=2,
        max_num_generations=1,
        num_refinements=2,
    ),
)

# %%

message_stacks[0]
