# %%

from logging import getLogger

from tqdm import tqdm

from reagency.message import Message
from scientist.config import ai
from scientist.datasets import ExperimentDataset, ExperimentRow
from scientist.models import Experiment, Idea, MaybeIdea

logger = getLogger(__name__)


@ai.task()
async def generate_ideas(
    experiment: Experiment,
    max_num_generations: int = 20,
    num_refinements: int = 5,
) -> list[Idea]:
    idea_archive = [*experiment.seed_ideas]
    progress_bar = tqdm(range(max_num_generations), desc="Generating idea")

    for _ in progress_bar:
        idea = await generate_idea(
            init_code=experiment.init_code,
            task_description=experiment.task_description,
            idea_archive=idea_archive,
            num_refinements=num_refinements,
        )
        idea_archive.append(idea)

    return idea_archive


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
            [
                {idea_archive_str}
            ]
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

    progress_bar = tqdm(range(num_rounds), desc="Ideating")

    for i in progress_bar:
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
            if maybe_idea.all_done:
                break
            idea = maybe_idea.idea

        append_idea(idea)

    # return the last idea
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
) -> MaybeIdea:
    maybe_idea = await ai.generate_object(
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
        type=MaybeIdea,
    )
    if maybe_idea.idea:
        score_idea(maybe_idea.idea)
    return maybe_idea


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


# %%

dataset = ExperimentDataset("grokking")
len(dataset)

# %%

message_stacks = []


@ai.hook(refine_idea)
def hook_refine_idea(row: ExperimentRow, output: MaybeIdea, **kwargs):
    message_stacks.append(kwargs["messages"])


results = ai.run(
    dataset=dataset,
    main=lambda row: generate_ideas(
        experiment=row.experiment,
        max_num_generations=1,
        num_refinements=2,
    ),
    hooks=[hook_refine_idea],
)

# %%

message_stacks[0]

# %%
