import asyncio
import json
import textwrap
from logging import getLogger

import pymupdf
import pymupdf4llm
from agentlens.message import Message
from aider.coders import Coder
from pypdf import PdfReader
from tqdm import tqdm

from scientist.config import ai
from scientist.constants import NEURIPS_FORM
from scientist.models import Decision, Review, Think

logger = getLogger(__name__)


@ai.task()
async def perform_review(
    text: str,
    model: str = "openai:gpt-4o-mini",
    instructions_form: str = NEURIPS_FORM,
    instructions_pos_valence: bool = False,
) -> Review:
    instructions = [
        ai.message.system(
            """\
            You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue.
            Be critical and cautious in your decision.
            """
            + "If a paper is good or you are unsure, give it good scores and accept it."
            if instructions_pos_valence
            else "If a paper is bad or you are unsure, give it bad scores and reject it."
        ),
        ai.message.user(f"""\
            {instructions_form}
            Here is the paper you are asked to review:
            {text}
        """),
    ]
    review = await generate_review(instructions, model)
    review = await refine_review(instructions, review, model)
    return review


@ai.task()
async def generate_review(
    instructions: list[Message],
    model: str,
    num_reviews_ensemble: int = 1,
    temperature: float = 0.75,
) -> Review:
    if num_reviews_ensemble > 1:
        tasks = [
            ai.generate_object(
                model=model,
                messages=instructions,
                type=Think[Review],
                temperature=temperature,
            )
            for _ in range(num_reviews_ensemble)
        ]
        results = await asyncio.gather(*tasks)
        reviews = [result.action for result in results]
        review = await generate_meta_review(reviews)
        return review
    else:
        result = await ai.generate_object(
            model=model,
            messages=instructions,
            type=Think[Review],
            temperature=temperature,
        )
        return result.action


@ai.task()
async def refine_review(
    instructions: list[Message],
    review: Review,
    model: str,
    num_refinements: int = 1,
    temperature: float = 0.75,
) -> Review:
    messages = [
        *instructions,
        ai.message.assistant(review.model_dump_json()),
        ai.message.user(f"""\
            Now I am going to have you iteratively refine your review, up to a maximum of {num_refinements} times.
            
            Some notes: 
            - In your thoughts, first carefully consider the accuracy and soundness of the review you just created.
            - Include any other factors that you think are important in evaluating the paper.
            - Ensure the review is clear and concise.
            - Do not make things overly complicated.
            - In the next attempt, try and refine and improve your review.
            - Stick to the spirit of the original review unless there are glaring issues.
            
            If there is nothing to improve, indicate that you are done by issuing a Decision action with value True.
        """),
    ]
    for i in tqdm(range(1, num_refinements), desc="Refining review"):
        result = await ai.generate_object(
            model=model,
            messages=[
                *messages,
                ai.message.user(f"Refinement round {i}/{num_refinements}, begin:"),
            ],
            type=Think[Review | Decision[bool]],
            temperature=temperature,
        )
        if isinstance(result.action, Decision):
            return review
        review = result.action
        messages.append(ai.message.assistant(result.model_dump_json()))

    return review


@ai.task()
async def generate_meta_review(
    reviews: list[Review],
    model: str = "openai:gpt-4o-mini",
    temperature: float = 0.75,
) -> Review:
    result = await ai.generate_object(
        model=model,
        messages=[
            ai.message.system(f"""\
                You are an Area Chair at a machine learning conference. 
                You are in charge of meta-reviewing a paper that was reviewed by {len(reviews)} reviewers. 
                Your job is to aggregate the reviews into a single meta-review in the same format. 
                Be critical and cautious in your decision, find consensus, and respect the opinion of all the reviewers.
            """),
            ai.message.user(
                "\n".join(
                    f"""\
                    Review {i + 1}/{len(reviews)}:
                    ```json
                    {r.model_dump_json()}
                    ```
                    """
                    for i, r in enumerate(reviews)
                )
            ),
        ],
        type=Think[Review],
        temperature=temperature,
    )
    return result.action


def load_paper(pdf_path, num_pages=None, min_size=100):
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("Text too short")
    except Exception as e:
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
        try:
            doc = pymupdf.open(pdf_path)  # open a document
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:  # iterate the document pages
                text = text + page.get_text()  # get plain text encoded as UTF-8
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                text = "".join(page.extract_text() for page in reader.pages)
            else:
                text = "".join(page.extract_text() for page in reader.pages[:num_pages])
            if len(text) < min_size:
                raise Exception("Text too short")

    return text


def load_review(path):
    with open(path, "r") as json_file:
        loaded = json.load(json_file)
    return loaded["review"]


def perform_improvement(review: Review, coder: Coder) -> None:
    coder.run(
        textwrap.dedent(f"""\
        The following review has been created for your research paper:
        {review}

        Improve the text using the review.
    """)
    )
