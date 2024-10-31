from enum import Enum
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field


class Idea(BaseModel):
    name: str = Field(
        # serialization_alias="Name",
        description="A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.",
    )
    title: str = Field(
        # serialization_alias="Title",
        description="A title for the idea, will be used for the report writing.",
    )
    experiment: str = Field(
        # serialization_alias="Experiment",
        description="An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...",
    )
    interestingness: int = Field(
        # serialization_alias="Interestingness",
        description="A rating from 1 to 10 (lowest to highest).",
    )
    feasibility: int = Field(
        # serialization_alias="Feasibility",
        description="A rating from 1 to 10 (lowest to highest).",
    )
    novelty: int = Field(
        # serialization_alias="Novelty",
        description="A rating from 1 to 10 (lowest to highest).",
    )
    # novel: bool | None = Field(description="Leave blank")


T = TypeVar("T")
D = TypeVar("D")


class Think(BaseModel, Generic[T]):
    reasoning: str = Field(description="Your reasoning")
    action: T = Field(description="The next action to be taken")


class Decision(BaseModel, Generic[T]):
    type: Literal["decision"]
    content: T = Field(description="Your decision")


class ExperimentName(Enum):
    GROKKING = "grokking"
    DIFFUSION = "2d_diffusion"
    MOBILE_NET = "mobilenetV3"
    NANOGPT = "nanoGPT"
    NANOGPT_LITE = "nanoGPT_lite"
    SEIR = "seir"
    SKETCH_RNN = "sketch_rnn"


class Experiment(BaseModel):
    name: ExperimentName
    task_description: str
    init_code: str
    seed_ideas: list[Idea]


class CodeModification(BaseModel):
    files: dict[str, str] = Field(
        description="A dictionary where keys are filenames and values are the new content."
    )
    message: str = Field(description="Your message or explanation.")


class Review(BaseModel):
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    originality: int = Field(..., ge=1, le=4)
    quality: int = Field(..., ge=1, le=4)
    clarity: int = Field(..., ge=1, le=4)
    significance: int = Field(..., ge=1, le=4)
    questions: list[str]
    limitations: list[str]
    ethical_concerns: bool
    soundness: int = Field(..., ge=1, le=4)
    presentation: int = Field(..., ge=1, le=4)
    contribution: int = Field(..., ge=1, le=4)
    overall: int = Field(..., ge=1, le=10)
    confidence: int = Field(..., ge=1, le=5)
    decision: Literal["accept"] | Literal["reject"]
