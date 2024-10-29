from enum import Enum

from pydantic import BaseModel, Field


class Idea(BaseModel):
    name: str = Field(
        description="A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed."
    )
    title: str = Field(description="A title for the idea, will be used for the report writing.")
    experiment: str = Field(
        description="An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ..."
    )
    interestingness: int = Field(description="A rating from 1 to 10 (lowest to highest).")
    feasibility: int = Field(description="A rating from 1 to 10 (lowest to highest).")
    novelty: int = Field(description="A rating from 1 to 10 (lowest to highest).")


class MaybeIdea(BaseModel):
    """Represents some reasoning through an idea (or a decision to stop ideating). Make sure to write your reasoning first!"""

    reasoning: str = Field(
        description="Briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones."
    )
    all_done: bool = Field(
        description="Indicates that no more ideas are needed. Write this to terminate the process -- you don't need to write an idea after this"
    )
    idea: Idea | None = Field(
        description="The idea to be added to the list of ideas. Only include this if all_done is False."
    )


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
