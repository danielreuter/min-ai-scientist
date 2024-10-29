# first line: 1
@ai.task(cache=True)
async def refine_idea(
    messages: list[Message],
    current_round: int,
    num_rounds: int,
    model: str = "openai:gpt-4o-mini",
) -> MaybeIdea:
    return await ai.generate_object(
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
