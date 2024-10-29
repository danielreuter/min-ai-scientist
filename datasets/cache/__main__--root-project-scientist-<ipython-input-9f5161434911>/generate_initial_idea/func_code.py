# first line: 1
@ai.task(cache=True)
async def generate_initial_idea(
    messages: list[Message],
    model: str = "openai:gpt-4o-mini",
) -> Idea:
    return await ai.generate_object(
        model=model,
        messages=messages,
        type=Idea,
    )
