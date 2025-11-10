from typer import Argument, Option, Typer

app = Typer()


@app.command(name='ask', help='Ask LLM a question')
def ask(
    question: str = Argument(help='The question to ask ChatGPT'),
    model_name: str = Option('gpt-4o', help='The name of the ChatGPT model to use'),
) -> None:
    pass
