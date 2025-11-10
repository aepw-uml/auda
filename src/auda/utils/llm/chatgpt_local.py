from auda.utils.llm.large_language_model import LargeLanguageModel


class ChatgptLocal(LargeLanguageModel):
    def __init__(self, name: str = 'gpt-sso'):
        super().__init__(name)

    def ask(self, user_prompt: str, developer_prompt: str | None = None) -> str:
        developer_prompt = developer_prompt or self.developer_prompt
        return ''
