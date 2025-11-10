import os
from typing import override

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from auda.core import project

from .large_language_model import LargeLanguageModel


class ChatgptRole:
    # The developer or application designer
    SYSTEM = 'system'

    # The end user
    USER = 'user'

    # The AI assistant
    ASSISTANT = 'assistant'


class ChatgptRemote(LargeLanguageModel):
    def __init__(self, name: str = 'gpt-4o'):
        super().__init__(name)

        self.client = OpenAI()

        # Set the OpenAI API key from the project environment
        os.environ['OPENAI_API_KEY'] = project.env.OPENAI_API_KEY

    @override
    def ask(self, user_prompt: str, developer_prompt: str | None = None) -> str:
        developer_prompt = developer_prompt or self.developer_prompt

        chat_completion = self.client.chat.completions.create(
            model=self.name,
            messages=[
                self._create_message(ChatgptRole.SYSTEM, developer_prompt),
                self._create_message(ChatgptRole.USER, user_prompt),
            ],
        )

        return chat_completion.choices[0].message.content or ''

    def _create_message(self, role: str, content: str) -> ChatCompletionMessageParam:
        return {'role': role, 'content': content}  # type: ignore
