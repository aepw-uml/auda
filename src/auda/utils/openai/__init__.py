from openai import OpenAI
from openai.types.chat import ChatCompletion

client = OpenAI()


class ChatgptRole:
    # The developer or application designer
    SYSTEM = 'system'

    # The end user
    USER = 'user'

    # The AI assistant
    ASSISTANT = 'assistant'


def ask_chatgpt(developer_prompt: str, user_prompt: str) -> ChatCompletion:
    return client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role': ChatgptRole.SYSTEM, 'content': developer_prompt},
            {'role': ChatgptRole.USER, 'content': user_prompt},
        ],
    )
