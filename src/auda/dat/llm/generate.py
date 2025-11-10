from typing import override

from auda.utils.llm.large_language_model import LargeLanguageModel
from auda.utils.pipeline import IOSpec, Task, task

from .__common import LLM_KIND, LlmISName, LlmOSName


@task(
    id='GENERATE',
    kind=LLM_KIND,
    description='Generates a natural-language answer using a large language model '
    '(LLM).',
    input_specs={
        LlmISName.MODEL: IOSpec(dtype=str),
        LlmISName.DEVELOPER_PROMPT: IOSpec(dtype=str),
        LlmISName.USER_PROMPT: IOSpec(dtype=str),
    },
    output_specs={LlmOSName.ANSWER: IOSpec(dtype=str)},
)
class Ask(Task):
    @override
    def run(self) -> None:
        from auda.dat.llm import llm_list

        model: str = self.get_input(LlmISName.MODEL)
        user_prompt: str = self.get_input(LlmISName.USER_PROMPT)
        developer_prompt: str = self.get_input(LlmISName.DEVELOPER_PROMPT)
        llm: LargeLanguageModel | None = next(
            (llm for llm in llm_list if llm.name == model), None
        )

        if llm is None:
            raise ValueError(f'LLM model {model} not found.')

        answer = llm.ask(user_prompt, developer_prompt)

        self.set_output(LlmOSName.ANSWER, answer)
