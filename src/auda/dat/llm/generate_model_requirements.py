from typing import List

from auda.core import project
from auda.dat.__common import run_pipeline
from auda.dat.db import SqlResults
from auda.utils.pipeline import IOSpec, Task, task

from .__common import LLM_KIND, LlmISName, LlmOSName


@task(
    id='GENERATE-MR',
    kind=LLM_KIND,
    description='',
    input_specs={
        LlmISName.MODEL: IOSpec(dtype=str, required=False, default='gpt-4o'),
        LlmISName.PROMPT: IOSpec(dtype=str),
        LlmISName.SQL_QUERY: IOSpec(dtype=str),
        LlmISName.SQL_RESULTS_LIST: IOSpec(dtype=List[SqlResults]),
    },
    output_specs={LlmOSName.ANSWER: IOSpec(dtype=str)},
)
class GenerateSql(Task):
    def run(self) -> None:
        prompt: str = self.get_input(LlmISName.PROMPT)
        sql_query: str = self.get_input(LlmISName.SQL_QUERY)
        sql_results_list: List[SqlResults] = self.get_input(LlmISName.SQL_RESULTS_LIST)
        developer_prompt = self.get_developer_prompt(
            prompt, sql_query, sql_results_list
        )

        inputs = {
            LlmISName.MODEL: self.get_input(LlmISName.MODEL),
            LlmISName.DEVELOPER_PROMPT: developer_prompt,
            LlmISName.USER_PROMPT: prompt,
        }
        outputs, _ = run_pipeline(['GENERATE'], inputs)
        self.set_output(LlmOSName.ANSWER, outputs[LlmOSName.ANSWER])

    def get_developer_prompt(
        self, prompt: str, sql_query: str, sql_results_list: List[SqlResults]
    ) -> str:
        """
        Generates the developer prompt for SQL generation using the data table columns.

        Returns:
            The developer prompt string.
        """
        sql_results = sql_results_list[0] if sql_results_list else None
        if sql_results is None:
            raise ValueError('SQL results are required to generate model requirements.')

        template = project.get_template(
            'developer_prompt/generate_model_requirements.template'
        )
        return template.substitute(
            user_prompt=prompt,
            sql_query=sql_query,
            column_names=sql_results.column_names,
            data=sql_results.data,
        )
