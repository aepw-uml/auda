from auda.core import project
from auda.dat.__common import run_pipeline
from auda.services.data import DataService
from auda.utils.pipeline import IOSpec, Task, task

from .__common import LLM_KIND, LlmISName, LlmOSName


@task(
    id='GENERATE-SQL',
    kind=LLM_KIND,
    description='Generates SQL queries from natural-language prompts using an LLM.',
    input_specs={
        LlmISName.MODEL: IOSpec(dtype=str, required=False, default='gpt-4o'),
        LlmISName.PROMPT: IOSpec(dtype=str),
    },
    output_specs={LlmOSName.ANSWER: IOSpec(dtype=str)},
)
class GenerateSql(Task):
    def run(self) -> None:
        developer_prompt = self.get_developer_prompt()

        inputs = {
            LlmISName.MODEL: self.get_input(LlmISName.MODEL),
            LlmISName.DEVELOPER_PROMPT: developer_prompt,
            LlmISName.USER_PROMPT: self.get_input(LlmISName.PROMPT),
        }
        outputs, _ = run_pipeline(['GENERATE'], inputs)
        self.set_output(LlmOSName.ANSWER, outputs[LlmOSName.ANSWER])

    def get_developer_prompt(self) -> str:
        """
        Generates the developer prompt for SQL generation using the data table columns.

        Returns:
            The developer prompt string.
        """
        # Get data table columns from DataService
        data_service = project.singleton_registry.get(DataService)
        data_table_columns = data_service.get_data_table_columns()

        # Prepare the developer prompt with data table columns
        data_table_count = len(data_table_columns)
        data_table_names = ', '.join(data_table_columns)
        data_table_details = '\n'.join(
            f'  - {table_name}: {columns}'
            for table_name, columns in data_table_columns.items()
        )
        countries = data_service.get_countries()

        template = project.get_template('developer_prompt/generate_sql.template')
        return template.substitute(
            data_table_count=data_table_count,
            data_table_names=data_table_names,
            data_table_details=data_table_details,
            country_names=', '.join(countries),
        )
