from typing import override

from auda.core import project
from auda.dat import run_pipeline
from auda.dat.db import DbISName, DbOSName
from auda.utils.pipeline import IOSpec, Task, task

from .__common import LLM_KIND, LlmISName, LlmOSName


@task(
    id='GENERATE-AR',
    kind=LLM_KIND,
    description='Generates an analysis report based on provided data and insights.',
    input_specs={
        LlmISName.MODEL: IOSpec(dtype=str, required=False, default='gpt-4o'),
        LlmISName.PROMPT: IOSpec(dtype=str),
    },
    output_specs={
        LlmOSName.ANSWER: IOSpec(dtype=str),
        LlmOSName.SQL_STATEMENTS: IOSpec(dtype=str),
    },
)
class GenerateAnalysisReport(Task):
    @override
    def run(self) -> None:
        logger = project.get_logger('RetrieveResults')
        outputs, _ = run_pipeline(
            ['GENERATE-SQL'],
            {
                LlmISName.MODEL: self.get_input(LlmISName.MODEL),
                LlmISName.PROMPT: self.get_input(LlmISName.PROMPT),
            },
        )

        answer: str = outputs[LlmOSName.ANSWER]
        if answer.startswith('Error:'):
            raise RuntimeError(f'Error generating analysis report: {answer}')

        sql_statements = answer
        logger.info(f'Generated SQL Statements: {sql_statements}')

        outputs, _ = run_pipeline(
            ['RETRIEVE-RESULTS'], {DbISName.SQL_STATEMENTS: sql_statements}
        )
        sql_results_list = outputs[DbOSName.SQL_RESULTS_LIST]

        outputs, _ = run_pipeline(
            ['GENERATE-MR'],
            {
                LlmISName.MODEL: self.get_input(LlmISName.MODEL),
                LlmISName.PROMPT: self.get_input(LlmISName.PROMPT),
                LlmISName.SQL_QUERY: sql_statements,
                LlmISName.SQL_RESULTS_LIST: sql_results_list,
            },
        )

        # ---- Populate outputs
        self.set_output(LlmOSName.ANSWER, outputs[LlmOSName.ANSWER])
        self.set_output(LlmOSName.SQL_STATEMENTS, sql_statements)
