import json
from typing import Any, Dict, List, override

from auda.core import project
from auda.dat import run_pipeline
from auda.dat.datasets import DatasetOSName, Samples
from auda.dat.db import DbISName, DbOSName
from auda.dat.db.__common import SqlResults
from auda.dat.plotters.__common import PlotterISName
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
        LlmOSName.MODEL_REQUIREMENTS: IOSpec(dtype=Dict[str, Any]),
    },
)
class GenerateAnalysisReport(Task):
    @override
    def run(self) -> None:
        logger = project.get_logger('RetrieveResults')

        # ---- Generate SQL statements
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

        # ---- Retrieve query results for each statement
        outputs, _ = run_pipeline(
            ['RETRIEVE-RESULTS'], {DbISName.SQL_STATEMENTS: sql_statements}
        )
        sql_results_list = outputs[DbOSName.SQL_RESULTS_LIST]

        # ---- Generate model requirements based on results
        outputs, _ = run_pipeline(
            ['GENERATE-MR'],
            {
                LlmISName.MODEL: self.get_input(LlmISName.MODEL),
                LlmISName.PROMPT: self.get_input(LlmISName.PROMPT),
                LlmISName.SQL_QUERY: sql_statements,
                LlmISName.SQL_RESULTS_LIST: sql_results_list,
            },
        )

        answer = outputs[LlmOSName.ANSWER]
        if answer.startswith('Error:'):
            raise RuntimeError(f'Error generating analysis report: {answer}')

        model_requirements: Dict[str, Any] = json.loads(answer)

        # ---- Generate images based on model requirements
        sql_results: SqlResults = sql_results_list[0]
        feature_names: List[str] = model_requirements.get('feature_names', [])
        target_name: str = model_requirements.get('target_name', '')
        data: List[List[Any]] = sql_results.data

        samples: Samples
        if target_name:
            # Supevised learning
            indexes_of_features = [
                sql_results.column_names.index(name) for name in feature_names
            ]
            index_of_target = sql_results.column_names.index(target_name)
            samples = [
                (
                    [float(data[i][j]) for j in indexes_of_features],
                    float(data[i][index_of_target]),
                )
                for i in range(len(data))
            ]
        else:
            # Unsupervised learning
            samples = []

        self.generate_images(model_requirements, samples)

        # ---- Populate outputs
        self.set_output(LlmOSName.ANSWER, outputs[LlmOSName.ANSWER])
        self.set_output(LlmOSName.SQL_STATEMENTS, sql_statements)
        self.set_output(LlmOSName.MODEL_REQUIREMENTS, model_requirements)

    # TODO: Define model requirements
    def generate_images(
        self, model_requirements: Dict[str, Any], samples: Samples
    ) -> None:
        model_name = model_requirements.get('model_name')
        if model_name is None:
            raise ValueError('Model name is missing in model requirements.')

        feature_names: List[str] = model_requirements.get('feature_names', [])
        units: List[str] = model_requirements.get('units', [])
        label: str = model_requirements.get('label', '')
        title: str = model_requirements.get('title', '')

        match model_name:
            case 'PR':
                run_pipeline(
                    ['ST-BASIC', 'MD-PR', 'PL-PR', 'SAVE'],
                    {
                        DatasetOSName.SAMPLES: samples,
                        DatasetOSName.ORIGINAL_SAMPLES: samples,
                        DatasetOSName.FEATURE_NAMES: feature_names,
                        DatasetOSName.UNITS: units,
                        DatasetOSName.LABEL: label,
                        PlotterISName.TITLE: title,
                        PlotterISName.SAVE_PATH: 'plot.png',
                        **model_requirements,
                    },
                )
            case 'SVR':
                run_pipeline(
                    ['ST-BASIC', 'MD-SVR', 'PL-SVR', 'SAVE'],
                    {
                        DatasetOSName.SAMPLES: samples,
                        DatasetOSName.ORIGINAL_SAMPLES: samples,
                        DatasetOSName.FEATURE_NAMES: feature_names,
                        DatasetOSName.UNITS: units,
                        DatasetOSName.LABEL: label,
                        PlotterISName.TITLE: title,
                        PlotterISName.SAVE_PATH: 'plot.png',
                        **model_requirements,
                    },
                )
