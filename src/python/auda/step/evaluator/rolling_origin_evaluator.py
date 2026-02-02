from typing import List, override

from auda.step import create_pipeline_from_pipe
from auda.step.dataset import DatasetBasedStep
from auda.step.evaluator.normal_evaluator import NormalEvaluator
from auda.step.preprocessing.pick_test_samples import PickTestSamples
from auda.step.preprocessing.sort import SortDataset
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import (
    IOValueMap,
    Pipeline,
    step,
)
from numpy import mean


@step(
    id='EV-RO',
    description='Rolling origin evaluator.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.SEED.optional(42),
        Spec.HT_PIPE,
        Spec.TRAINING_PIPE,
    ],
    output_specs=[Spec.MAE, Spec.RMSE, Spec.R2, Spec.MAPE],
)
class RolloingOriginEvaluator(DatasetBasedStep):
    @override
    def run(
        self,
        on: str | Dataset,
        seed: int,
        ht_pipe: str | Pipeline,
        training_pipe: str | Pipeline,
    ) -> IOValueMap:
        dataset = self.get_dataset_from_on(on)
        num_samples = self.get_num_samples(dataset)
        if num_samples < 3:
            raise ValueError(
                'Dataset must have at least 3 samples for rolling origin '
                + ' evaluation.'
            )

        maes: List[float] = []
        rmses: List[float] = []
        r2s: List[float] = []
        mapes: List[float] = []

        for i in range(num_samples):
            preprocessing_pipe = (
                Pipeline()
                .append(
                    SortDataset,
                    {
                        Spec.ON.name: dataset,
                        Spec.SORT_BY_FEATURE_INDEX.name: 0,
                    },
                )
                .append(
                    PickTestSamples,
                    {
                        Spec.ON.name: Spec.SORTED_DATASET.name,
                        Spec.TEST_SAMPLE_INDEXES.name: [i],
                    },
                )
            )
            preprocessing_pipe.run()

            training_set = preprocessing_pipe.get_value(Spec.TRAINIING_SET.name)
            test_set = preprocessing_pipe.get_value(Spec.TEST_SET.name)

            _ht_pipe = create_pipeline_from_pipe(ht_pipe).clone()
            _ht_pipe.run({Spec.SEED.name: seed, Spec.ON.name: training_set})
            # Get the best hyperparameters from the pipeline
            best_hyperparameters = _ht_pipe.get_value(
                Spec.BEST_HYPERPARAMETERS.name
            )
            hyperparameter_names = _ht_pipe.get_value(
                Spec.HYPERPARAMETER_NAMES.name
            )

            # Re-run the pipeline with the best hyperparameters on the full
            # training set
            _training_pipe = create_pipeline_from_pipe(training_pipe)
            _training_pipe._step_inputs[0] = {
                **_training_pipe._step_inputs[0],
                **{
                    name: value
                    for name, value in zip(
                        hyperparameter_names,
                        best_hyperparameters,
                    )
                },
            }
            evaluator_pipe = (
                Pipeline()
                .append(
                    NormalEvaluator,
                    {
                        Spec.ON.name: training_set,
                        Spec.SEED.name: seed,
                        Spec.PIPE.name: _training_pipe,
                        Spec.TRAINIING_SET.name: training_set,
                        Spec.TEST_SET.name: test_set,
                    },
                )
                .run()
            )

            mae: float = evaluator_pipe.get_value(Spec.MAE.name)
            rmse: float = evaluator_pipe.get_value(Spec.RMSE.name)
            r2: float = evaluator_pipe.get_value(Spec.R2.name)
            mape: float = evaluator_pipe.get_value(Spec.MAPE.name)
            maes.append(mae)
            rmses.append(rmse)
            r2s.append(r2)
            mapes.append(mape)

        print(maes)
        print(rmses)
        print(r2s)
        print(mapes)

        return {
            Spec.MAE.name: mean(maes),
            Spec.RMSE.name: mean(rmses),
            Spec.R2.name: mean(r2s),
            Spec.MAPE.name: mean(mapes),
        }
