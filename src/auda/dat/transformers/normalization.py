from typing import override

from auda.dat.__common import run_pipeline
from auda.dat.datasets import LabeledSamples
from auda.utils.pipeline import IOSpec, task

from .__common import TRANSFORMER_KIND, TransformerISName, TransformerOSName
from .min_max_scaler import MinMaxScalerTransformer


@task(
    id='TF-NORM',
    kind=TRANSFORMER_KIND,
    description='Normalizes features (and optionally labels) to [0, 1].',
    input_specs={
        TransformerISName.SAMPLES: IOSpec(dtype=LabeledSamples),
    },
    output_specs={
        TransformerOSName.SAMPLES: IOSpec(dtype=LabeledSamples),
        TransformerOSName.NORMALIZED_SAMPLES: IOSpec(dtype=LabeledSamples),
    },
)
class NormalizationTransformer(MinMaxScalerTransformer):
    @override
    def run(self) -> None:
        inputs = {
            **self._inputs,
            TransformerISName.SCALER_MIN: 0.0,
            TransformerISName.SCALER_MAX: 1.0,
        }

        outputs, _ = run_pipeline(['TF-SCALE'], inputs)

        # ---- Populate outputs
        self._outputs[TransformerOSName.SAMPLES] = outputs[TransformerOSName.SAMPLES]
        self._outputs[TransformerOSName.NORMALIZED_SAMPLES] = outputs[
            TransformerOSName.SCALED_SAMPLES
        ]
