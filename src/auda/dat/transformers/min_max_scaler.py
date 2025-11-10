from typing import Tuple, override

from auda.dat.datasets import LabeledSamples
from auda.utils.pipeline import IOSpec, Task, task

from .__common import TRANSFORMER_KIND, TransformerISName, TransformerOSName


@task(
    id='TF-SCALE',
    kind=TRANSFORMER_KIND,
    description='Rescales feature values in samples to a user-defined range using '
    'Min–Max normalization.',
    input_specs={
        TransformerISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        TransformerISName.SCALER_MIN: IOSpec(dtype=float, required=False, default=-1.0),
        TransformerISName.SCALER_MAX: IOSpec(dtype=float, required=False, default=1.0),
    },
    output_specs={
        TransformerOSName.SAMPLES: IOSpec(dtype=LabeledSamples),
        TransformerOSName.SCALED_SAMPLES: IOSpec(dtype=LabeledSamples),
    },
)
class MinMaxScalerTransformer(Task):
    @override
    def run(self) -> None:
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler

        samples: LabeledSamples = self.get_input('samples')
        min_value: float = float(self.get_input('scaler_min'))
        max_value: float = float(self.get_input('scaler_max'))
        feature_range: Tuple[float, float] = (min_value, max_value)

        # ----- Normalization
        is_labeled_samples = isinstance(samples[0], tuple)
        x = (
            np.array([sample[0] for sample in samples])
            if is_labeled_samples
            else np.array(samples)
        )
        x_scaled = MinMaxScaler(feature_range=feature_range).fit_transform(x)

        # ---- Convert it back to the original format
        if is_labeled_samples:
            scaled_samples = [
                (x_scaled[i].tolist(), samples[i][1]) for i in range(len(samples))
            ]
        else:
            scaled_samples = x_scaled.tolist()

        # Output
        self.set_output(TransformerOSName.SAMPLES, scaled_samples)
        self.set_output(TransformerOSName.NORMALIZED_SAMPLES, scaled_samples)
