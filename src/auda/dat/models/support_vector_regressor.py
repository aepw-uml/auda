from typing import List, override

from auda.dat.datasets import LabeledSamples
from auda.dat.transformers import TransformerISName, TransformerOSName
from auda.utils.pipeline import IOSpec, Task, task

from .__common import MODEL_KIND, ModelISName, ModelOSName, ModelType


@task(
    id='MD-SVR',
    kind=MODEL_KIND,
    description='Trains a Support Vector Regression (SVR) model for predictive '
    'analysis.',
    input_specs={
        ModelISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        ModelOSName.INLIER_SAMPLES: IOSpec(dtype=LabeledSamples, required=False),
        ModelISName.REGULARIZATION_PARAMETER: IOSpec(
            dtype=float, required=False, default=1.0
        ),
        ModelISName.EPSILON: IOSpec(dtype=float, required=False, default=0.1),
        ModelISName.X_STANDARD_DEVIATION: IOSpec(dtype=List[float]),
        ModelISName.X_MEAN: IOSpec(dtype=List[float]),
        ModelISName.Y_STANDARD_DEVIATION: IOSpec(dtype=float),
        ModelISName.Y_MEAN: IOSpec(dtype=List[float]),
    },
    output_specs={
        ModelOSName.MODEL_TYPE: IOSpec(dtype=str),
        ModelOSName.MODEL: IOSpec(dtype=object),
        ModelOSName.REGULARIZATION_PARAMETER: IOSpec(dtype=float),
        ModelOSName.EPSILON: IOSpec(dtype=float),
        ModelOSName.INTERCEPT: IOSpec(dtype=float),
        ModelOSName.SUPPORT_VECTORS_COUNT: IOSpec(dtype=int),
        ModelOSName.SUPPORT_INDICES: IOSpec(dtype=List[int]),
        ModelOSName.DUAL_COEFFICIENTS: IOSpec(dtype=List[float]),
        ModelOSName.GAMMA: IOSpec(dtype=str),
    },
)
class SupportVectorRegressorModel(Task):
    @override
    def run(self):
        """
        Fits an SVR (RBF kernel) to a labeled Dataset and returns learned parameters and
        numeric hyperparameters in an AnalysisResult.
        """
        import numpy as np
        from sklearn.svm import SVR

        from auda.dat import run_pipeline

        c = float(self.get_input(ModelISName.REGULARIZATION_PARAMETER))
        epsilon = float(self.get_input(ModelISName.EPSILON))

        samples = self.get_input(ModelISName.SAMPLES)
        inlier_samples = self.get_input(ModelISName.INLIER_SAMPLES)
        if inlier_samples is not None:
            samples = inlier_samples

        # ---- Preprocessing: standardization
        outputs, _ = run_pipeline(
            ['TF-Z-NORM'],
            {
                TransformerISName.SAMPLES: samples,
                TransformerISName.TO_STANDARDIZE_Y: True,
            },
        )
        x_scaled: np.ndarray = outputs[TransformerOSName.X_STANDARDIZED]
        y_std: np.ndarray = outputs[TransformerOSName.Y_STANDARDIZED]

        # ---- Model fitting (support vector regression)
        svr_model = SVR(kernel='rbf', C=c, epsilon=epsilon, gamma='scale')
        svr_model.fit(x_scaled, y_std)
        support_indices = svr_model.support_.tolist()
        support_vectors_count = len(support_indices)
        dual_coefficients = (
            svr_model.dual_coef_.ravel().astype(float).tolist()  # type: ignore
        )
        intercept = float(svr_model.intercept_[0])
        gamma = svr_model.gamma

        # ---- Populate outputs
        self.set_output(ModelOSName.MODEL_TYPE, ModelType.SUPPORT_VECTOR_REGRESSION)
        self.set_output(ModelOSName.MODEL, svr_model)
        self.set_output(ModelOSName.REGULARIZATION_PARAMETER, c)
        self.set_output(ModelOSName.EPSILON, epsilon)
        self.set_output(ModelOSName.SUPPORT_VECTORS_COUNT, support_vectors_count)
        self.set_output(ModelOSName.SUPPORT_INDICES, support_indices)
        self.set_output(ModelOSName.INTERCEPT, float(intercept))
        self.set_output(ModelOSName.DUAL_COEFFICIENTS, dual_coefficients)
        self.set_output(ModelOSName.GAMMA, str(gamma))
