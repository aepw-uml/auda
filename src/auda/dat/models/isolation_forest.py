from typing import List

from auda.dat.datasets import LabeledSamples
from auda.utils.pipeline import IOSpec, Task, task

from .__common import MODEL_KIND, ModelISName, ModelOSName


@task(
    id='MD-IF',
    kind=MODEL_KIND,
    description='Trains an Isolation Forest model for anomaly detection.',
    input_specs={
        ModelISName.SAMPLES: IOSpec(dtype=LabeledSamples),
    },
    output_specs={
        ModelOSName.SAMPLES: IOSpec(dtype=LabeledSamples),
        ModelOSName.INLIER_SAMPLES: IOSpec(dtype=LabeledSamples),
        ModelOSName.OUTLIER_INDICES: IOSpec(dtype=List[int]),
        ModelOSName.INLIER_INDICES: IOSpec(dtype=List[int]),
        ModelOSName.ANOMALY_SCORES: IOSpec(dtype=List[float]),
        ModelOSName.CONTAINATION_RATE: IOSpec(dtype=float),
    },
)
class IsolationForestModel(Task):
    def run(self) -> None:
        """
        Train an Isolation Forest model for anomaly detection.
        """
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        samples: LabeledSamples = self.get_input(ModelISName.SAMPLES)
        data = np.array([[*sample[0], sample[1]] for sample in samples])
        data_standardized = StandardScaler().fit_transform(data)

        # ---- Fit IsolationForest
        isolation_forest = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination='auto',
            random_state=42,
            n_jobs=-1,
        )
        isolation_forest.fit(data_standardized)

        # ---- Predict Anomalies ----
        # Note that the values are inverted: higher scores indicate more likely to be
        # anomalies
        scores = -isolation_forest.score_samples(data_standardized)

        # ---- Determine threshold for anomalies (top 10% as anomalies)
        threshold = np.quantile(scores, 0.90)

        # ---- Generate predictions based on the threshold
        pred = np.where(scores >= threshold, -1, 1)

        # ---- Collect indices of outliers and inliers
        outlier_indices = [i for i, y in enumerate(pred) if y == -1]
        inlier_indices = [i for i, y in enumerate(pred) if y == 1]

        # ---- Normalize anomaly scores to [0, 1] for easier interpretation
        raw_scores = -isolation_forest.score_samples(data_standardized)
        s_min, s_max = float(raw_scores.min()), float(raw_scores.max())
        if s_max > s_min:
            normalized_scores = (raw_scores - s_min) / (s_max - s_min)
        else:
            # Edge case: all scores are identical
            normalized_scores = np.zeros_like(raw_scores)

        # ---- Build the modified dataset: keep only inliers ----
        inlier_samples: LabeledSamples = [samples[i] for i in inlier_indices]

        # The contamination rate refers to the percentage of data points labeled as
        # outliers
        contamination_rate = float((pred == -1).mean())

        # ---- Populate outputs
        self.set_output(ModelOSName.SAMPLES, inlier_samples)
        self.set_output(ModelOSName.INLIER_SAMPLES, inlier_samples)
        self.set_output(ModelOSName.OUTLIER_INDICES, outlier_indices)
        self.set_output(ModelOSName.INLIER_INDICES, inlier_indices)
        self.set_output(ModelOSName.ANOMALY_SCORES, normalized_scores)
        self.set_output(ModelOSName.CONTAINATION_RATE, contamination_rate)
