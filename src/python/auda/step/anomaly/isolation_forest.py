from typing import override

import numpy as np
from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='AD-IF',
    description='Trains an Isolation Forest model for anomaly detection.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.ISOLATION_FOREST_THRESHOLD.optional(0.99),
    ],
    output_specs=[
        Spec.INLIER_INDEXES,
        Spec.OUTLIER_INDEXES,
        Spec.INLIER_DATASET,
        Spec.ANOMALY_SCORES,
        Spec.CONTAMINATION_RATE,
    ],
)
class IsolationForest(DatasetBasedStep):
    @override
    def run(
        self, on: str | Dataset, isolation_forest_threshold: float
    ) -> IOValueMap:
        from sklearn.ensemble import IsolationForest

        X, y = self.get_dataset_from_on(on)
        num_samples = len(X)
        # if num_samples < 15:
        #     return {
        #         Spec.INLIER_INDEXES.name: list(range(num_samples)),
        #         Spec.OUTLIER_INDEXES.name: [],
        #         Spec.INLIER_DATASET.name: (X, y),
        #         Spec.ANOMALY_SCORES.name: [0.0] * num_samples,
        #         Spec.CONTAMINATION_RATE.name: 0.0,
        #     }

        data = np.concatenate([X, y.reshape(-1, 1)], axis=1)

        # ---- Train an isolation forest model
        isolation_forest = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination='auto',
            random_state=42,
            n_jobs=-1,
        )
        isolation_forest.fit(data)

        # The values are inverted: higher scores indicate more likely to be
        # anomalies; if the threshold is 0.99, the top 1% samples with the
        # highest scores are labeled as outliers
        scores = -isolation_forest.score_samples(data)
        threshold = np.quantile(scores, isolation_forest_threshold)
        pred = np.where(scores >= threshold, -1, 1)

        # ---- Separate inliers and outliers
        inlier_indexes = [i for i, y in enumerate(pred) if y == 1]
        outlier_indexes = [i for i, y in enumerate(pred) if y == -1]
        inlier_dataset = X[inlier_indexes], y[inlier_indexes]

        # ---- Normalize anomaly scores to [0, 1] for easier interpretation
        raw_scores = -isolation_forest.score_samples(data)
        range_ = np.ptp(raw_scores)
        if range_ > 0:
            normalized_scores = (raw_scores - raw_scores.min()) / range_
        else:
            normalized_scores = np.zeros_like(raw_scores)

        return {
            Spec.INLIER_INDEXES.name: inlier_indexes,
            Spec.OUTLIER_INDEXES.name: outlier_indexes,
            Spec.INLIER_DATASET.name: inlier_dataset,
            Spec.ANOMALY_SCORES.name: normalized_scores.tolist(),
            Spec.CONTAMINATION_RATE.name: (pred == -1).mean(),
        }
