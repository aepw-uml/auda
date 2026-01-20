from typing import cast, override

import numpy as np
from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, LabeledDataset, Spec, UnlabeledDataset
from auda.utils.pipeline import IOValueMap, step


@step(
    id='PP-SORT',
    description='Sorts a dataset based on a specified feature or label index.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.SORT_BY_FEATURE_INDEX.optional(-1),
        Spec.SORT_BY_LABEL_INDEX.optional(-1),
        Spec.SORT_ASCENDINGLY.optional(True),
    ],
    output_specs=[
        Spec.SORTED_DATASET,
    ],
)
class SORT(DatasetBasedStep):
    @override
    def run(
        self,
        on: str | Dataset,
        sort_by_feature_index: int,
        sort_by_label_index: int,
        sort_ascendingly: bool,
    ) -> IOValueMap:
        dataset: Dataset = self.get_dataset_from_on(on)
        num_samples: int = self.get_num_samples(dataset)
        is_labeled: bool = self.is_dataset_labeled(dataset)

        if is_labeled:
            X, y = cast(LabeledDataset, dataset)
            is_y_1d = len(y.shape) == 1
            num_features = X.shape[1]
            Xy = np.hstack((X, y.reshape(num_samples, 1)))

            if sort_by_feature_index < 0:
                if sort_by_label_index < 0:
                    raise ValueError(
                        'Either sort_by_feature_index or '
                        + 'sort_by_label_index must be non-negative.'
                    )
                elif sort_by_label_index >= y.shape[1]:
                    raise ValueError(
                        f'sort_by_label_index {sort_by_label_index} is out '
                        + f'of bounds for dataset with {y.shape[1]} labels.'
                    )
                column_index = num_features + sort_by_label_index
            elif sort_by_feature_index >= num_features:
                raise ValueError(
                    f'sort_by_feature_index {sort_by_feature_index} is out '
                    + f'of bounds for dataset with {num_features} features.'
                )
            else:
                column_index = sort_by_feature_index

            sorted_Xy = self._sort_np_array_by_column(
                Xy, column_index, sort_ascendingly
            )

            sorted_X, sorted_y = np.split(sorted_Xy, [-1], axis=1)
            if is_y_1d:
                sorted_y = sorted_y.ravel()

            sorted_dataset = (sorted_X, sorted_y)
        else:
            # Unlabeled dataset
            X = cast(UnlabeledDataset, dataset)
            num_samples = len(X)

            # In this branch, "sort_by_label_index" is ignored.
            if (
                sort_by_feature_index < 0
                or sort_by_feature_index >= num_samples
            ):
                raise ValueError(
                    f'sort_by_feature_index {sort_by_feature_index} is out '
                    + f'of bounds for dataset with {num_samples} features.'
                )

            column_index = sort_by_feature_index
            sorted_dataset = self._sort_np_array_by_column(
                X, column_index, sort_ascendingly
            )

        return {Spec.SORTED_DATASET.name: sorted_dataset}

    def _sort_np_array_by_column(
        self,
        array: np.ndarray,
        column_index: int,
        ascending: bool,
    ) -> np.ndarray:
        """Sort a 2D numpy array by a specific column.

        Args:
            array (np.ndarray): The input 2D numpy array to be sorted.
            column_index (int): The index of the column to sort by.
            ascending (bool): Whether to sort in ascending order.

        Returns:
            np.ndarray: The sorted 2D numpy array.
        """

        sorting_function = array[:, column_index].argsort()
        if not ascending:
            sorting_function = sorting_function[::-1]

        return array[sorting_function]
