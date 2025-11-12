from typing import List, Optional, Tuple, TypeVar


class DatasetISName:
    LOCATION = 'location'
    YEAR = 'year'


class DatasetOSName:
    """
    Dataset output sepc names.
    """

    SAMPLES = 'samples'
    LABEL = 'label'
    FEATURE_NAMES = 'feature_names'
    UNITS = 'units'

    # Original samples before any processing
    ORIGINAL_SAMPLES = 'original_samples'


DATASET_KIND = 'dataset'

Label = TypeVar('Label', float, int, str)

UnlabeledSample = List[float]
LabeledSample = Tuple[List[float], Label]

UnlabeledSamples = List[UnlabeledSample]
LabeledSamples = List[LabeledSample]
Samples = UnlabeledSamples | LabeledSamples

Unit = Optional[str]
Units = List[Unit]


def verify_feature_space_dimension(
    samples: Samples,
    expected_dimension: int,
) -> None:
    """
    Verify that the feature space dimension of the samples matches the expected
    dimension.

    Args:
        samples: The samples to verify.
        expected_dimension: The expected feature space dimension.

    Raises:
        ValueError: If the feature space dimension does not match the expected
        dimension.
    """
    if not samples:
        return

    actual_dimension = (
        len(samples[0][0]) if isinstance(samples[0], tuple) else len(samples[0])
    )
    if actual_dimension != expected_dimension:
        raise ValueError(
            f'Expected feature space dimension {expected_dimension}, '
            f'but got {actual_dimension}.'
        )


def get_feature_label(feature_name: str, unit: str | None) -> str:
    return feature_name if unit is None else f'{feature_name} ({unit})'
