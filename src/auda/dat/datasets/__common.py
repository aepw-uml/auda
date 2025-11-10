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
UnlabeledUnits = List[Unit]
LabeledUnits = Tuple[List[Unit], Unit]
Units = UnlabeledUnits | LabeledUnits
