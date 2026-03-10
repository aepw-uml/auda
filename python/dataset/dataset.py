from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSchema:
    """Schema for dataset.

    Attributes:
        feature_names: List of feature names.
        label_names: List of label names.
        feature_units: List of feature units.
        label_units: List of label units.
        class_names: List of class names (for classification models).
    """

    # Regression models
    feature_names: list[str]
    feature_units: list[str]
    label_names: list[str] | None = None
    label_units: list[str] | None = None

    # Classification models
    class_names: list[str] | None = None
