from typing import List, override

from auda.step import get_dataset_from_step
from auda.step.plot import PlotStep
from auda.step.spec import Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='PL-SP',
    description='plots scatter points of sample data.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.FEATURE_NAMES,
        Spec.LABEL_NAMES,
        Spec.FEATURE_UNITS,
        Spec.LABEL_UNITS,
        Spec.TITLE.optional(''),
    ],
    output_specs=[Spec.FIGURE],
)
class PlotScatterPlot(PlotStep):
    @override
    def run(
        self,
        on: str,
        feature_names: List[str],
        label_names: List[str],
        feature_units: List[str],
        label_units: List[str],
        title: str,
    ) -> IOValueMap:
        import matplotlib.pyplot as plt
        import seaborn as sns

        self.check_feature_dimension(expected_dimension=1)
        X, y = get_dataset_from_step(self, on)
        print(X.shape, y.shape)

        # ---- Create scatter plot
        figure, ax = plt.subplots()
        sns.set_theme(style='whitegrid')
        sns.scatterplot(x=X.ravel(), y=y.ravel(), ax=ax)  # type: ignore

        feature_labels = self.combine_names_units(feature_names, feature_units)
        label_labels = self.combine_names_units(label_names, label_units)
        ax.set_xlabel(feature_labels[0])
        ax.set_ylabel(label_labels[0])
        ax.set_title(title)

        return {Spec.FIGURE.name: figure}
