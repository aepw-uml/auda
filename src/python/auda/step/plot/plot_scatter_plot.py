from typing import override

from auda.step.dataset import DatasetBasedStep
from auda.step.plot import PlotStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='PL-SP',
    description='Plots scatter points of sample data.',
    input_specs=[
        Spec.FIGURE.optional(),
        Spec.AXES.optional(),
        Spec.ON.optional(Spec.DATASET.name),
        Spec.SAMPLE_POINT_SIZE.optional(60),
        Spec.SAMPLE_POINT_COLOR.optional('cornflowerblue'),
        Spec.SAMPLE_POINT_EDGE_COLOR.optional('white'),
        Spec.SAMPLE_POINT_LABEL.optional('Samples'),
    ],
    output_specs=[Spec.FIGURE, Spec.AXES],
)
class PlotScatterPlot(PlotStep, DatasetBasedStep):
    @override
    def run(
        self,
        on: str | Dataset,
        figure,
        axes,
        sample_point_size: float,
        sample_point_color: str,
        sample_point_edge_color: str,
        sample_point_label: str,
    ) -> IOValueMap:
        X, y = self.get_dataset_from_on(on)
        self.check_feature_dimension(X, expected_dimension=1)

        # ---- Create scatter plot
        figure, axes = self.create_plot_or_default(figure, axes)
        axes.scatter(
            X.ravel(),
            y.ravel(),
            s=int(sample_point_size),
            zorder=1,
            label=sample_point_label,
            color=sample_point_color,
            edgecolor=sample_point_edge_color,
        )

        # ---- Set labels
        # if feature_names and feature_units and label_names and label_units:
        #     feature_labels = self.combine_names_units(
        #         feature_names, feature_units
        #     )
        #     label_labels = self.combine_names_units(label_names, label_units)
        #     axes.set_xlabel(feature_labels[0])
        #     axes.set_ylabel(label_labels[0])

        # if title:
        #     axes.set_title(title)

        return self.regular_output(figure, axes)
