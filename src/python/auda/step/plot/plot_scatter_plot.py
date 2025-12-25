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
        Spec.Z_ORDER.optional(1),
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
        z_order: int,
        sample_point_size: float,
        sample_point_color: str,
        sample_point_edge_color: str,
        sample_point_label: str,
    ) -> IOValueMap:
        X, y = self.get_dataset_from_on(on)
        self.check_feature_dimension(X, expected_dimension=1)

        figure, axes = self.create_plot_or_default(figure, axes)
        axes.scatter(
            X.ravel(),
            y.ravel(),
            s=int(sample_point_size),
            zorder=z_order,
            label=sample_point_label,
            color=sample_point_color,
            edgecolor=sample_point_edge_color,
        )

        return self.regular_output(figure, axes)
