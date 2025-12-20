from typing import override

from auda.step import get_dataset_from_step
from auda.step.plot import PlotStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='PL-Curve',
    description='Plots a curve plot from given x and y data.',
    input_specs=[
        Spec.FIGURE.optional(),
        Spec.AXES.optional(),
        Spec.ON,
        Spec.LINE_COLOR.optional('orange'),
        Spec.LINE_WIDTH.optional(2.5),
        Spec.LINE_LABEL.optional('Fitting Curve'),
        Spec.LINE_STYLE.optional('solid'),
    ],
    output_specs=[Spec.FIGURE, Spec.AXES],
)
class PlotCurve(PlotStep):
    @override
    def run(
        self,
        figure,
        axes,
        on: str | Dataset,
        line_color: str,
        line_width: float,
        line_label: str,
        line_style: str,
    ) -> IOValueMap:
        figure, axes = self.create_plot_or_default(figure, axes)
        x_curve, y_curve = get_dataset_from_step(self, on)

        axes.plot(
            x_curve,
            y_curve,
            color=line_color,
            lw=line_width,
            label=line_label,
            linestyle=line_style,
            zorder=3,
        )

        return self.regular_output(figure, axes)
