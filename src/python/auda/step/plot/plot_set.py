from typing import override

from auda.step.dataset import DatasetSchema
from auda.step.plot import PlotStep
from auda.step.spec import Spec
from auda.utils.pipeline import step


@step(
    id='PL-SET',
    description='Sets plot properties such as labels and title.',
    input_specs=[
        Spec.FIGURE.optional(),
        Spec.AXES.optional(),
        Spec.DATASET_SCHEMA.optional(),
        Spec.TITLE.optional(),
        Spec.GRID_ALPHA.optional(),
        Spec.LEGEND_LOCATION.optional(),
        Spec.TIGHT_LAYOUT.optional(True),
    ],
    output_specs=[Spec.FIGURE, Spec.AXES],
)
class PlotSet(PlotStep):
    @override
    def run(
        self,
        figure,
        axes,
        dataset_schema: DatasetSchema | None,
        title: str | None,
        grid_alpha: float | None,
        legend_location: str | None,
        tight_layout: bool,
    ) -> dict:
        figure, axes = self.create_plot_or_default(figure, axes)

        # ---- Set labels
        # TODO: generalize to 3D plots
        if dataset_schema is not None:
            feature_names = dataset_schema.feature_names
            feature_units = dataset_schema.feature_units
            label_names = dataset_schema.label_names
            label_units = dataset_schema.label_units

            feature_labels = self.combine_names_units(
                feature_names, feature_units
            )
            axes.set_xlabel(feature_labels[0])

            if label_names and label_units:
                label_labels = self.combine_names_units(
                    label_names, label_units
                )
                axes.set_ylabel(label_labels[0])

        if title:
            axes.set_title(title)

        # ---- Set grid
        if grid_alpha:
            axes.grid(True, alpha=grid_alpha)

        # ---- Set legend
        if legend_location:
            axes.legend(loc=legend_location)

        # ---- Adjust layout
        if tight_layout:
            figure.tight_layout()

        return self.regular_output(figure, axes)
