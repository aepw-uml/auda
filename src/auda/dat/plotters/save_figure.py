from typing import override

from auda.core import project
from auda.utils.pipeline import IOSpec, Task, task

from .__common import PLOTTER_KIND, PlotterISName


@task(
    id='SAVE',
    kind=PLOTTER_KIND,
    description='Saves a generated figure to a file.',
    input_specs={
        PlotterISName.FIGURE: IOSpec(dtype=object),
        PlotterISName.SAVE_PATH: IOSpec(dtype=str),
        PlotterISName.SAVE_DPI: IOSpec(dtype=int, required=False, default=200),
        PlotterISName.SAVE_TRANSPARENT: IOSpec(
            dtype=bool, required=False, default=False
        ),
    },
)
class SaveFigurePlotter(Task):
    """
    This task saves a Matplotlib figure to a file.
    """

    @override
    def run(self) -> None:
        import os

        fig = self.get_input(PlotterISName.FIGURE)
        path = str(self.get_input(PlotterISName.SAVE_PATH))
        dpi = self.get_input(PlotterISName.SAVE_DPI)
        transparent = bool(self.get_input(PlotterISName.SAVE_TRANSPARENT))

        print(path)

        figure_path = project.results_dir / path
        os.makedirs(os.path.dirname(figure_path) or '.', exist_ok=True)
        fig.savefig(
            figure_path,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=transparent,
        )
