from typing import override

from auda.utils.pipeline import IOSpec, Task, task

from .__common import PLOTTER_KIND, PlotterISName


@task(
    id='SHOW',
    kind=PLOTTER_KIND,
    description='Displays the generated figure interactively.',
    input_specs={
        PlotterISName.FIGURE: IOSpec(dtype=object),
    },
)
class ShowFigurePlotter(Task):
    """
    This task shows a Matplotlib figure in the context.
    """

    @override
    def run(self) -> None:
        from matplotlib import pyplot as plt

        figure = self.get_input(PlotterISName.FIGURE)
        if figure is not None:
            try:
                plt.figure(figure.number)
            except Exception:
                pass

        self.pipeline.schedule(lambda: plt.show())
