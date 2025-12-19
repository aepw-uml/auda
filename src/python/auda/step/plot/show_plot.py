from auda.step.spec import Spec
from auda.utils.pipeline import Step, step
from matplotlib.figure import Figure


@step(
    id='PL-SHOW',
    description='Displays the generated figure interactively.',
    input_specs=[Spec.FIGURE],
)
class ShowPlot(Step):
    def run(self, figure: Figure) -> None:
        from matplotlib import pyplot as plt

        try:
            plt.figure(figure.number)
        except Exception:
            pass

        self.pipeline.schedule(lambda _: plt.show())
