from typing import override

from auda.core import auda
from auda.step.spec import Spec
from auda.utils.pipeline import Step, step


@step(
    id='PL-SAVE-FIGURE',
    description='Saves the generated figure to a file.',
    input_specs=[
        Spec.FIGURE,
    ],
)
class SaveFigurePlotter(Step):
    @override
    def run(self, figure, path: str, dpi: int, transparent: bool) -> None:
        import os

        figure_path = auda.results_dir / path
        os.makedirs(os.path.dirname(figure_path) or '.', exist_ok=True)

        figure.savefig(
            figure_path,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=transparent,
        )
