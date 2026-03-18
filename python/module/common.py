from pathlib import Path

from common.files import save_content_to_file
from common.names import to_kebab
from experiment.experiment_group import ExperimentGroup
from step.plot.mrs_best_so_far import MrsBestSoFar
from step.tuner.random_search import HyperparameterScore
from util.table import Table


def create_time_table(group: ExperimentGroup) -> dict[str, str]:
    tuning_time_ms_dict: dict[str, str] = {}
    for experiment in group.experiments:
        if 'tuning_elapsed_ms' not in experiment.context:
            tuning_time_ms_dict[experiment.name] = 'N/A'
        else:
            elapsed_ms = experiment.context['tuning_elapsed_ms']
            tuning_time_ms_dict[experiment.name] = f'{elapsed_ms:.3f}'

    return tuning_time_ms_dict


def save_time_table(
    module_path: Path, tuning_time_ms_dict: dict[str, str]
) -> None:
    time_table = Table(headers=['Experiment', 'Tuning Time (ms)'])
    for name, tuning_time_ms in tuning_time_ms_dict.items():
        time_table.append_row(name, tuning_time_ms)

    time_table_path: Path = module_path / 'time_table'
    save_content_to_file(time_table_path, time_table.__repr__())
    print(f'Saved time table to "{time_table_path}".')
    print()
    print(time_table.__repr__())
    print()


def save_best_so_far_plots(module_path: Path, group: ExperimentGroup) -> None:
    for experiment in group.experiments:
        if 'hyperparameter_scores_lists' not in experiment.context:
            continue

        hyperparameter_scores_lists: list[list[HyperparameterScore]] = (
            experiment.context['hyperparameter_scores_lists']
        )

        plotter = MrsBestSoFar(hyperparameter_scores_lists)
        plotter.plot()

        best_so_far_dir: Path = module_path / 'best_so_far'
        plot_path: Path = best_so_far_dir / f'{to_kebab(experiment.name)}.png'
        plotter.save(plot_path)
        print(
            f'Saved best-so-far plot for "{experiment.name}" to "{plot_path}".'
        )
