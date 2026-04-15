from pathlib import Path
from typing import Any

from common.experiment.experiment_group import ExperimentGroup
from common.files import save_content_to_file
from common.metrics.regression_metrics import RegressionMetrics
from step.plot.plotter import Plotter
from util.names import to_kebab
from util.table import Table


def save_metric_table(
    metrics_dict: dict[str, RegressionMetrics], task_path: Path
) -> None:
    """Saves a metric table for the given metrics dictionary to a file.

    Args:
        metrics_dict: A dictionary mapping experiment names to their
            corresponding regression metrics.
        task_path: The path to the task directory where the metric table will be
            saved.
    """

    metric_table = Table(
        headers=['Experiment', 'MAE', 'RMSE', 'R²', 'MAPE', 'WAPE']
    )
    for name, metrics in metrics_dict.items():
        [mae_str, rmse_str, r2_str, mape_str, wape_str] = metrics.item_strs()
        metric_table.append_row(
            name,
            mae_str,
            rmse_str,
            r2_str,
            mape_str,
            wape_str,
        )

    metric_table_path: Path = task_path / 'metric_table'
    save_content_to_file(metric_table_path, metric_table.__repr__())

    print(f'Saved metric table to "{metric_table_path}".\n')
    print(metric_table.__repr__() + '\n')


def build_and_save_metric_table(
    group: ExperimentGroup, task_path: Path
) -> None:
    """Builds a metric table from the given metrics dictionary and saves it to
    a file.

    Args:
        metrics_dict: A dictionary mapping experiment names to their
            corresponding regression metrics.
        task_path: The path to the task directory where the metric table will be
            saved.
    """

    metrics_dict: dict[str, RegressionMetrics] = {
        experiment.name: experiment.get_metrics()
        for experiment in group.experiments
        if experiment.get_metrics() is not None
    }

    save_metric_table(metrics_dict, task_path)


def get_hyperparameters_str(hyperparameters: dict[str, Any]) -> str:
    """Formats the hyperparameters into a string for display.

    Args:
        hyperparameters: A dictionary of hyperparameters to format.

    Returns:
        A string representation of the hyperparameters.
    """

    items: list[str] = []
    for name, value in hyperparameters.items():
        if isinstance(value, float):
            items.append(f'{name}={value:.3e}')
        elif isinstance(value, list):
            items.append(f'{name}=[{", ".join(str(v) for v in value)}]')
        else:
            items.append(f'{name}={value}')

    return ', '.join(items)


def save_hyperparameter_table(group: ExperimentGroup, task_path: Path) -> None:
    """Saves a hyperparameter table for the given experiment group to a file.

    Args:
        group: The experiment group containing the experiments with their
            hyperparameters.
        task_path: The path to the task directory where the hyperparameter table
            will be saved.
    """

    hyperparameter_table = Table(headers=['Experiment', 'Hyperparameters'])
    for experiment in group.experiments:
        hyerparameters_str = get_hyperparameters_str(experiment.hyperparameters)
        hyperparameter_table.append_row(
            experiment.name,
            hyerparameters_str if hyerparameters_str else '-',
        )
    hyperparameter_table_path: Path = task_path / 'hyperparameter_table'
    save_content_to_file(
        hyperparameter_table_path,
        hyperparameter_table.__repr__(),
    )

    print(f'Saved hyperparameter table to "{hyperparameter_table_path}".\n')
    print(hyperparameter_table.__repr__() + '\n')


def save_plots(group: ExperimentGroup, task_path: Path) -> None:
    plots_dir: Path = task_path / 'plots'
    for experiment in group.experiments:
        experiment.context['plot_title'] = ''
        plotter: Plotter | None = experiment.plot()
        if plotter is None:
            continue

        plot_path: Path = plots_dir / to_kebab(experiment.name)
        file_path: str = plotter.save(plot_path)
        print(f'Saved plot for "{experiment.name}" to "{file_path}".')


def create_time_table(group: ExperimentGroup) -> dict[str, str]:
    """Creates a time table for the given experiment group.

    Args:
        group: The experiment group containing the experiments with their tuning
            times.
    """

    tuning_time_ms_dict: dict[str, str] = {}
    for experiment in group.experiments:
        if 'tuning_elapsed_ms' not in experiment.context:
            tuning_time_ms_dict[experiment.name] = 'N/A'
        else:
            elapsed_ms = experiment.context['tuning_elapsed_ms']
            tuning_time_ms_dict[experiment.name] = f'{elapsed_ms:.3f}'

    return tuning_time_ms_dict


def save_time_table(
    group: ExperimentGroup,
    task_path: Path,
) -> None:
    """Saves a time table for the given experiment group to a file.

    Args:
        group: The experiment group containing the experiments with their tuning
            times.
        task_path: The path to the task directory where the time table will
            be saved.
    """

    tuning_time_ms_dict: dict[str, str] = create_time_table(group)
    time_table = Table(headers=['Experiment', 'Tuning Time (ms)'])
    for name, tuning_time_ms in tuning_time_ms_dict.items():
        time_table.append_row(name, tuning_time_ms)

    time_table_path: Path = task_path / 'time_table'
    save_content_to_file(time_table_path, time_table.__repr__())

    print(f'Saved time table to "{time_table_path}".\n')
    print(time_table.__repr__() + '\n')
