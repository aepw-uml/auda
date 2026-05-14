from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from common.files import save_content_to_file
from common.metrics.regression_metrics import (
    REGRESSION_METRIC_NAMES,
    RegressionMetricName,
    RegressionMetrics,
    average_regression_metrics,
    format_regression_metric_summary,
    get_regression_metric_label,
    std_regression_metrics,
)
from common.task import Task
from step.plot.plotter import Plotter
from util.names import to_snake
from util.table import Table


def save_metric_table(
    metrics_dict: dict[str, RegressionMetrics], dir_path: Path
) -> None:
    """Saves a metric table for the given metrics dictionary to a file.

    Args:
        metrics_dict: A dictionary mapping experiment names to their
            corresponding regression metrics.
        dir_path: The path to the directory where the metric table will be
            saved.
    """

    metric_table = Table(
        headers=['Experiment', 'MAE', 'RMSE', 'R²', 'WAPE', 'sMAPE']
    )
    for name, metrics in metrics_dict.items():
        [mae_str, rmse_str, r2_str, wape_str, smape_str] = metrics.item_strs()
        metric_table.append_row(
            name,
            mae_str,
            rmse_str,
            r2_str,
            wape_str,
            smape_str,
        )

    metric_table_path: Path = dir_path / 'metric_table'
    save_content_to_file(metric_table_path, metric_table.__repr__())

    print(f'Saved metric table to "{metric_table_path}".\n')
    print(metric_table.__repr__() + '\n')


def build_and_save_metric_table(
    task: Task,
    task_path: Path,
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
        for experiment in task.experiments
        if experiment.get_metrics() is not None
    }

    save_metric_table(metrics_dict, task_path)


def collect_metrics_by_name(
    tasks: list[Task],
) -> dict[str, list[RegressionMetrics]]:
    """Collects regression metrics across repeated tasks by experiment name.

    Args:
        tasks: Repeated task runs that contain compatible experiments.

    Returns:
        A dictionary mapping each experiment name to the metrics observed across
        repeated task runs.
    """

    metrics_by_name: dict[str, list[RegressionMetrics]] = {}
    for task in tasks:
        for experiment in task.experiments:
            metrics_by_name.setdefault(experiment.name, []).append(
                experiment.get_metrics()
            )

    return metrics_by_name


def summarize_metrics_by_name(
    metrics_by_name: dict[str, list[RegressionMetrics]],
) -> tuple[dict[str, RegressionMetrics], dict[str, RegressionMetrics]]:
    """Summarizes repeated metrics by experiment name.

    Args:
        metrics_by_name: Metrics collected by experiment name.

    Returns:
        A tuple containing dictionaries of mean metrics and sample standard
        deviations by experiment name.
    """

    mean_metrics: dict[str, RegressionMetrics] = {}
    std_metrics: dict[str, RegressionMetrics] = {}
    for name, metrics_list in metrics_by_name.items():
        mean_metrics[name] = average_regression_metrics(metrics_list)
        std_metrics[name] = std_regression_metrics(metrics_list)

    return mean_metrics, std_metrics


def save_metric_summary_plot(
    metrics_by_name: dict[str, list[RegressionMetrics]],
    dir_path: Path,
    metric: RegressionMetricName = 'wape',
) -> None:
    """Saves an error-bar plot of mean metric values by model.

    The plot shows the mean and sample standard deviation for one metric across
    repeated runs, keeping the metric table compact while preserving uncertainty
    information in a figure.

    Args:
        metrics_by_name: Metrics collected by experiment name.
        dir_path: Directory where the plot should be saved.
        metric: Metric to summarize in the plot.
    """

    if not metrics_by_name:
        return

    if metric not in REGRESSION_METRIC_NAMES:
        raise ValueError(f'Unknown metric for summary plot: {metric}.')

    names = list(metrics_by_name.keys())
    mean_metrics, std_metrics = summarize_metrics_by_name(metrics_by_name)
    means = np.array(
        [mean_metrics[name].get_value_by_name(metric) for name in names],
        dtype=float,
    )
    stds = np.array(
        [std_metrics[name].get_value_by_name(metric) for name in names],
        dtype=float,
    )

    plot_means = means.copy()
    plot_stds = stds.copy()
    x_label = get_regression_metric_label(metric)
    if metric in ('wape', 'smape'):
        plot_means *= 100
        plot_stds *= 100
        x_label = f'{x_label} (%)'

    y_positions = np.arange(len(names))
    fig_height = max(3.5, 0.45 * len(names) + 1.4)
    fig, ax = plt.subplots(figsize=(8.0, fig_height), dpi=200)
    ax.barh(
        y_positions,
        plot_means,
        xerr=plot_stds,
        color='steelblue',
        alpha=0.85,
        capsize=4,
    )

    span = float(np.max(np.abs(plot_means) + plot_stds))
    offset = max(span * 0.02, 0.05)
    for y_position, name, plot_mean, plot_std in zip(
        y_positions, names, plot_means, plot_stds
    ):
        summary = format_regression_metric_summary(
            metric,
            mean_metrics[name].get_value_by_name(metric),
            std_metrics[name].get_value_by_name(metric),
        )
        text_x = plot_mean + plot_std + offset
        horizontal_alignment = 'left'
        if plot_mean < 0:
            text_x = plot_mean - plot_std - offset
            horizontal_alignment = 'right'

        ax.text(
            text_x,
            y_position,  # type: ignore
            summary,
            va='center',
            ha=horizontal_alignment,
            fontsize=8,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_title(f'{get_regression_metric_label(metric)} Mean $\\pm$ SD')
    ax.grid(axis='x', alpha=0.25)

    dir_path.mkdir(parents=True, exist_ok=True)
    figure_path = dir_path / f'metric_summary_{metric}.png'
    fig.tight_layout()
    fig.savefig(figure_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f'Saved metric summary plot to "{figure_path}".')


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


def save_hyperparameter_table(task: Task, task_path: Path) -> None:
    """Saves a hyperparameter table for the given task to a file.

    Args:
        task: The task containing the experiments with their hyperparameters.
        task_path: The path to the task directory where the hyperparameter table
            will be saved.
    """

    hyperparameter_table = Table(headers=['Experiment', 'Hyperparameters'])
    for experiment in task.experiments:
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


def save_plots(task: Task, task_path: Path) -> None:
    plots_dir: Path = task_path / 'plots'
    for experiment in task.experiments:
        experiment.context['plot_title'] = ''
        plotter: Plotter | None = experiment.plot()
        if plotter is None:
            continue

        plot_path: Path = plots_dir / to_snake(experiment.name)
        file_path: str = plotter.save(plot_path)
        print(f'Saved plot for "{experiment.name}" to "{file_path}".')


def create_time_table(task: Task) -> dict[str, str]:
    """Creates a time table for the given task.

    Args:
        task: The task containing the experiments with their tuning times.
    """

    tuning_time_ms_dict: dict[str, str] = {}
    for experiment in task.experiments:
        if 'tuning_elapsed_ms' not in experiment.context:
            tuning_time_ms_dict[experiment.name] = 'N/A'
        else:
            elapsed_ms = experiment.context['tuning_elapsed_ms']
            tuning_time_ms_dict[experiment.name] = f'{elapsed_ms:.3f}'

    return tuning_time_ms_dict


def save_time_table(
    task: Task,
    task_path: Path,
) -> None:
    """Saves a time table for the given task to a file.

    Args:
        task: The task containing the experiments with their tuning times.
        task_path: The path to the task directory where the time table will
            be saved.
    """

    tuning_time_ms_dict: dict[str, str] = create_time_table(task)
    time_table = Table(headers=['Experiment', 'Tuning Time (ms)'])
    for name, tuning_time_ms in tuning_time_ms_dict.items():
        time_table.append_row(name, tuning_time_ms)

    time_table_path: Path = task_path / 'time_table'
    save_content_to_file(time_table_path, time_table.__repr__())

    print(f'Saved time table to "{time_table_path}".\n')
    print(time_table.__repr__() + '\n')
