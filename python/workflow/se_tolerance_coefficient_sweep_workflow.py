import csv
import math
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from statistics import mean
from typing import Callable, override

# Configure Matplotlib before importing pyplot in headless runs.
os.environ.setdefault('MPLCONFIGDIR', '/tmp/auda-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/auda-cache')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.dataset import Dataset, DatasetSchema
from common.files import save_content_to_file
from common.workflow import Workflow
from experiment.reconstruction_task import run_reconstruction_tasks
from step.evaluator.one_standard_error import standard_error
from step.tuner.types import Hyperparameters, Trial
from util.names import to_snake
from util.table import Table

DEFAULT_SE_TOLERANCE_COEFFICIENTS = [0.0, 0.1, 0.2, 0.5, 1.0]


@dataclass(frozen=True)
class TrialSummary:
    """Stores score and complexity details for one tuning trial."""

    hyperparameters: Hyperparameters
    mean_score: float
    standard_error: float
    complexity_rank: int


@dataclass(frozen=True)
class SelectionRecord:
    """Stores one selected hyperparameter configuration for a sweep value."""

    run_index: int
    experiment_name: str
    se_tolerance_coefficient: float
    selected_hyperparameters: Hyperparameters
    selected_mean_score: float
    selected_complexity_rank: int
    best_mean_hyperparameters: Hyperparameters
    best_mean_score: float
    changed_from_best_mean: bool


class SEToleranceCoefficientSweepWorkflow(Workflow):
    """Sweeps the standard-error tolerance coefficient."""

    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        num_experiments = int(context.get('num_experiments', '16'))
        seed = int(context.get('seed', '42'))
        coefficients = parse_se_tolerance_coefficients(
            context.get('se_tolerance_coefficients')
        )

        tasks, _ = run_reconstruction_tasks(
            num_experiments, dataset, schema, context, seed=seed
        )

        selection_records: list[SelectionRecord] = []
        trial_rows: list[dict[str, str]] = []
        for run_index, task in enumerate(tasks):
            for experiment in task.experiments:
                if 'trials' not in experiment.context:
                    continue

                tuning_parameters = experiment.context['tuning_parameters']
                hyperparameter_names = tuning_parameters['hyperparameter_names']
                complexity_key = tuning_parameters['complexity_key']
                trials = experiment.context['trials']
                trial_summaries = summarize_trials(
                    trials=trials,
                    complexity_key=complexity_key,
                )
                trial_rows.extend(
                    build_trial_rows(
                        run_index=run_index,
                        experiment_name=experiment.name,
                        hyperparameter_names=hyperparameter_names,
                        trials=trials,
                        trial_summaries=trial_summaries,
                    )
                )

                best_mean_record = select_trial(
                    trial_summaries=trial_summaries,
                    complexity_key=complexity_key,
                    se_tolerance_coefficient=0.0,
                )
                for coefficient in coefficients:
                    selected_record = select_trial(
                        trial_summaries=trial_summaries,
                        complexity_key=complexity_key,
                        se_tolerance_coefficient=coefficient,
                    )
                    selection_records.append(
                        SelectionRecord(
                            run_index=run_index,
                            experiment_name=experiment.name,
                            se_tolerance_coefficient=coefficient,
                            selected_hyperparameters=(
                                selected_record.hyperparameters
                            ),
                            selected_mean_score=selected_record.mean_score,
                            selected_complexity_rank=(
                                selected_record.complexity_rank
                            ),
                            best_mean_hyperparameters=(
                                best_mean_record.hyperparameters
                            ),
                            best_mean_score=best_mean_record.mean_score,
                            changed_from_best_mean=(
                                selected_record.hyperparameters
                                != best_mean_record.hyperparameters
                            ),
                        )
                    )

        location = to_snake(context.get('location', ''))
        dir_path = Path('results') / (
            'se_tolerance_coefficient_sweep'
            if not location
            else f'se_tolerance_coefficient_sweep_{location}'
        )
        save_selection_records(selection_records, dir_path)
        save_trial_rows(trial_rows, dir_path)
        save_summary_table(selection_records, dir_path)
        save_figure(selection_records, dir_path)


def parse_se_tolerance_coefficients(
    value: object,
) -> list[float]:
    """Parses the coefficient list from workflow context.

    Args:
        value: Optional comma-separated coefficient list.

    Returns:
        The coefficient values to evaluate.
    """

    if value is None:
        return DEFAULT_SE_TOLERANCE_COEFFICIENTS

    coefficients: list[float] = []
    for raw_value in str(value).split(','):
        normalized_value = raw_value.strip()
        if not normalized_value:
            continue

        coefficients.append(float(normalized_value))

    if not coefficients:
        raise ValueError(
            'se_tolerance_coefficients must include at least one value.'
        )

    return coefficients


def summarize_trials(
    trials: list[Trial],
    complexity_key: Callable[[Hyperparameters], tuple[float, ...]],
) -> list[TrialSummary]:
    """Summarizes trial WAPE values and complexity ranks.

    Args:
        trials: Tuning trials containing hyperparameters and fold metrics.
        complexity_key: Function that sorts hyperparameters from simpler to
            more complex.

    Returns:
        Trial summaries aligned with the input trials.
    """

    sorted_hyperparameters = sorted(
        [hyperparameters for hyperparameters, _ in trials],
        key=complexity_key,
    )
    complexity_ranks = {
        tuple(hyperparameters): rank
        for rank, hyperparameters in enumerate(sorted_hyperparameters, start=1)
    }

    trial_summaries: list[TrialSummary] = []
    for hyperparameters, metrics_list in trials:
        scores = [metrics.wape for metrics in metrics_list]
        trial_summaries.append(
            TrialSummary(
                hyperparameters=hyperparameters,
                mean_score=mean(scores),
                standard_error=standard_error(scores),
                complexity_rank=complexity_ranks[tuple(hyperparameters)],
            )
        )

    return trial_summaries


def select_trial(
    trial_summaries: list[TrialSummary],
    complexity_key: Callable[[Hyperparameters], tuple[float, ...]],
    se_tolerance_coefficient: float,
) -> TrialSummary:
    """Selects the simplest trial within the coefficient-adjusted threshold.

    Args:
        trial_summaries: Summaries of the evaluated trials.
        complexity_key: Function that sorts hyperparameters from simpler to
            more complex.
        se_tolerance_coefficient: Multiplier applied to the standard error of
            the best-mean trial.

    Returns:
        The selected trial summary.
    """

    if not trial_summaries:
        raise ValueError('No trials provided.')

    best_mean_trial = min(trial_summaries, key=lambda trial: trial.mean_score)
    threshold = (
        math.inf
        if math.isinf(se_tolerance_coefficient)
        else best_mean_trial.mean_score
        + se_tolerance_coefficient * best_mean_trial.standard_error
    )
    candidates = [
        trial for trial in trial_summaries if trial.mean_score <= threshold
    ]
    return min(
        candidates,
        key=lambda trial: complexity_key(trial.hyperparameters),
    )


def build_trial_rows(
    run_index: int,
    experiment_name: str,
    hyperparameter_names: list[str],
    trials: list[Trial],
    trial_summaries: list[TrialSummary],
) -> list[dict[str, str]]:
    """Builds CSV rows for trial-level WAPE scores.

    Args:
        run_index: Zero-based repeated reconstruction run index.
        experiment_name: Name of the tuned experiment.
        hyperparameter_names: Names corresponding to the hyperparameter vector.
        trials: Tuning trials containing fold metrics.
        trial_summaries: Summaries aligned with ``trials``.

    Returns:
        Trial-level CSV rows.
    """

    rows: list[dict[str, str]] = []
    for trial_index, ((hyperparameters, metrics_list), summary) in enumerate(
        zip(trials, trial_summaries)
    ):
        row = {
            'run_index': str(run_index),
            'experiment': experiment_name,
            'trial_index': str(trial_index),
            'mean_wape': format_float(summary.mean_score),
            'standard_error': format_float(summary.standard_error),
            'complexity_rank': str(summary.complexity_rank),
        }
        for name, value in zip(hyperparameter_names, hyperparameters):
            row[name] = format_float(value)
        for fold_index, metrics in enumerate(metrics_list):
            row[f'fold_{fold_index}_wape'] = format_float(metrics.wape)

        rows.append(row)

    return rows


def save_selection_records(
    selection_records: list[SelectionRecord],
    dir_path: Path,
) -> None:
    """Saves coefficient-level selections as CSV."""

    rows = [
        {
            'run_index': str(record.run_index),
            'experiment': record.experiment_name,
            'se_tolerance_coefficient': format_float(
                record.se_tolerance_coefficient
            ),
            'selected_hyperparameters': format_hyperparameters(
                record.selected_hyperparameters
            ),
            'selected_mean_wape': format_float(record.selected_mean_score),
            'selected_complexity_rank': str(record.selected_complexity_rank),
            'best_mean_hyperparameters': format_hyperparameters(
                record.best_mean_hyperparameters
            ),
            'best_mean_wape': format_float(record.best_mean_score),
            'changed_from_best_mean': str(record.changed_from_best_mean),
            'wape_penalty': format_float(
                record.selected_mean_score - record.best_mean_score
            ),
        }
        for record in selection_records
    ]
    save_csv(dir_path / 'selections.csv', rows)


def save_trial_rows(
    trial_rows: list[dict[str, str]],
    dir_path: Path,
) -> None:
    """Saves trial-level WAPE scores as CSV."""

    save_csv(dir_path / 'trials.csv', trial_rows)


def save_summary_table(
    selection_records: list[SelectionRecord],
    dir_path: Path,
) -> None:
    """Saves a summary table for the coefficient sweep."""

    records_by_group: dict[tuple[str, float], list[SelectionRecord]] = {}
    for record in selection_records:
        group_key = (
            record.experiment_name,
            record.se_tolerance_coefficient,
        )
        records_by_group.setdefault(group_key, []).append(record)

    table = Table(
        headers=[
            'Experiment',
            'c_SE',
            'Selection Units',
            'Changed',
            'Changed Rate',
            'Mean Best WAPE',
            'Mean Selected WAPE',
            'Mean WAPE Penalty',
            'Mean Complexity Rank',
        ]
    )
    for experiment_name, coefficient in sorted(records_by_group):
        records = records_by_group[(experiment_name, coefficient)]
        num_records = len(records)
        changed_count = sum(
            1 for record in records if record.changed_from_best_mean
        )
        mean_best_wape = mean(record.best_mean_score for record in records)
        mean_selected_wape = mean(
            record.selected_mean_score for record in records
        )
        mean_penalty = mean(
            record.selected_mean_score - record.best_mean_score
            for record in records
        )
        mean_complexity_rank = mean(
            record.selected_complexity_rank for record in records
        )
        table.append_row(
            experiment_name,
            format_coefficient(coefficient),
            num_records,
            changed_count,
            format_percentage(changed_count / num_records),
            format_percentage(mean_best_wape),
            format_percentage(mean_selected_wape),
            format_percentage(mean_penalty),
            f'{mean_complexity_rank:.2f}',
        )

    table_path = dir_path / 'summary_table'
    save_content_to_file(table_path, table.__repr__())
    print(f'Saved summary table to "{table_path}".\n')
    print(table.__repr__() + '\n')


def save_figure(
    selection_records: list[SelectionRecord],
    dir_path: Path,
) -> None:
    """Saves Figure 4 for the coefficient sweep."""

    records_by_group: dict[tuple[str, float], list[SelectionRecord]] = {}
    coefficients = sorted(
        {record.se_tolerance_coefficient for record in selection_records}
    )
    experiment_names = sorted(
        {record.experiment_name for record in selection_records}
    )
    for record in selection_records:
        records_by_group.setdefault(
            (
                record.experiment_name,
                record.se_tolerance_coefficient,
            ),
            [],
        ).append(record)

    fig, ax = plt.subplots(figsize=(10.0, 5.5), dpi=200)
    x_positions = list(range(len(coefficients)))
    bar_width = 0.8 / max(1, len(experiment_names))
    for experiment_index, experiment_name in enumerate(experiment_names):
        offsets = [
            x_position - 0.4 + bar_width / 2.0 + experiment_index * bar_width
            for x_position in x_positions
        ]
        changed_rates: list[float] = []
        for coefficient in coefficients:
            records = records_by_group[(experiment_name, coefficient)]
            changed_rates.append(
                100.0
                * sum(1 for record in records if record.changed_from_best_mean)
                / len(records)
            )

        ax.bar(
            offsets,
            changed_rates,
            width=bar_width,
            label=experiment_name,
        )

    ax.axhline(30.0, color='#555555', linewidth=1.0, linestyle='--')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [format_coefficient(coefficient) for coefficient in coefficients]
    )
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel('Standard-Error Tolerance Coefficient')
    ax.set_ylabel('Selections Changed from Best-Mean (%)')
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.35)
    ax.legend()
    fig.tight_layout()

    figure_path = dir_path / 'selection_sensitivity_to_se_tolerance.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved SE tolerance coefficient to "{figure_path}".')


def save_csv(file_path: Path, rows: list[dict[str, str]]) -> None:
    """Saves dictionary rows as a CSV file."""

    if not rows:
        save_content_to_file(file_path, '')
        return

    fieldnames: list[str] = []
    for row in rows:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    save_content_to_file(file_path, buffer.getvalue())


def format_hyperparameters(hyperparameters: Hyperparameters) -> str:
    """Formats hyperparameters for compact CSV output."""

    return ';'.join(format_float(value) for value in hyperparameters)


def format_coefficient(value: float) -> str:
    """Formats a coefficient value for display."""

    return 'inf' if math.isinf(value) else f'{value:g}'


def format_float(value: float) -> str:
    """Formats a float for stable text output."""

    return f'{value:.12g}'


def format_percentage(value: float) -> str:
    """Formats a ratio as a percentage."""

    return f'{100.0 * value:.2f}%'
