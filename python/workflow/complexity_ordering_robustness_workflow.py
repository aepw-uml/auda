from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, override

import matplotlib.pyplot as plt
from common.dataset import Dataset, DatasetSchema
from common.files import save_content_to_file
from common.workflow import Workflow
from experiment.imputation_task import run_imputation_tasks
from step.tuner.types import Hyperparameters
from util.names import to_snake
from util.table import Table

from workflow.se_tolerance_coefficient_sweep_workflow import (
    format_float,
    format_hyperparameters,
    format_percentage,
    save_csv,
    select_candidates,
    select_trial,
    summarize_trials,
)

DEFAULT_SE_TOLERANCE_COEFFICIENT = 0.01


@dataclass(frozen=True)
class ComplexityOrderingRecord:
    """Stores one original-vs-flipped ordering comparison."""

    run_index: int
    experiment_name: str
    se_tolerance_coefficient: float
    num_candidate_configurations: int
    original_hyperparameters: Hyperparameters
    flipped_hyperparameters: Hyperparameters
    original_mean_wape: float
    flipped_mean_wape: float
    original_complexity_rank: int
    flipped_complexity_rank: int
    changed_by_flipped_ordering: bool

    @property
    def wape_degradation(self) -> float:
        """Returns the WAPE increase from using the flipped ordering."""

        return self.flipped_mean_wape - self.original_mean_wape


class ComplexityOrderingRobustnessWorkflow(Workflow):
    """Tests whether the complexity ordering direction affects selection."""

    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        num_experiments = int(context.get('num_experiments', '16'))
        seed = int(context.get('seed', '42'))
        se_tolerance_coefficient = float(
            context.get(
                'se_tolerance_coefficient',
                DEFAULT_SE_TOLERANCE_COEFFICIENT,
            )
        )

        tasks, _ = run_imputation_tasks(
            num_experiments, dataset, schema, context, seed=seed
        )

        records: list[ComplexityOrderingRecord] = []
        for run_index, task in enumerate(tasks):
            for experiment in task.experiments:
                if 'trials' not in experiment.context:
                    continue

                tuning_parameters = experiment.context['tuning_parameters']
                complexity_key = tuning_parameters['complexity_key']
                flipped_complexity_key = flip_complexity_key(complexity_key)
                trials = experiment.context['trials']

                original_summaries = summarize_trials(
                    trials=trials,
                    complexity_key=complexity_key,
                )
                flipped_summaries = summarize_trials(
                    trials=trials,
                    complexity_key=flipped_complexity_key,
                )
                original_selection = select_trial(
                    trial_summaries=original_summaries,
                    complexity_key=complexity_key,
                    se_tolerance_coefficient=se_tolerance_coefficient,
                )
                flipped_selection = select_trial(
                    trial_summaries=flipped_summaries,
                    complexity_key=flipped_complexity_key,
                    se_tolerance_coefficient=se_tolerance_coefficient,
                )
                # The candidate set is fixed by the mean-score threshold and is
                # independent of the ordering direction, so both keys share it.
                candidates = select_candidates(
                    trial_summaries=original_summaries,
                    se_tolerance_coefficient=se_tolerance_coefficient,
                )
                records.append(
                    ComplexityOrderingRecord(
                        run_index=run_index,
                        experiment_name=experiment.name,
                        se_tolerance_coefficient=se_tolerance_coefficient,
                        num_candidate_configurations=len(candidates),
                        original_hyperparameters=(
                            original_selection.hyperparameters
                        ),
                        flipped_hyperparameters=(
                            flipped_selection.hyperparameters
                        ),
                        original_mean_wape=original_selection.mean_score,
                        flipped_mean_wape=flipped_selection.mean_score,
                        original_complexity_rank=(
                            original_selection.complexity_rank
                        ),
                        flipped_complexity_rank=(
                            flipped_selection.complexity_rank
                        ),
                        changed_by_flipped_ordering=(
                            original_selection.hyperparameters
                            != flipped_selection.hyperparameters
                        ),
                    )
                )

        location = to_snake(context.get('location', ''))
        dir_path = Path('results') / (
            'complexity_ordering_robustness'
            if not location
            else f'complexity_ordering_robustness_{location}'
        )
        save_records(records, dir_path)
        save_summary_table(records, dir_path)
        save_figure(records, dir_path)


def flip_complexity_key(
    complexity_key: Callable[[Hyperparameters], tuple[float, ...]],
) -> Callable[[Hyperparameters], tuple[float, ...]]:
    """Builds a complexity key with every ordering sign flipped.

    Args:
        complexity_key: Original complexity key from Table 1.

    Returns:
        A complexity key with every tuple component multiplied by ``-1``.
    """

    def flipped_complexity_key(
        hyperparameters: Hyperparameters,
    ) -> tuple[float, ...]:
        return tuple(-value for value in complexity_key(hyperparameters))

    return flipped_complexity_key


def save_records(
    records: list[ComplexityOrderingRecord],
    dir_path: Path,
) -> None:
    """Saves original-vs-flipped selections as CSV."""

    rows = [
        {
            'run_index': str(record.run_index),
            'experiment': record.experiment_name,
            'se_tolerance_coefficient': format_float(
                record.se_tolerance_coefficient
            ),
            'num_candidate_configurations': str(
                record.num_candidate_configurations
            ),
            'original_hyperparameters': format_hyperparameters(
                record.original_hyperparameters
            ),
            'flipped_hyperparameters': format_hyperparameters(
                record.flipped_hyperparameters
            ),
            'original_mean_wape': format_float(record.original_mean_wape),
            'flipped_mean_wape': format_float(record.flipped_mean_wape),
            'wape_degradation': format_float(record.wape_degradation),
            'original_complexity_rank': str(record.original_complexity_rank),
            'flipped_complexity_rank': str(record.flipped_complexity_rank),
            'changed_by_flipped_ordering': str(
                record.changed_by_flipped_ordering
            ),
        }
        for record in records
    ]
    save_csv(dir_path / 'ordering_robustness.csv', rows)


def save_summary_table(
    records: list[ComplexityOrderingRecord],
    dir_path: Path,
) -> None:
    """Saves a summary table for the robustness experiment."""

    records_by_experiment: dict[str, list[ComplexityOrderingRecord]] = {}
    for record in records:
        records_by_experiment.setdefault(
            record.experiment_name,
            [],
        ).append(record)

    table = Table(
        headers=[
            'Experiment',
            'c_SE',
            'Selection Units',
            'Mean Candidate Configs',
            'Changed',
            'Changed Rate',
            'Mean Original WAPE',
            'Mean Flipped WAPE',
            'Mean WAPE Degradation',
            'Max WAPE Degradation',
        ]
    )
    for experiment_name in sorted(records_by_experiment):
        experiment_records = records_by_experiment[experiment_name]
        num_records = len(experiment_records)
        changed_count = sum(
            1
            for record in experiment_records
            if record.changed_by_flipped_ordering
        )
        coefficient = experiment_records[0].se_tolerance_coefficient
        mean_num_candidates = mean(
            record.num_candidate_configurations for record in experiment_records
        )
        mean_original_wape = mean(
            record.original_mean_wape for record in experiment_records
        )
        mean_flipped_wape = mean(
            record.flipped_mean_wape for record in experiment_records
        )
        mean_degradation = mean(
            record.wape_degradation for record in experiment_records
        )
        max_degradation = max(
            record.wape_degradation for record in experiment_records
        )
        table.append_row(
            experiment_name,
            format_float(coefficient),
            num_records,
            f'{mean_num_candidates:.2f}',
            changed_count,
            format_percentage(changed_count / num_records),
            format_percentage(mean_original_wape),
            format_percentage(mean_flipped_wape),
            format_percentage(mean_degradation),
            format_percentage(max_degradation),
        )

    table_path = dir_path / 'summary_table'
    save_content_to_file(table_path, table.__repr__())
    print(f'Saved summary table to "{table_path}".\n')
    print(table.__repr__() + '\n')


def save_figure(
    records: list[ComplexityOrderingRecord],
    dir_path: Path,
) -> None:
    """Saves the complexity-ordering robustness figure."""

    records_by_experiment: dict[str, list[ComplexityOrderingRecord]] = {}
    for record in records:
        records_by_experiment.setdefault(
            record.experiment_name,
            [],
        ).append(record)

    experiment_names = sorted(records_by_experiment)
    mean_degradations = [
        100.0
        * mean(
            record.wape_degradation
            for record in records_by_experiment[experiment_name]
        )
        for experiment_name in experiment_names
    ]
    changed_rates = [
        100.0
        * sum(
            1
            for record in records_by_experiment[experiment_name]
            if record.changed_by_flipped_ordering
        )
        / len(records_by_experiment[experiment_name])
        for experiment_name in experiment_names
    ]

    fig, ax = plt.subplots(figsize=(10.0, 5.5), dpi=200)
    x_positions = list(range(len(experiment_names)))
    bars = ax.bar(
        x_positions,
        mean_degradations,
        color=['#365f91', '#7f7f7f', '#9e4b4b'][: len(experiment_names)],
    )
    ax.axhline(0.0, color='#333333', linewidth=1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(experiment_names, rotation=20, ha='right')
    ax.set_ylabel('Mean WAPE Degradation from Flipped Ordering (pp)')
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.35)

    for bar, changed_rate in zip(bars, changed_rates):
        height = bar.get_height()
        y_offset = 0.02 if height >= 0 else -0.08
        va = 'bottom' if height >= 0 else 'top'
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_offset,
            f'{changed_rate:.0f}% changed',
            ha='center',
            va=va,
            fontsize=9,
        )

    fig.tight_layout()

    figure_path = dir_path / 'complexity_ordering_robustness.png'
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved complexity-ordering robustness figure to "{figure_path}".')
