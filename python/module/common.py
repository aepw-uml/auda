from pathlib import Path

from common.files import save_content_to_file
from experiment.experiment_group import ExperimentGroup
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
