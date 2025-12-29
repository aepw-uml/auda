from typing import Dict, cast

from auda.core import auda
from auda.utils.pipeline import (
    Pipeline,
    add_kind,
    create_pipeline_from_str,
    get_all_step_specs,
    scan_package,
    set_module_name_getter,
)

STEP_PACKAGE_PATH = __path__
STEP_PACKAGE_NAME = __name__
STEP_ID_MODULE_NAME_MAP_CACHE_KEY = 'step_id_module_name_cache'


def module_name_getter(step_id: str) -> str | None:
    """Gets module name for a given step ID.

    Args:
        step_id: Step ID.
    """

    map: Dict[str, str] | None = cast(
        Dict[str, str] | None, auda.cache.get(STEP_ID_MODULE_NAME_MAP_CACHE_KEY)
    )

    if map is None or step_id not in map:
        scan_package(__path__, __name__)

        # Set cache again
        step_specs = get_all_step_specs()
        map = {}
        for spec in step_specs:
            map[spec.id] = spec.implementation.__module__

        auda.cache.set(STEP_ID_MODULE_NAME_MAP_CACHE_KEY, map)
    else:
        return map[step_id]


set_module_name_getter(module_name_getter)

# Setup kinds
add_kind('DS', 'dataset')
add_kind('MD', 'model')
add_kind('ST', 'statistics')
add_kind('TF', 'transformation')
add_kind('EV', 'evaluation')
add_kind('MD', 'model')
add_kind('PL', 'plotting')
add_kind('AD', 'anomaly_detection')
add_kind('HT', 'hyperparameter_tuning')
add_kind('PD', 'prediction')
add_kind('UT', 'utility')
add_kind('SY', 'synthetic_generation')


def create_pipeline_from_pipe(pipe: str | Pipeline) -> Pipeline:
    """Creates a Pipeline object from a given pipe input.

    Args:
        pipe: Pipe input, either a pipeline string or a Pipeline instance.

    Returns:
        A Pipeline instance.
    """

    if isinstance(pipe, Pipeline):
        return pipe
    else:
        return create_pipeline_from_str(pipe)
