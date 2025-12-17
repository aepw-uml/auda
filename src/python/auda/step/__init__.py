from typing import Dict, cast

from auda.core import auda
from auda.utils.pipeline import (
    add_kind,
    get_all_step_specs,
    scan_package,
    set_module_name_getter,
)

STEP_ID_MODULE_NAME_MAP_CACHE_KEY = 'step_id_module_name_cache'


def module_name_getter(step_id: str) -> str | None:
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
