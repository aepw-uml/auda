from importlib import import_module
from pkgutil import iter_modules

from .__common import get_task_spec, run_pipeline

# Dynamically import all submodules and subpackages.
#
# This for loop will import all modules and subpackages in the current package,
# except those whose names start with an underscore (_).
for _, name, is_package in iter_modules(__path__, f'{__name__}.'):
    if is_package and not name.startswith('_'):
        package = import_module(name)

        basename = name.rsplit('.', 1)[-1]
        for _, module_name, is_package in iter_modules(package.__path__, name + '.'):
            if module_name.rsplit('.', 1)[-1].startswith('_'):
                continue

            import_module(module_name)

__all__ = [
    'get_task_spec',
    'run_pipeline',
]
