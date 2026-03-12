from typing import Any


def get_hyperparameters_str(hyperparameters: dict[str, Any]) -> str:
    items: list[str] = []
    for name, value in hyperparameters.items():
        if isinstance(value, float):
            items.append(f'{name}={value:.3e}')
        elif isinstance(value, list):
            items.append(f'{name}=[{", ".join(str(v) for v in value)}]')
        else:
            items.append(f'{name}={value}')

    return ', '.join(items)
