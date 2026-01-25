if __name__ == '__main__':
    import sys
    from pathlib import Path
    from typing import Dict, List

    import tabulate

    if len(sys.argv) != 3:
        print('Usage: python make_table.py <input_file> <output_file>')
        sys.exit(1)

    filepath = sys.argv[1]
    with open(filepath, 'r') as file:
        lines = file.readlines()

    keys: List[str] = []
    models: Dict[str, Dict[str, str]] = {}
    current_model_name = ''
    current_dict: Dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue

        if line.startswith('-'):
            if current_model_name:
                models[current_model_name] = current_dict

            current_model_name = line.strip().lstrip('-').strip()
            current_dict = {}
        else:
            key, value = line.split(' ', 1)
            key = key.strip()
            value = value.strip()
            current_dict[key] = value
            if key not in keys:
                keys.append(key)

    if current_model_name:
        models[current_model_name] = current_dict

    table_data: List[List[str]] = []
    for model_name, model_data in models.items():
        row = [model_name] + [model_data.get(key, '').strip() for key in keys]
        table_data.append(row)

    # Make the table using tabulate
    headers = ['Model'] + keys
    colalign = ['left'] + ['right'] * len(keys)
    content = table = tabulate.tabulate(
        table_data, headers=headers, tablefmt='github', colalign=colalign
    )

    # Output the table to a file
    output_filepath = sys.argv[2]
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, 'w') as file:
        file.write(content)

    print(content)
