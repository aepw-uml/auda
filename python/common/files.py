from pathlib import Path


def save_content_to_file(file_path: Path, content: str) -> None:
    """
    Save the given content to a file at the specified path.

    Args:
        file_path: The path to the file where the content will be saved.
        content: The content to be saved.
    """

    # Ensure the parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the content to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
