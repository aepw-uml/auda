from pathlib import Path
from string import Template


class TemplateManager:
    """Manages template files stored in a specified directory."""

    def __init__(self, templates_dir: Path):
        """Initializes the TemplateManager.

        Args:
            templates_dir: Path object representing the templates directory.

        Attributes:
            templates_dir: Path object representing the templates directory.
        """

        self.templates_dir = templates_dir

    def get(self, template_name: str) -> Template:
        """Reads and returns the content of a template file.

        Args:
            template_name: The name of the template file.

        Returns:
            The content of the template file as a string.
        """

        template_path = self.templates_dir / template_name

        with open(template_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Remove all the lines that start with '#'
            content = '\n'.join(
                line
                for line in content.splitlines(keepends=False)
                if not line.lstrip().startswith('#')
            )

            return Template(content)
