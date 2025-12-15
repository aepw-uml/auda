# This script sets up the environment for the AUDA project.

# Activate Python virtual environment
source .venv/bin/activate

# Set the PYTHONPATH to include the all the modules under `src/python`
PYTHONPATH="$PYTHONPATH:$PWD/src/python"
export PYTHONPATH

# Add the `src/bin` directory to the PATH
PATH="$PWD/src/bin:$PATH"
export PATH
