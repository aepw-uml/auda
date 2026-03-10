# Activate Python virtual environment.
source .venv/bin/activate

# Set the PYTHONPATH to include all the modules under `python`.
PYTHONPATH="$PYTHONPATH:$PWD/python" && export PYTHONPATH

# Add the `bin` directory to the PATH
PATH="$PWD/bin:$PATH" && export PATH
