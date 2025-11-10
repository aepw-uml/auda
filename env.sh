# This script sets up the virtual Python environment for this project. The
# virtual environment is created using Poetry, and it includes all necessary
# dependencies for the AEPW Data Analysis project.
source .venv/bin/activate

# Set the PYTHONPATH to include the `ada` module
PYTHONPATH="$PYTHONPATH:$(pwd)/src"
export PYTHONPATH

# Add the `src/bin` directory to the PATH
PATH="$(pwd)/src/bin:$PATH"
export PATH

echo "The AUDA (UML x AEPW Data Analysis) environment setup has been completed."
