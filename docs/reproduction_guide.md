# Reproduction Guide

To reproduce the results of this project, first install the following prerequisites in your environment:

- `Python` 3.12.11 or higher.
- `Poetry` 2.1.3 or higher.

Use the following commands to check if you have the required versions of Python and Poetry installed:

```bash
python --version
poetry --version
```

Then, create a new virtual environment (`.venv`) and install the project dependencies using Poetry:

```bash
poetry install --no-root
```

Copy the `.env.example` file to `.env`. AUDA detects the presence of the `.env` file on startup, but no environment variables are required to set up to run the reproduction.

```bash
cp .env.example .env
```

Copy the `.cache-reproduction` directory to `cache`. Because the original source records from the PRISM production database are not publicly available, we have included a cache of the source records in the `.cache-reproduction` directory. AUDA always retrieves data from the `cache` directory if it exists, so copying the `.cache-reproduction` directory to `cache` allows you to reproduce the results without needing access to the original source records.

```bash
cp -r .cache-reproduction cache
```

Next, source the `env.sh` file to set up the project-wise environment variables and functions:

```bash
source env.sh
```

Finally, run the `reproduce` command to execute the reproduction:

```bash
reproduce
```

All the results of the reproduction will present in the `results` directory.

## All-in-One Command

The following command combines all the above steps into a single command for convenience:

```bash
python --version && \
poetry --version && \
poetry install --no-root && \
cp .env.example .env && \
cp -r .cache-reproduction cache && \
source env.sh && \
reproduce
```
