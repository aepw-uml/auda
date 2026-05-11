# Reproduction Guide

> [!NOTE]
>
> AUDA currently supports **Linux** (including WSL) and **macOS** only.

To reproduce the results in this project, make sure your environment has the following prerequisites:

- `Python` 3.12.11 or later.
- `Poetry` 2.1.3 or later.

Check the installed versions with:

```bash
python --version
poetry --version
```

Then install the project dependencies with Poetry. This creates a `.venv` virtual environment if one does not already exist.

```bash
poetry install --no-root
```

Copy `.env.example` to `.env`. AUDA checks for this file at startup, but the reproduction does not require you to configure any environment variables.

```bash
cp .env.example .env
```

Copy `.cache-reproduction` to `cache`. The original source records from the PRISM production database are not publicly available, so this project includes a cached copy in `.cache-reproduction`. When the `cache` directory exists, AUDA reads source records from it, which lets you reproduce the results without access to the production database.

```bash
cp -r .cache-reproduction cache
```

Next, source `env.sh` to set up the project environment variables and shell functions:

```bash
source env.sh
```

Finally, run the reproduction:

```bash
reproduce
```

The reproduction outputs are written to the `results` directory.

## All-in-One Command

For convenience, the full workflow can be run as one command:

```bash
python --version && \
poetry --version && \
poetry install --no-root && \
cp .env.example .env && \
cp -r .cache-reproduction cache && \
source env.sh && \
reproduce
```
