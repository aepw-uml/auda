# AUDA

**AUDA (AEPW x UML Data Analysis)** is a collaboration between **AEPW (Alliance to End Plastic Waste)** and **UML (University of Massachusetts Lowell)**. The project focuses on analyzing plastic waste data and exploring its environmental impact using upstream data from **PRISM (Plastics Recovery Insights Steering Model)**. PRISM is an AEPW tool that gathers plastic waste data from multiple sources and organizes it into a structured format. AUDA builds on that foundation to support analysis and research.

_This project is open source and welcomes contributions from developers, researchers, and anyone else interested in the work._

## Installation

> [!IMPORTANT]
>
> Installation requires access to the AEPW database and the analysis tables used by this project. If you are an AEPW developer, follow the steps below. If you are a researcher trying to reproduce the experiments, use the [Experiment Reproduction Guide](#experiment-reproduction-guide) instead.

AUDA currently supports Linux and macOS only.

Before you start, make sure the following tools are installed:

- `Python` 3.12.11 or higher.
- `Poetry` 2.1.3 or higher.
- `PostgreSQL` 18.3 or higher.

Install the Python environment and project dependencies with:

```bash
poetry install --no-root
```

This creates a virtual environment in `.venv` and installs the dependencies defined in `pyproject.toml`.

Next, create your local environment file:

```bash
cp .env.example .env
```

Update `DB_URL` in `.env` so it points to the correct PostgreSQL instance.

## Run modules

## Experiment Reproduction Guide
