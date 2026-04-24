# Activate Python virtual environment.
source .venv/bin/activate

# Set the PYTHONPATH to include all the modules under `python`.
_path="$PWD/python"
if [[ ":$PYTHONPATH:" != *":$_path:"* ]]; then
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$_path"
fi

# Add the `bin` directory to the PATH.
_path="$PWD/bin"
if [[ ":$PATH:" != *":$_path:"* ]]; then
    export PATH="${PATH:+$PATH:}$_path"
fi

function reproduce() {
    auda workflow run YearPWG Forecasting --location=Japan --seed=471 \
        --tune_search_type=grid
    auda workflow run YearPWG Forecasting --location=Japan --seed=471 \
        --tune_search_type=random
    auda workflow run GlobalPlasticsProduction Forecasting \
        --seed=471 --workflow_name=global_forecasting
    auda workflow run YearPWG Reconstruction '--location=United States' \
        --seed=471
    auda workflow run YearPPC MultipleReconstruction --location=Japan \
        --seed=471
    auda workflow run PWDriverFeatureSet Correlation
    auda workflow run PWGPredictors FeatureImportances \
        --contamination=0.2 --seed=471
    auda workflow run PWDrivers NNForecasting --seed=471
    auda workflow run PWDrivers MultivariateForecasting --seed=471 \
        --location=Japan
    auda workflow run PWDrivers MultivariateForecasting --seed=471 \
        --location=Slovenia --enable_tuning=0
}
