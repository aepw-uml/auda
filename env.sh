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
    # Experiment 1 - Correlation analysis
    auda workflow run PWDriverFeatureSet Correlation

    # Experiment 2 - Importance analysis
    auda workflow run PWGPredictors FeatureImportances --contamination=0.2 \
        --seed=471

    # Experiment 3 - Reconstruction (TRC for the United States & TRC for Japan)
    auda workflow run YearTRC MultipleReconstruction \
        --location=United\ States --seed=471
    auda workflow run YearTRC MultipleReconstruction --location=Japan \
        --seed=471

    auda workflow run YearTRC Reconstruction --location=United\ States \
        --seed=475
    auda workflow run YearTRC Reconstruction --location=Japan --seed=472

    # Experiment 4 - SE tolerance coefficient sweep and complexity ordering
    # robustness (TRC reconstruction for the United States)
    auda workflow run YearTRC SEToleranceCoefficientSweep \
        --location=United\ States --seed=471
    auda workflow run YearTRC ComplexityOrderingRobustness \
        --location=United\ States --seed=471 --se_tolerance_coefficient=0.1

    # Experiment 5 - Global plastic production forecasting
    auda workflow run GlobalPlasticsProduction MultipleForecasting --seed=471 \
        --workflow_name=multiple_global_forecasting_grid_search \
        --tune_search_type=grid
    auda workflow run GlobalPlasticsProduction MultipleForecasting --seed=471 \
        --workflow_name=multiple_global_forecasting_random_search \
        --tune_search_type=random

    auda workflow run GlobalPlasticsProduction Forecasting --seed=471 \
        --workflow_name=global_forecasting --tune_search_type=grid
    auda workflow run GlobalPlasticsProduction Forecasting --seed=471 \
        --workflow_name=global_forecasting --tune_search_type=random

    # Experiment 6 - PWG forecasting (Japan)
    auda workflow run YearPWG MultipleForecasting --location=Japan --seed=471
    auda workflow run YearPWG MultipleForecasting --location=United\ States \
        --seed=471

    auda workflow run YearPWG Forecasting --location=Japan --seed=471
    auda workflow run YearPWG Forecasting --location=United\ States --seed=471

    # Experiment 7 - NN PWG forecasting
    auda workflow run PWDrivers NNForecasting --seed=471

    # Experiment 8 - PWG multivariate forecasting (Japan & Slovenia)
    auda workflow run PWDrivers MultipleMultivariateForecasting --seed=471 \
        --location=Japan --workflow_name=multiple_multivariate_forecasting_japan
    auda workflow run PWDrivers MultipleMultivariateForecasting --seed=471 \
        --location=Slovenia --enable_tuning=0 \
        --workflow_name=multiple_multivariate_forecasting_slovenia
}

function move-figures() {
    DEST="paper/src/figures"
    mkdir -p "$DEST"

    # Prism data extraction
    cp images/prism_data_extraction.png \
        "$DEST/prism_data_extraction.png"

    # Experiment 1 - Correlation analysis
    cp "results/correlation/correlation_matrix.png" \
        "$DEST/correlation_matrix.png"

    # Experiment 2 - Importance analysis
    cp "results/feature_importances/feature_importances.png" \
        "$DEST/feature_importances.png"

    # Experiment 3 - Reconstruction (TRC for the United States & TRC for Japan)
    DIR="results/reconstruction_japan/plots"
    cp "$DIR/ridge_regression.png" \
        "$DEST/japan_trc_reconstruction_ridge_regression.png"
    cp "$DIR/support_vector_regression.png" \
        "$DEST/japan_trc_reconstruction_svr.png"
    DIR="results/reconstruction_united_states/plots"
    cp "$DIR/gaussian_process_regression.png" \
        "$DEST/united_states_trc_reconstruction_gpr.png"
    cp "$DIR/support_vector_regression.png" \
        "$DEST/united_states_trc_reconstruction_svr.png"

    # Experiment 4 - SE tolerance coefficient sweep and complexity ordering
    # robustness (TRC reconstruction for the United States)
    DIR="results/se_tolerance_coefficient_sweep_united_states"
    cp "$DIR/selection_sensitivity_to_se_tolerance.png" \
        "$DEST/selection_sensitivity_to_se_tolerance.png"
    DIR="results/complexity_ordering_robustness_united_states"
    cp "$DIR/complexity_ordering_robustness.png" \
        "$DEST/complexity_ordering_robustness.png"

    # Experiment 5 - Global plastic production forecasting (random search)
    DIR="results/global_forecasting_random_search/plots"
    cp "$DIR/theil_sen_regression.png" \
        "$DEST/global_forecasting_theil_sen_regression.png"
    cp "$DIR/ridge_regression.png" \
        "$DEST/global_forecasting_ridge_regression.png"
    cp "$DIR/gaussian_process_regression.png" \
        "$DEST/global_forecasting_gpr.png"
    cp "$DIR/support_vector_regression.png" \
        "$DEST/global_forecasting_svr.png"

    # Experiment 6 - PWG forecasting
    DIR="results/forecasting_random_search/plots"
    cp "$DIR/arima_regression.png" \
        "$DEST/japan_pwg_forecasting_arima_regression.png"
    cp "$DIR/ridge_regression.png" \
        "$DEST/japan_pwg_forecasting_ridge_regression.png"
    cp "$DIR/gaussian_process_regression.png" \
        "$DEST/japan_pwg_forecasting_gpr.png"
    cp "$DIR/support_vector_regression.png" \
        "$DEST/japan_pwg_forecasting_svr.png"
}

function datasets() {
    # Experiment 1 - Correlation analysis
    auda dataset show PWDriverFeatureSet --no-samples
    printf '%*s\n' 80 '' | tr ' ' '-'

    # Experiment 2 - Importance analysis
    auda dataset show PWGPredictors --no-samples
    printf '%*s\n' 80 '' | tr ' ' '-'

    # Experiment 3 - Reconstruction (TRC for the United States & TRC for Japan)
    auda dataset show YearTRC --location=Japan --no-samples
    auda dataset show YearTRC --location=United\ States --no-samples
    printf '%*s\n' 80 '' | tr ' ' '-'

    # Experiment 5 - Global plastic production forecasting
    auda dataset show GlobalPlasticsProduction --no-samples
    printf '%*s\n' 80 '' | tr ' ' '-'

    # Experiment 6 - PWG forecasting (Japan & United States)
    auda dataset show YearPWG --location=Japan --no-samples
    auda dataset show YearPWG --location=United\ States --no-samples
    printf '%*s\n' 80 '' | tr ' ' '-'

    # Experiment 7 - NN PWG forecasting
    auda dataset show PWDrivers --no-samples
    printf '%*s\n' 80 '' | tr ' ' '-'

    # Experiment 8 - PWG multivariate forecasting (Japan & Slovenia)
    auda dataset show PWDrivers --location=Japan --no-samples
    auda dataset show PWDrivers --location=Slovenia --no-samples
}
