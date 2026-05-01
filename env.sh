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
    auda workflow run PWDriverFeatureSet Correlation
    auda workflow run PWGPredictors FeatureImportances --contamination=0.2 \
        --seed=471
    auda workflow run YearTRC MultipleReconstruction \
        --location=United\ States --seed=471
    auda workflow run YearPPC MultipleReconstruction --location=Japan \
        --seed=471
    auda workflow run YearTRC Reconstruction --location=United\ States \
        --seed=471
    auda workflow run YearPPC Reconstruction --location=Japan --seed=471
    auda workflow run GlobalPlasticsProduction Forecasting --seed=471 \
        --workflow_name=global_forecasting --tune_search_type=grid
    auda workflow run GlobalPlasticsProduction Forecasting --seed=471 \
        --workflow_name=global_forecasting --tune_search_type=random
    auda workflow run YearPWG Forecasting --location=Japan --seed=471
    auda workflow run PWDrivers NNForecasting --seed=471
    auda workflow run PWDrivers MultivariateForecasting --seed=471 \
        --location=Japan
    auda workflow run PWDrivers MultivariateForecasting --seed=471 \
        --location=Slovenia --enable_tuning=0
}

function move-figures() {
    DEST="paper/src/figures"
    mkdir -p "$DEST"

    # Prism data extraction
    cp images/prism_data_extraction.png \
        "$DEST/prism_data_extraction.png"

    # Correlation matrix
    cp "results/correlation/correlation_matrix.png" \
        "$DEST/correlation_matrix.png"

    # Importance analysis histogram
    cp "results/feature_importances/feature_importances.png" \
        "$DEST/feature_importances.png"

    # Reconstruction (Japan)
    DIR="results/multiple_reconstruction_japan/plots"
    cp "$DIR/theil-sen-regression.png" \
        "$DEST/japan_trc_reconstruction_theil_sen_regression.png"
    cp "$DIR/ridge-regression.png" \
        "$DEST/japan_trc_reconstruction_ridge_regression.png"
    cp "$DIR/gaussian-process-regression.png" \
        "$DEST/japan_trc_reconstruction_gpr.png"
    cp "$DIR/support-vector-regression.png" \
        "$DEST/japan_trc_reconstruction_svr.png"

    # Reconstruction (United States)
    DIR="results/multiple_reconstruction_united-states/plots"
    cp "$DIR/theil-sen-regression.png" \
        "$DEST/united_states_ppc_reconstruction_theil_sen_regression.png"
    cp "$DIR/ridge-regression.png" \
        "$DEST/united_states_ppc_reconstruction_ridge_regression.png"
    cp "$DIR/gaussian-process-regression.png" \
        "$DEST/united_states_ppc_reconstruction_gpr.png"
    cp "$DIR/support-vector-regression.png" \
        "$DEST/united_states_ppc_reconstruction_svr.png"

    # Global forecasting random search
    DIR="results/global_forecasting_random_search/plots"
    cp "$DIR/theil-sen-regression.png" \
        "$DEST/global_forecasting_theil_sen_regression.png"
    cp "$DIR/ridge-regression.png" \
        "$DEST/global_forecasting_ridge_regression.png"
    cp "$DIR/gaussian-process-regression.png" \
        "$DEST/global_forecasting_gpr.png"
    cp "$DIR/support-vector-regression.png" \
        "$DEST/global_forecasting_svr.png"

    # Japan PWG forecasting
    DIR="results/forecasting_random_search/plots"
    cp "$DIR/arima-regression.png" \
        "$DEST/japan_pwg_forecasting_arima_regression.png"
    cp "$DIR/ridge-regression.png" \
        "$DEST/japan_pwg_forecasting_ridge_regression.png"
    cp "$DIR/gaussian-process-regression.png" \
        "$DEST/japan_pwg_forecasting_gpr.png"
    cp "$DIR/support-vector-regression.png" \
        "$DEST/japan_pwg_forecasting_svr.png"
}
