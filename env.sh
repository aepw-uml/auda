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

# Deprecated: use the `auda` command instead.
function run_module() {
    python "python/module/$1.py" "${@:2}"
}

# Reproduce the results in the paper.
function reproduce() {
    rm -rf results/figure
    rm -rf results/metric_table
    rm -rf results/hyperparameter_table
    rm -rf results/time

    mkdir -p results/figure
    mkdir -p results/metric_table
    mkdir -p results/hyperparameter_table
    mkdir -p results/time

    echo "Module: projection (year_pwg)"
    auda module run projection year_pwg --location=Japan --seed=160
    for f in results/module/projection/plots/*.png; do
        cp "$f" "results/figure/projection_pwg_$(basename "$f")"
    done
    cp results/module/projection/metric_table \
        results/metric_table/projection_pwg
    cp results/module/projection/hyperparameter_table \
        results/hyperparameter_table/projection_pwg
    cp results/module/projection/time_table \
        results/time_table/projection_pwg

    echo "Module: projection (global-plastics-production = gpp)"
    auda module run projection global_year_plastics_production --seed=150
    for f in results/module/projection/plots/*.png; do
        cp "$f" "results/figure/projection_gpp_$(basename "$f")"
    done
    cp results/module/projection/metric_table \
        results/metric_table/projection_gpp
    cp results/module/projection/hyperparameter_table \
        results/hyperparameter_table/projection_gpp
    cp results/module/projection/time_table \
        results/time_table/projection_gpp

    echo "Module: reconstruction (year_ppc)"
    auda module run reconstruction year_ppc --location=Japan --seed=149
    for f in results/module/reconstruction/plots/*.png; do
        cp "$f" "results/figure/reconstruction_ppc_$(basename "$f")"
    done
    cp results/module/reconstruction/metric_table \
        results/metric_table/reconstruction_ppc
    cp results/module/reconstruction/hyperparameter_table \
        results/hyperparameter_table/reconstruction_ppc
    cp results/module/projection/time_table \
        results/time_table/projection_ppc

    echo "Module: multiple reconstruction (year_ppc)"
    auda module run multiple_reconstruction year_ppc --location=Japan \
        --seed=120
    cp results/module/reconstruction/multiple_reconstruction/metric_table \
        results/metric_table/multiple_reconstruction_ppc
}
