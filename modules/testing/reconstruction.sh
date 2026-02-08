#!/usr/bin/env zsh

set -euo pipefail

location="United States"
seed="140"

auda pipe run \
    "-p HT-SVR:metric=mape;use_time_series=1;num_k_folds=4" \
    "-p TF-Z-NORM:on=training_set -> MD-SVR:on=normalized_dataset" \
    "DS-YEAR-PW:location=$location;seed=$seed" \
    EV-RO:on=dataset\;ht_pipe=\@0\;training_pipe=\@1
