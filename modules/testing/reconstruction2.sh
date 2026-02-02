#!/usr/bin/env zsh

set -euo pipefail

location="United States"
seed="140"

auda pipe run \
    "-p HT-SVR:metric=mape;use_time_series=1;num_k_folds=4" \
    "-p TF-Z-NORM:on=training_set -> MD-SVR:on=normalized_dataset" \
    "DS-YEAR-PW:location=$location;seed=$seed" \
    EV-RO:on=dataset\;ht_pipe=\@0\;training_pipe=\@1

auda pipe run \
    "DS-YEAR-PW:location=$location;seed=$seed" \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-PTS:no=sorted_dataset\;test_sample_indexes=1 \
    HT-PR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

auda pipe run \
    "DS-YEAR-PW:location=$location;seed=$seed" \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-PTS:no=sorted_dataset\;test_sample_indexes=1 \
    HT-SVR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

auda pipe run \
    "DS-YEAR-PW:location=$location;seed=$seed" \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-PTS:no=sorted_dataset\;test_sample_indexes=2 \
    HT-SVR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

auda pipe run \
    "DS-YEAR-PW:location=$location;seed=$seed" \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-PTS:no=sorted_dataset\;test_sample_indexes=3 \
    HT-SVR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

auda pipe run \
    "DS-YEAR-PW:location=$location;seed=$seed" \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-PTS:no=sorted_dataset\;test_sample_indexes=4 \
    HT-SVR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

auda pipe run \
    "DS-YEAR-PW:location=$location;seed=$seed" \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-PTS:no=sorted_dataset\;test_sample_indexes=5 \
    HT-SVR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

auda pipe run \
    "DS-YEAR-PW:location=$location;seed=$seed" \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-PTS:no=sorted_dataset\;test_sample_indexes=6 \
    HT-SVR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

auda pipe run \
    "DS-YEAR-PW:location=$location;seed=$seed" \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-PTS:no=sorted_dataset\;test_sample_indexes=7 \
    HT-SVR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4
