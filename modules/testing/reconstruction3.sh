# ------------------------------------------------------------------------------
# Baselines; location = United\ States; seed = 140
# ------------------------------------------------------------------------------

# Naive persistence model
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-NAIVE:on=inlier_dataset" \
    DS-YEAR-PW:location=United\ States \
    PP-SPLIT:on=dataset\;seed=140 \
    EV-TSCV:on=training_set\;pipe=\@0

# Drift model
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-DRIFT:on=inlier_dataset;seed=140" \
    DS-YEAR-PW:location=United\ States \
    PP-SPLIT:on=dataset\;seed=140 \
    EV-TSCV:on=training_set\;pipe=\@0

# Exponential smoothing
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-ETS:on=inlier_dataset;seed=140" \
    DS-YEAR-PW:location=United\ States \
    PP-SPLIT:on=dataset\;seed=140 \
    EV-TSCV:on=training_set\;pipe=\@0

# ------------------------------------------------------------------------------
# MSRS Hyperparameter Optimization; location = "United States"; metric = MAPE;
# seed = 140; training_portion = 0.7; num_k_folds = 4 (because the dataset is
# too small)
# ------------------------------------------------------------------------------

# Polynomial regression
# [Results] degree = 2
auda pipe run \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    HT-PR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

# Polynomial Ridge regression
# [Results] degree = 3.0; alpha = 14.487
auda pipe run \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    HT-PRR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

# Gaussian Process Regression
# [Results] length_scale = 55.911, noise_level = 6.487
auda pipe run \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    HT-GPR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

# Support Vector Regression
# [Results] C = 72.281, epsilon = 0.006
auda pipe run \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    HT-SVR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

# ------------------------------------------------------------------------------
# Test with Best Hyperparameters; location = "United States"; metric = MAPE;
# seed = 140; training_portion = 0.7; should_shuffle = 0
# ------------------------------------------------------------------------------

# Polynomial regression
# [Hyperparameters] degree = 2
# [Results] MAPE = 2.780%
auda pipe run \
    "-p TF-Z-NORM:on=test_set -> AD-IF:on=normalized_dataset -> 
    MD-PR:on=inlier_dataset;degree=2" \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    EV-NORMAL:pipe=\@0

# Polynomial Ridge regression
# [Hyperparameters] degree = 3; alpha = 14.487
# [Results] MAPE = 10.505%
auda pipe run \
    "-p TF-Z-NORM:on=test_set -> AD-IF:on=normalized_dataset -> 
    MD-PRR:on=inlier_dataset;degree=3;alpha=14.487" \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    EV-NORMAL:pipe=\@0

# Gaussian Process Regression
# [Hyperparameters] length_scale = 55.911, noise_level = 6.487
# [Results] MAPE = 1.588%
auda pipe run \
    "-p TF-Z-NORM:on=test_set -> AD-IF:on=normalized_dataset -> 
    MD-GPR:on=inlier_dataset;length_scale=158.016;noise_level=6.093" \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    EV-NORMAL:pipe=\@0

# Support Vector Regression
# [Hyperparameters] C = 72.281, epsilon = 0.006
# [Results] MAPE = 1.177%
auda pipe run \
    "-p TF-Z-NORM:on=test_set -> AD-IF:on=normalized_dataset -> 
    MD-SVR:on=inlier_dataset;c=72.281;epsilon=0.006" \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    EV-NORMAL:pipe=\@0

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

# Polynomial regression
auda pipe run \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    TF-Z-NORM:on=training_set \
    MD-PR:on=normalized_dataset\;degree=2 \
    PL-PR:on=training_set \
    PL-SHOW

# Polynomial Ridge regression
auda pipe run \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    TF-Z-NORM:on=training_set \
    MD-PRR:on=normalized_dataset\;degree=3\;alpha=14.487 \
    PL-PRR:on=training_set \
    PL-SHOW

# Gaussian Process Regression
auda pipe run \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    TF-Z-NORM:on=training_set \
    MD-PR:on=normalized_dataset\;degree=2 \
    PL-GPR:on=training_set \
    PL-SHOW

# Support Vector Regression
auda pipe run \
    DS-YEAR-PW:location=United\ States \
    PP-SORT:on=dataset\;sort_by_feature_index=0 \
    PP-SPLIT:on=sorted_dataset\;training_portion=0.7\;seed=140 \
    TF-Z-NORM:on=training_set \
    MD-SVR:on=normalized_dataset\;c=72.281\;epsilon=0.002\;seed=140 \
    PL-SVR:on=training_set \
    PL-SHOW

# -----------------------------------------------------------------------------
# Outdated methods
# -----------------------------------------------------------------------------

# # C = 15.416, epsilon = 0.2019 (Best R2)
# auda pipe run \
#     DS-YEAR-PW:location=Japan \
#     HT-SVR:metric=r2\;seed=130\;num_iterations=50\;use_anomaly_detection=0
#
# # Plot SVR results for USA with best parameters
# auda pipe run \
#     DS-YEAR-PW:location=United\ States \
#     AD-IF:on=dataset \
#     TF-Z-NORM:on=inlier_dataset \
#     MD-SVR:on=normalized_dataset\;c=15.416\;epsilon=0.2019 \
#     PD-SVR:x_pred_values=2008,2013,2021 \
#     PL-SVR:on=inlier_dataset \
#     PL-SHOW
