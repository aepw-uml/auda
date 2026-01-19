# ------------------------------------------------------------------------------
# Baselines; location = Sweden; seed = 140
# ------------------------------------------------------------------------------

# Naive persistence model
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-NAIVE:on=inlier_dataset" \
    DS-YEAR-PW:location=Sweden \
    PP-SPLIT:on=dataset\;seed=140 \
    EV-TSCV:on=training_set\;pipe=\@0

# Drift model
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-DRIFT:on=inlier_dataset;seed=140" \
    DS-YEAR-PW:location=Sweden \
    PP-SPLIT:on=dataset\;seed=140 \
    EV-TSCV:on=training_set\;pipe=\@0

# Exponential smoothing
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-ETS:on=inlier_dataset;seed=140" \
    DS-YEAR-PW:location=Sweden \
    PP-SPLIT:on=dataset\;seed=140 \
    EV-TSCV:on=training_set\;pipe=\@0

# ------------------------------------------------------------------------------
# MSRS Hyperparameter Optimization; location = Sweden; metric = MAPE; seed = 140
# ------------------------------------------------------------------------------

# Polynomial regression
# [Results] degree = 9
auda pipe run \
    DS-YEAR-PW:location=Sweden \
    PP-SPLIT:on=dataset\;seed=140 \
    HT-PR:on=training_set\;metric=mape\;use_time_series=1

# Ridge regression
# [Results] alpha = 14.487
auda pipe run \
    DS-YEAR-PW:location=Sweden \
    PP-SPLIT:on=dataset\;seed=140 \
    HT-RR:on=training_set\;metric=mape\;use_time_series=1

# Gaussian Process Regression
# [Results] length_scale = 350.558, noise_level = 2.542
auda pipe run \
    DS-YEAR-PW:location=Sweden \
    PP-SPLIT:on=dataset\;seed=140 \
    HT-GPR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

# Support Vector Regression
# [Results] C = 1.756, epsilon = 0.153
auda pipe run \
    DS-YEAR-PW:location=Japan \
    PP-SPLIT:on=dataset\;seed=140 \
    HT-SVR:on=training_set\;metric=mape\;use_time_series=1\;num_k_folds=4

# ------------------------------------------------------------------------------
# Outdated methods
# ------------------------------------------------------------------------------

# C = 15.416, epsilon = 0.2019 (Best R2)
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-SVR:metric=r2\;seed=130\;num_iterations=50\;use_anomaly_detection=0

# Plot SVR results for USA with best parameters
auda pipe run \
    DS-YEAR-PW:location=United\ States \
    AD-IF:on=dataset \
    TF-Z-NORM:on=inlier_dataset \
    MD-SVR:on=normalized_dataset\;c=15.416\;epsilon=0.2019 \
    PD-SVR:x_pred_values=2008,2013,2021 \
    PL-SVR:on=inlier_dataset \
    PL-SHOW
