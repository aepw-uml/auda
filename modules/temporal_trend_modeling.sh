# ------------------------------------------------------------------------------
# Baseline Models; seed = 140
# ------------------------------------------------------------------------------

# Naive persistence model
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-NAIVE:on=inlier_dataset" \
    DS-YEAR-PW:location=Japan \
    PP-SPLIT:on=dataset\;seed=140 \
    EV-TSCV:on=training_set\;pipe=\@0

# Drift model
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-DRIFT:on=inlier_dataset;seed=140" \
    DS-YEAR-PW:location=Japan \
    PP-SPLIT:on=dataset\;seed=140 \
    EV-TSCV:on=training_set\;pipe=\@0

# Exponential smoothing
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-ETS:on=inlier_dataset;seed=140" \
    DS-YEAR-PW:location=Japan \
    PP-SPLIT:on=dataset\;seed=140 \
    EV-TSCV:on=training_set\;pipe=\@0

# ------------------------------------------------------------------------------
# MSRS Hyperparameter Optimization; location = Japan; metric = MAPE; seed = 140
# ------------------------------------------------------------------------------

# Polynomial regression
# [Results] degree = 3
auda pipe run \
    DS-YEAR-PW:location=Japan \
    PP-SPLIT:on=dataset\;seed=140 \
    HT-PR:on=training_set\;metric=mape\;use_anomaly_detection=0

# Ridge regression
# [Results] alpha = 14.487
auda pipe run \
    DS-YEAR-PW:location=Japan \
    PP-SPLIT:on=dataset\;seed=140 \
    HT-RR:on=training_set\;metric=mape\;use_anomaly_detection=0

# Gaussian Process Regression
# [Results] length_scale = 350.558, noise_level = 2.542
auda pipe run \
    DS-YEAR-PW:location=Japan \
    PP-SPLIT:on=dataset\;seed=140 \
    HT-GPR:on=training_set\;metric=mape\;use_anomaly_detection=0

# Support Vector Regression
# [Results] C = 1.756, epsilon = 0.153
auda pipe run \
    DS-YEAR-PW:location=Japan \
    PP-SPLIT:on=dataset\;seed=140 \
    HT-SVR:on=training_set\;metric=mape\;use_anomaly_detection=0

# ------------------------------------------------------------------------------
# Test with Best Hyperparameters; location = Japan; seed = 140
# ------------------------------------------------------------------------------

# Polynomial Regression
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-PR:on=inlier_dataset;degree=3" \
    DS-YEAR-PW:location=Japan \
    PP-SPLIT:on=dataset\;seed=140 \
    EV-CV:pipe=\@0

# Ridge Regression
auda pipe run \
    "-p TF-Z-NORM:on=test_set -> AD-IF:on=normalized_dataset -> 
    MD-RR:on=inlier_dataset;alpha=14.487" \
    DS-YEAR-PW:location=Sweden \
    PP-SPLIT:on=dataset\;seed=140 \
    EV-CV:pipe=\@0

# ------------------------------------------------------------------------------
# Calculate the performance of the best hyperparameters found
# ------------------------------------------------------------------------------

# GPR
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset;seed=140 -> 
    MD-GPR:on=inlier_dataset;length_scale=192.848;noise_level=8.101" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0\;seed=140

# SVR
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset;seed=140 -> 
    MD-SVR:on=inlier_dataset;c=10.816;epsilon=0.0958" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0\;seed=140
