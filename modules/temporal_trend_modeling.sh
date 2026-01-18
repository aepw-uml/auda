# ------------------------------------------------------------------------------
# MSRS Hyperparameter Optimization (Japan); metric = MAPE; seed = 140
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
# Find the best hyperparameters for best MAPE (seed = 140)
# ------------------------------------------------------------------------------

# Polynomial regression with MAPE optimization
# Results: degree = 3
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-PR:metric=mape\;seed=140\;use_anomaly_detection=0

# Ridge regression with MAPE optimization
# Results: alpha = 14.487
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-RR:metric=mape\;seed=140\;use_anomaly_detection=0

# GPR with MAPE optimization
# Results: length_scale = 192.848, noise_level = 8.101
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-GPR:metric=mape\;seed=140\;use_anomaly_detection=0

# SVR with MAPE optimization
# Results: C = 10.816, epsilon = 0.0958
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-SVR:metric=mape\;seed=140\;use_anomaly_detection=0

# ------------------------------------------------------------------------------
# Calculate the performance of the best hyperparameters found
# ------------------------------------------------------------------------------

# Polynomial Regression
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset;seed=140 -> 
    MD-PR:on=inlier_dataset;degree=3" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0\;seed=140

# Ridge Regression
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset;seed=140 -> 
    MD-RR:on=inlier_dataset;alpha=14.487" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0\;seed=140

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

# ------------------------------------------------------------------------------
# Baseline models
# ------------------------------------------------------------------------------

# Naive last value
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-NAIVE:on=inlier_dataset;seed=140" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0\;seed=140

# Drift model
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-DRIFT:on=inlier_dataset;seed=140" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0\;seed=140

# Exponential smoothing
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-ETS:on=inlier_dataset;seed=140" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0\;seed=140
