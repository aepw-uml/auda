# C = 33.079, epsilon = 0.4670 (Best R2)
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-SVR:metric=r2\;seed=130\;use_anomaly_detection=0 \
    PL-MSRS-BSF \
    PL-SHOW

# C = 8.452, epsilon = 0.2773 (Best MAPE)
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-SVR:metric=mape\;seed=165 \
    PL-MSRS-BSF \
    PL-SHOW

# R2 = 0.7457, MAPE = 3.4480 (Best)
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-SVR:on=inlier_dataset;c=33.079;epsilon=0.4670;seed=130" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0

# R2 = -0.0454, MAPE = 3.5274
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> MD-SVR:on=inlier_dataset;
    c=8.452;epsilon=0.2773;seed=165" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0

# Plot with the best R2 model
auda pipe run \
    DS-YEAR-PW:location=Japan \
    TF-Z-NORM \
    MD-SVR:on=normalized_dataset\;c=33.079\;epsilon=0.4670\;seed=130 \
    PL-SVR:on=dataset \
    PL-SHOW

# Plot with the best MAPE model
auda pipe run \
    DS-YEAR-PW:location=Japan \
    TF-Z-NORM \
    MD-SVR:on=normalized_dataset\;c=8.452\;epsilon=0.2773\;seed=165 \
    PL-SVR:on=dataset \
    PL-SHOW

# ------------------------------------------------------------------------------

# SVR with R2 optimization
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-GPR:metric=mape\;seed=165\;num_iterations=50 \
    PL-MSRS-BSF \
    PL-SHOW

# ------------------------------------------------------------------------------

# Naive last value
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-NAIVE:on=inlier_dataset" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0

# Drift baseline
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-DRIFT:on=inlier_dataset" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0

# Exponential smoothing baseline
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-ETS:on=inlier_dataset" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0

# Polynomial regression
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-PR:on=inlier_dataset;degree=4" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0

# Gaussian process regression
auda pipe run \
    "-p TF-Z-NORM -> AD-IF:on=normalized_dataset -> 
    MD-GPR:on=inlier_dataset;degree=4" \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe=\@0

# ------------------------------------------------------------------------------
# Lab (Find best seeds to showcase)

# SVR - R2
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-SVR:metric=r2\;use_anomaly_detection=0\
    PL-MSRS-BSF \
    PL-SHOW

# SVR - MAPE
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-SVR:metric=mape\;use_anomaly_detection=0\
    PL-MSRS-BSF \
    PL-SHOW

# GPR - MAPE
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-GPR:metric=mape\;use_anomaly_detection=0\
    PL-MSRS-BSF \
    PL-SHOW

# Ridge Regression - MAPE
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-RR:metric=mape\;use_anomaly_detection=0\
    PL-MSRS-BSF \
    PL-SHOW

# ----------------------------------------------------------------------------- 
# Tuning time
# ----------------------------------------------------------------------------- 

auda pipe run \
    DS-YEAR-PW:location=Japan \
    UT-TIMER-START \
    HT-SVR:metric=mape\;seed=130 \
    UT-TIMER-STOP \

auda pipe run \
    DS-YEAR-PW:location=Japan \
    UT-TIMER-START \
    HT-PR:metric=mape\;seed=130 \
    UT-TIMER-STOP \

