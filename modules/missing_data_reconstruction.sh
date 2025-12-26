
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
    PL-SHOW \
