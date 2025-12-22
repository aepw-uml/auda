# C = 33.079, epsilon = 0.4670 (Best R2)
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-SVR:metric=r2\;seed=130\;num_iterations=50 \
    PL-MSRS-BSF \
    PL-SHOW

# C = 8.452, epsilon = 0.2773 (Best MAPE)
auda pipe run \
    DS-YEAR-PW:location=Japan \
    HT-SVR:metric=mape\;seed=165\;num_iterations=50 \
    PL-MSRS-BSF \
    PL-SHOW

# R2 = 0.7457, MAPE = 3.4480
auda pipe run \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe='TF-Z-NORM AD-IF:on=normalized_dataset RG-SVR:on=inlier_dataset;c=33.079;epsilon=0.4670;seed=130'

# R2 = -0.0454, MAPE = 3.5274
auda pipe run \
    DS-YEAR-PW:location=Japan \
    EV-CV:pipe='TF-Z-NORM AD-IF:on=normalized_dataset RG-SVR:on=inlier_dataset;c=8.452;epsilon=0.2773;seed=165'

# Show best R2 model
auda pipe run \
    DS-YEAR-PW:location=Japan \
    TF-Z-NORM \
    RG-SVR:on=normalized_dataset\;c=33.079\;epsilon=0.4670\;seed=130 \
    PL-SVR:on=dataset \
    PL-SHOW

# Show best MAPE model
auda pipe run \
    DS-YEAR-PW:location=Japan \
    TF-Z-NORM \
    RG-SVR:on=normalized_dataset\;c=8.452\;epsilon=0.2773\;seed=165 \
    PL-SVR:on=dataset \
    PL-SHOW
