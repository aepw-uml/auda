#!/usr/bin/env bash

# This script runs a temporal trend modeling pipeline for a specified location.

# [1] DS-YEAR-PW (dataset)
#     Retrieves yearly plastic waste generation data for a specific location.
# [2] ST-BASIC (stat)
#     Computes basic descriptive statistics of the input samples.
# [3] MD-IF (model)
#     Trains an Isolation Forest model for anomaly detection.
# [4] MD-SVR (model)
#     Trains a Support Vector Regression (SVR) model for predictive analysis.
# [5] PL-SVR (plotter)
#     Generates predictions using a trained Support Vector Regression (SVR)
#     model.
# [6] SHOW (plotter)
#     Displays the generated figure interactively.
auda pipe run DS-YEAR-PW ST-BASIC MD-IF MD-SVR PL-SVR SHOW \
    '--inputs=location=Japan'

# Minimal MAPE
# auda pipe run DS-YEAR-PW SVR-TUNER "--inputs=location=Japan;indicator=mape"
auda pipe run DS-YEAR-PW ST-BASIC MD-IF MD-SVR PL-SVR SHOW \
    '--inputs=location=Japan;regularization_parameter=46.272;epsilon=0.22'

# Maximal 
# auda pipe run DS-YEAR-PW SVR-TUNER "--inputs=location=Japan;indicator=r2"
auda pipe run DS-YEAR-PW ST-BASIC MD-IF MD-SVR PL-SVR SHOW \
    '--inputs=location=Japan;regularization_parameter=56.848;epsilon=0.535'

# United States
auda pipe run DS-YEAR-PW ST-BASIC MD-SVR PL-SVR SHOW \
    '--inputs=location=United States;regularization_parameter=79.353;epsilon=0.044'
