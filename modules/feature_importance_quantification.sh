#! /usr/bin/env bash

# This script runs two analysis pipelines related to plastic waste generation
# and its correlation with demographic and waste management indicators.

# [1] DS-SIGNIFICANT-FEATURES (dataset)
#     Collects samples containing key demographic and waste management
#     indicators.
# [2] MD-CM (model)
#     Computes the correlation matrix among input features.
# [3] PL-CM (plotter)
#     Visualizes a correlation matrix as a heatmap.
# [4] SHOW (plotter)
#     Displays the generated figure interactively.
auda pipe run DS-SIGNIFICANT-FEATURES MD-CM PL-CM SHOW

# [1] DS-PW-RELATED (dataset)
#     Retrieves plastic waste generation data along with relevant demographic
#     indicators.
# [2] MD-IF (model)
#     Trains an Isolation Forest model for anomaly detection.
# [3] MD-RF (model)
#     Trains a Random Forest regressor for feature importance and prediction.
# [4] PL-FI (plotter)
#     Visualizes feature importances from a trained model.
# [5] SHOW (plotter)
#     Displays the generated figure interactively.
auda pipe run DS-PW-RELATED MD-IF MD-RF PL-FI SHOW
