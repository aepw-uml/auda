### Reproduction Commands

## Correlation Analysis of PWDriverFeatureSet

```bash
auda workflow run PWDriverFeatureSet Correlation
```

## Importance Analysis of PWDriverFeatureSet

```bash
auda workflow run PWGPredictors FeatureImportances --contamination=0.2 --seed=471
```

## Multiple Reconstruction of the United States total resin consumption

```bash
# Multiple reconstruction does not produce any images
auda workflow run YearTRC MultipleReconstruction --location=United\ States --seed=471
auda workflow run YearPPC MultipleReconstruction --location=Japan --seed=471

# Generate images for the multiple reconstruction results
auda workflow run YearTRC Reconstruction --location=United\ States --seed=471
auda workflow run YearPPC Reconstruction --location=Japan --seed=471
```

## Forecasting of global plastics production

```bash
auda workflow run GlobalPlasticsProduction Forecasting --seed=471 --workflow_name=global_forecasting --tune_search_type=grid
auda workflow run GlobalPlasticsProduction Forecasting --seed=471 --workflow_name=global_forecasting --tune_search_type=random
```

## Forecasting of Japan plastic waste generation

```bash
auda workflow run YearPWG Forecasting --location=Japan --seed=471
```

## Neural Network Forecasting

```bash
auda workflow run PWDrivers NNForecasting --seed=471
```

## Multivariate Forecasting

```bash
auda workflow run PWDrivers MultivariateForecasting --seed=471 --location=Japan --workflow_name=multivariate_forecasting_japan
auda workflow run PWDrivers MultivariateForecasting --seed=471 --location=Slovenia --enable_tuning=0 --workflow_name=multivariate_forecasting_slovenia
```
