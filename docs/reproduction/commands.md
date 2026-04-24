### Reproduction Commands

## Forecasting of Japan plastic waste generation

```bash
auda workflow run YearPWG Forecasting --location=Japan --seed=471 --tune_search_type=grid
auda workflow run YearPWG Forecasting --location=Japan --seed=471 --tune_search_type=random
```

## Forecasting of global plastics production

```bash
auda workflow run GlobalPlasticsProduction Forecasting --seed=471 --workflow_name=global_forecasting
```

## Reconstruction of United States plastic waste generation

```bash
auda workflow run YearPWG Reconstruction '--location=United States' --seed=471
```

## Multiple Reconstruction of the United States total resin consumption

```bash
auda workflow run YearTRC MultipleReconstruction --location=United\ States --seed=471;
```

## Correlation Analysis of PWDriverFeatureSet

```bash
auda workflow run PWDriverFeatureSet Correlation
```

## Importance Analysis of PWDriverFeatureSet

```bash
auda workflow run PWGPredictors FeatureImportances --contamination=0.2 --seed=471
```

## Neural Network Forecasting

```bash
auda workflow run PWDrivers NNForecasting --seed=471
```

## Multivariate Forecasting

```bash
auda workflow run PWDrivers MultivariateForecasting --seed=471 --location=Japan
auda workflow run PWDrivers MultivariateForecasting --seed=471 --location=Slovenia --enable_tuning=0
```
