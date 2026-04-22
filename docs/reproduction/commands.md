### Reproduction Commands

## Forecasting of Japan plastic waste generation

```bash
auda workflow run YearPWG Forecasting --location=Japan --seed=471 --tune_search_type=grid
```

## Forecasting of global plastics production

```bash
auda workflow run GlobalYearPlasticsProduction Forecasting --seed=471 --workflow_name=global_forecasting
```

## Reconstruction of United States plastic waste generation

```bash
auda workflow run YearPWG Reconstruction '--location=United States' --seed=471
```

## Multiple Reconstruction of Japan polymer PET consumption

```bash
auda workflow run YearPPC MultipleReconstruction --location=Japan --seed=471;
```

## Correlation Analysis of PlasticWasteDriver Feature Set

```bash
auda workflow run PlasticWasteDriverFeatureSet Correlation
```

## Importance Analysis of PlasticWasteDriver Feature Set

```bash
auda workflow run PlasticWasteGenerationPredictors FeatureImportances --contamination=0.2 --seed=471
```

## Neural Network Forecasting

```bash
auda workflow run PlasticWasteDrivers NNForecasting --seed=471
```

## Multivariate Forecasting

```bash
auda workflow run PlasticWasteDrivers MultivariateForecasting --seed=471 --location=Japan
auda workflow run PlasticWasteDrivers MultivariateForecasting --seed=471 --location=Slovenia --enable_tuning=0
```
