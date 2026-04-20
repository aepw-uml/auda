### Reproduction Commands

## Forecasting of Japan plastic waste generation

```bash
auda workflow run YearPWG Forecasting --location=Japan --seed=471 --tune_search_type=grid
```

## Forecasting of global plastics production

```bash
auda workflow run GlobalYearPlasticsProduction Forecasting --seed=471 --workflow_name="global_forecasting"
```

## Reconstruction of United States plastic waste generation

```bash
auda workflow run YearPWG Reconstruction '--location=United States' --seed=471
```

## Multiple Reconstruction of Japan polymer PET consumption

```bash
auda workflow run YearPPC MultipleReconstruction --location=Japan --seed=471;
```

## Neural Network Forecasting

```bash
auda workflow run PlasticWasteDrivers NNForecasting --seed=471
```

## Multivariate Forecasting

```bash
auda workflow run PlasticWasteDrivers MultivariateForecasting --seed=471 --location=Japan
```
