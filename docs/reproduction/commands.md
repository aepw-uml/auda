### Reproduction Commands

## Forecasting of Japan plastic waste generation

```bash
auda task run YearPWG Forecasting --location=Japan --seed=470
```

## Forecasting of global plastics production

```bash
auda task run GlobalYearPlasticsProduction Forecasting --seed=471
```

## Reconstruction of United States plastic waste generation

```bash
auda task run YearPWG Reconstruction '--location=United States' --seed=471
```

## Multiple Reconstruction of Japan polymer PET consumption

```bash
auda task run YearPPC MultipleReconstruction --location=Japan --seed=471;
```

## Neural Network Forecasting

```bash
auda task run PlasticWasteDrivers NNForecasting --seed=471
```
