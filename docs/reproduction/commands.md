### Reproduction Commands

## Forecasting of global plastics production

```bash
auda task run YearPWG Forecasting --location=Japan --seed=471
```

## Forecasting of Japanese plastic waste generation

```bash
auda task run YearPWG Reconstruction --location=Japan --seed=471
```

## Reconstruction of global plastics production

```bash
auda task run YearPPC reconstruction year_ppc --location=Japan --seed=149
```

## Multiple Reconstruction of global plastics production

```bash
auda module run multiple_reconstruction year_ppc --location=Japan --seed=120;
```
