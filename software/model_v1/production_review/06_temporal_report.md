# Temporal levers: rate features + feature-level EMA

All at the live 20Hz cadence (subsampled): train n=44904, test n=15699.

## (a) Rate-of-change features (GBM heads, same split)

- base: Force MAE=1.2105N (all test), Contact acc (U1+2)=0.8898
- base+rate: Force MAE=1.2021N (all test), Contact acc (U1+2)=0.8951

(If base+rate is not clearly better, rate features are not worth the live-pipeline statefulness they'd add.)

## (b) Feature-level EMA before the deployed model (no retraining)

| alpha (1.0 = no smoothing) | idle Force std (N) | idle p90 |Force| (N) | touch Force MAE (N) | mean release lag (ticks @20Hz) |
|---|---|---|---|---|
| 1.0 | 0.8186 | 0.6129 | 1.1300 | 11.63 |
| 0.6 | 0.7450 | 0.4662 | 1.1049 | 14.05 |
| 0.4 | 0.7305 | 0.4329 | 1.0926 | 15.23 |
| 0.25 | 0.7219 | 0.4351 | 1.0835 | 16.47 |
| 0.15 | 0.7196 | 0.4297 | 1.0888 | 20.10 |

EMA on the delta features is mathematically upstream of the existing FORCE_SMOOTHING_WINDOW display average -- if adopted it should replace (not stack on) part of that, and the release-lag column is the cost side of the trade.