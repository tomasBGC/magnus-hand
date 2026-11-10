# Robustness stress-test report

Base frames: 3000 idle Units-1+2 test frames (true Force<0.3N), plus a median-force real touch frame for the scale sweep.

## 1. Single-channel glitches below the SANE_DELTA_MAX=2000 guard

Spike added to one random channel per frame (sign random). Frames whose corrupted deltas exceed 2000 are excluded (the live guard catches those). 'confirm-feed rate' = fraction with Contact==1 AND Force>0.1N, i.e. frames that would push the live confirm streak; 5 consecutive such frames = a false CONTACT on screen.

| Spike (raw units) | Deployed confirm-feed rate | Deployed mean Force if confirmed (N) | GBM confirm-feed rate | GBM mean Force (N) |
|---|---|---|---|---|
| 300 | 0.2197 | 7.61 | 0.1313 | 1.75 |
| 600 | 0.1953 | 10.27 | 0.1407 | 1.65 |
| 1000 | 0.1563 | 10.73 | 0.1357 | 1.74 |
| 1500 | 0.1700 | 10.40 | 0.1267 | 1.73 |
| 1900 | 0.1667 | 9.46 | 0.1350 | 1.66 |

## 2. Scale sweep on a real touch frame (OOD extrapolation)

Base frame: true Force=4.64N, Unit 1, theta=60, phi=-90.

| Scale | Deployed Force (N) | GBM Force (N) | Deployed Contact | GBM Contact |
|---|---|---|---|---|
| x0.25 | 1.35 | 0.68 | 1 | 1 |
| x0.5 | 4.51 | 4.07 | 1 | 1 |
| x1 | 3.23 | 3.95 | 1 | 1 |
| x2 | 5.99 | 6.92 | 1 | 1 |
| x4 | 3.16 | 6.12 | 1 | 0 |
| x8 | -1.77 | -0.16 | 1 | 1 |
| x15 | -4.49 | -0.71 | 1 | 1 |
| x30 | -14.28 | 0.51 | 1 | 1 |

## 3. Noise gain (sigma=2.5 raw units, 50 draws per frame)

- idle frames: deployed Force std=0.5150N per 2.5-unit sensor noise; GBM std=0.3935N
- touch frames: deployed Force std=0.4651N per 2.5-unit sensor noise; GBM std=0.3842N

This is the model's own noise amplification -- the floor that display smoothing has to fight. Lower = calmer displayed Force at rest and under steady press.

## 4. Exact-zero delta (perfect baseline)

- Deployed: Force=+0.3910N, Contact=0
- GBM:      Force=+0.1334N, Contact=0
