# True-zero dataset: false-nonzero-Force analysis

500 combinations x 4 fingers = 2000 (combo, finger) pairs analyzed. All captured with genuinely no touch anywhere -- ground truth Force is 0.0N for every single one.

## Headline

- **523/2000 (26.2%)** (combo, finger) pairs show nonzero displayed Force (>0.01N) despite no touch.
- Mean Force where nonzero: 1.888 N (max: 5.881 N)
- Mean confirmed-contact fraction where nonzero: 74.5%

## Per-finger breakdown

| Finger | Combos affected | Rate | Mean max_force (nonzero only) | Max max_force |
|---|---|---|---|---|
| Index | 100/500 | 20.0% | 2.265 N | 5.252 N |
| Middle | 159/500 | 31.8% | 2.151 N | 5.881 N |
| Ring | 264/500 | 52.8% | 1.588 N | 3.498 N |
| Thumb | 0/500 | 0.0% | 0.000 N | 0.000 N |

## Boundary case (Index=Thumb=15%) vs. general combinations

- Boundary (Index=Thumb=15%): 13/60 (21.7%) nonzero
- General (LHS): 510/1940 (26.3%) nonzero

## Position-range patterns

For each finger, mean/median position of ITS OWN commanded pose, split by whether IT read nonzero Force, and mean position of the OTHER 3 fingers at those same moments (checks whether nonzero readings correlate with a NEIGHBOUR's position -- i.e. residual crosstalk -- rather than the reading finger's own position).

| Finger | Own pos (nonzero rows) | Own pos (zero rows) | Mean OTHERS' pos (nonzero) | Mean OTHERS' pos (zero) |
|---|---|---|---|---|
| Index | 13.2% | 33.2% | 49.9% | 41.6% |
| Middle | 40.1% | 54.2% | 37.6% | 35.9% |
| Ring | 48.5% | 52.6% | 36.7% | 35.7% |
| Thumb | nan% | 29.7% | nan% | 43.1% |

### Extreme-position correlation (0-15% or 85-100% on ANY finger)

Checks whether nonzero readings cluster near the travel extremes (matching this project's own prior findings -- crosstalk-fit residual and self-motion noise both documented as worst near 0%/100%).

- At least one finger in [0,15] or [85,100]: 523/2000 (26.2% nonzero)
- All fingers in (15,85): n/a

## Worst individual (combo, finger) cases

| combo_index | finger | Index | Middle | Ring | Thumb | max_force | confirmed_contact_fraction |
|---|---|---|---|---|---|---|---|
| 242 | Middle | 46 | 30 | 74 | 9 | 5.881 | 100.0% |
| 412 | Middle | 75 | 25 | 96 | 13 | 5.798 | 100.0% |
| 363 | Middle | 36 | 30 | 87 | 13 | 5.508 | 100.0% |
| 232 | Index | 1 | 70 | 40 | 67 | 5.252 | 100.0% |
| 137 | Index | 1 | 83 | 59 | 74 | 5.126 | 100.0% |
| 134 | Middle | 32 | 26 | 75 | 10 | 5.114 | 100.0% |
| 90 | Index | 2 | 35 | 74 | 88 | 4.927 | 48.3% |
| 499 | Middle | 12 | 4 | 89 | 75 | 4.906 | 100.0% |
| 377 | Middle | 56 | 22 | 64 | 7 | 4.870 | 100.0% |
| 226 | Middle | 13 | 10 | 82 | 11 | 4.833 | 100.0% |
| 195 | Middle | 12 | 10 | 76 | 11 | 4.727 | 100.0% |
| 305 | Middle | 29 | 29 | 86 | 4 | 4.674 | 100.0% |
| 275 | Middle | 83 | 29 | 70 | 12 | 4.576 | 100.0% |
| 23 | Middle | 10 | 2 | 74 | 35 | 4.417 | 100.0% |
| 464 | Middle | 51 | 33 | 67 | 1 | 4.245 | 100.0% |
| 452 | Index | 7 | 72 | 63 | 84 | 4.230 | 38.3% |
| 164 | Middle | 49 | 13 | 99 | 8 | 4.124 | 100.0% |
| 343 | Index | 0 | 45 | 13 | 66 | 4.103 | 100.0% |
| 267 | Index | 0 | 21 | 82 | 91 | 4.075 | 100.0% |
| 359 | Middle | 10 | 10 | 50 | 88 | 4.058 | 100.0% |
