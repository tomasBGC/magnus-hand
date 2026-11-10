# True-zero Contact classifier retrain -- before/after report

- Jig (Units 1+2) Contact rows: 332631
- Self-closure rows (already deployed): 8424
- True-zero rows: 119400 total, from 500 combinations
- True-zero TRAIN split: 400 combos, 95448 rows
- True-zero HELD-OUT split: 100 combos, 23952 rows (combo_index: [1, 7, 8, 9, 10, 16, 24, 34, 38, 39, 43, 45, 47, 48, 52, 64, 66, 67, 81, 97, 101, 107, 115, 118, 128, 147, 157, 176, 189, 193, 196, 202, 207, 208, 210, 219, 226, 230, 244, 254, 256, 258, 261, 265, 270, 272, 278, 286, 295, 297, 300, 301, 306, 310, 316, 324, 329, 339, 346, 347, 349, 356, 361, 364, 371, 372, 376, 379, 384, 385, 391, 396, 403, 408, 418, 423, 424, 426, 428, 429, 430, 431, 432, 435, 436, 437, 444, 449, 452, 454, 457, 460, 461, 462, 467, 479, 483, 488, 489, 494])

## (a) Regression check: held-out jig test set (test_data_backregion.csv)

| Model | Contact accuracy |
|---|---|
| Currently deployed | 0.7771 |
| True-zero fix | 0.7830 |
| Delta | +0.0060 |

## (b) True-zero HELD-OUT combos (never seen during retraining)

| Model | Contact accuracy (all should read 0) | False-positive rate |
|---|---|---|
| Currently deployed | 0.7986 | 0.2014 |
| True-zero fix | 0.9980 | 0.0020 |

### Per-finger false-positive rate, held-out true-zero combos

| Finger | Old FP rate | New FP rate |
|---|---|---|
| Index | 0.1762 | 0.0003 |
| Middle | 0.2350 | 0.0002 |
| Ring | 0.3918 | 0.0038 |
| Thumb | 0.0027 | 0.0037 |
