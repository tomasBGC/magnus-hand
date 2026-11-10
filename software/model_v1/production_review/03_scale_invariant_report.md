# Scale-invariant features: cross-unit transfer test

## Contact classifier -- leave-one-unit-out

| Train unit(s) | Test unit (test-set rows) | Raw-feature acc | Invariant-feature acc |
|---|---|---|---|
| [1] | 2 | 0.8345 | 0.8399 |
| [1] | 3 | 0.7054 | 0.6948 |
| [2] | 1 | 0.9251 | 0.9255 |
| [2] | 3 | 0.4680 | 0.7462 |
| [1, 2] | 3 | 0.5372 | 0.7242 |

Within-distribution (train U1+2 -> test U1+2): raw=0.8808, invariant=0.8831

## Phi classifier (theta>=30 rows only) -- leave-one-unit-out

Unit 3 has no theta>=30 rows with Phi in {-90,0,90} in test except theta=90 -- Unit 3's test recordings ARE theta=90, so this is exactly the back-region transfer question.

| Train unit(s) | Test unit | Raw acc | Invariant acc | n test |
|---|---|---|---|---|
| [1] | 2 | 0.8544 | 0.8132 | 43296 |
| [1] | 3 | 0.5104 | 0.4218 | 22712 |
| [2] | 1 | 0.7233 | 0.7769 | 27889 |
| [2] | 3 | 0.5715 | 0.6649 | 22712 |
| [1, 2] | 3 | 0.5512 | 0.3602 | 22712 |
