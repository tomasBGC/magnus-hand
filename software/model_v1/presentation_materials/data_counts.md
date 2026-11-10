# Exact recordings & samples used to train the currently deployed model

Every number below is either a direct row count of a file actually referenced by the
training scripts that built `model_v1/active_model.pkl`, or a count taken directly
from the recording folders. Numbers that fed the bundle currently loaded by
`model_v1/sensing_gui.py` are marked **[DEPLOYED]**; datasets that exist in the repo
but are confirmed NOT part of this bundle are marked **[NOT DEPLOYED]** for clarity.

## 1. Original bench characterisation (jig-mounted units)

Force/direction ground truth collected by pressing the fingertip against a load cell
at fixed angles while logging the magnetic sensor array in sync.

| Unit | Recordings | Raw samples (rows) |
|---|---|---|
| Unit 1 | 21 | 202,196 |
| Unit 2 | 22 | 233,784 |
| **Units 1+2 combined** | **43** | **435,980** |

Source: `recordings/master_characterisation.csv`, `recordings/master_characterisation_2.csv`.

Split for training: **[DEPLOYED]** train/test split across all 3 units (see #2) gives
a jig-only (Units 1+2) training pool of **332,631 rows** — this is what actually
trains the Force/Theta regressor's jig contribution, the base Contact classifier
component, and (after further filtering, see #3) the Phi/direction classifier.

## 2. Back-region supplement (Unit 3, hand-held, theta=90 only)

| | Recordings | Rows |
|---|---|---|
| Unit 3 raw | 5 | 50,485 |

Source: `recordings/master_characterisation_3.csv`.

**[DEPLOYED — regressor only]** Unit 3 rows ARE merged into the Force/Theta
regressor's training data (all 3 units combined, 360,404 rows train + 126,061 rows
held-out test = 486,465 total). **[NOT part of Phi/Contact training]** — the Phi and
Contact classifiers are deliberately kept jig-only (Units 1+2), so Unit 3 does not
feed those two components.

## 3. No-theta0 Phi (direction) head retrain — currently deployed Phi model

Same 332,631-row jig-only pool as #1, with all rows where the press angle (theta) was
exactly 0° removed, because a straight-on press carries no usable direction
information and was found to be actively confusing the direction model.

| | Rows |
|---|---|
| Jig-only pool before filtering | 332,631 |
| Theta=0 rows removed | 44,723 |
| **Rows actually used to train the deployed Phi head** | **287,908** |

No new recordings — this is a refiltering of the same physical data as #1.

## 4. Self-closure supplementary dataset — folded into deployed Contact classifier

Fixes the case where a finger closing against its own mechanism (no external object)
was being misread as a real touch.

| | Recordings | Rows |
|---|---|---|
| Self-closure dataset | 0 new recordings (re-derived from the same 20 crosstalk recordings, see #6) | 8,424 |

Collected on this specific physical hand's own 4 fingertips.

## 5. True-zero dataset — folded into deployed Contact classifier

Realistic simultaneous multi-finger poses, hand genuinely touching nothing, used to
fix the broader multi-finger false-alarm problem.

| | Combinations (recordings) | Rows |
|---|---|---|
| Total recorded | 500 | 119,400 |
| Used for training | 400 | 95,448 |
| Held out for testing (not trained on) | 100 | 23,952 |

Collected on this specific physical hand's own 4 fingertips.

## Deployed Contact classifier — total training data

| Source | Rows |
|---|---|
| Jig (Units 1+2) | 332,631 |
| Self-closure | 8,424 |
| True-zero (train split) | 95,448 |
| **Total** | **436,503** |

## 6. Crosstalk (finger-interference) correction — two different backends deployed

The deployed system uses a **different correction method per finger**, chosen
per-finger based on which performed best in live testing:

| Finger | Backend | Recordings | Rows |
|---|---|---|---|
| Index | nearest-neighbour (smoothed) | 100 simultaneous 4-finger position combinations | 6,000 |
| Middle, Ring, Thumb | pairwise-additive | 20 (4 fingers × 5 positions each, one finger moved at a time) | 900 |

A third backend ("ridge regression") was fit on the same 100-combination/6,000-row
dataset as the Index backend and measured better on paper, but caused a live failure
(two fingers got stuck reporting "touching" after a grip-and-release test) and was
**reverted — it is not active for any finger in the currently deployed system.**

## 7. Data NOT used to train the deployed model

- **`model_v1/self_motion/` dataset (5,051 rows)** — purely diagnostic. It was used to
  measure how much noise a finger's own movement injects into its own readings, and
  that finding motivated the self-closure dataset above, but the self-motion data
  itself was never fed into any trained model.
- **No separate per-unit "fine-tuning" dataset exists beyond #4 and #5 above** — those
  two datasets, collected specifically on this hand's own fingertips, *are* the
  per-unit calibration data.

## Grand total: distinct physical recording/capture sessions feeding the deployed model

- 43 jig bench recordings (Units 1+2)
- 5 hand-held back-region recordings (Unit 3, regressor only)
- 500 true-zero multi-finger combinations (400 used in training)
- 20 crosstalk position-sweep recordings (also source of the self-closure data, no
  double-count)
- 100 simultaneous 4-finger crosstalk combinations (Index's correction backend)

**≈ 668 distinct physical recording/capture sessions** (43 jig + 5 back-region + 500
true-zero combinations + 20 crosstalk sweeps + 100 crosstalk-lookup combinations),
spanning **621,189 raw captured rows** across all independently-collected source
datasets (435,980 jig + 50,485 back-region + 119,400 true-zero + 900 crosstalk
pairwise-additive + 6,000 crosstalk-lookup + 8,424 self-closure). Individual model
components each train on a filtered subset of this pool, not the full 621,189 rows —
see the per-component tables above for exactly what each one uses (e.g. the deployed
Contact classifier uses 436,503 rows, the deployed Phi head uses 287,908).
