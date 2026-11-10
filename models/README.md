# Models

Each model is a Python pickle (loaded with `joblib`). **Loading a pickle can
run code, so only load these files from a source you trust**, and check them
against `SHA256SUMS.txt` first. The files were written with scikit-learn and
are best loaded with the versions in `requirements.txt` (the analysis in this
repository was re-run with scikit-learn 1.9.0).

The three contact-classifier models share one bundle format: a Force/Theta
regressor, a direction (`Phi`) classifier, a contact classifier and a shared
scaler. Only the contact classifier differs between them (checked by comparing the
weights); the first two also carry a probability calibrator that the deployed
model no longer has.

| Folder | What it is | Which paper numbers it explains |
|---|---|---|
| `before_true_zero/` | The **old contact classifier** (the version trained on rig data plus the "closed finger, no touch" examples, before the true-zero data was added). | The "before" column of the before/after comparison: on held-out no-touch hand positions it reports contact when nothing is touching **20.1% of the time** (20.14%; per finger Index 17.6%, Middle 23.5%, Ring 39.2%, Thumb 0.3%). |
| `after_true_zero/` | The **newer contact classifier**, retrained with the true-zero data: 400 of the 500 no-touch position combinations were used for training and the other 100 were held out for testing. | The "after" column: **0.2%** false positives on the 100 held-out combinations (0.20%; Index 0.03%, Middle 0.02%, Ring 0.38%, Thumb 0.37%), with accuracy on the rig's genuine-touch test set unchanged (77.7% to 78.3%). |
| `deployed/` | The model **currently used live on the hand**. Its contact classifier was retrained again on 2026-09-07 using all 500 true-zero combinations plus the training split of the grasp data (`data/load_cell_dataset.csv`). The Force/Theta regressor, direction classifier and scaler are the same as in the two models above. | The per-finger contact accuracy in the paper (Index 99.0%, Middle 93.1%, Ring 97.7%, Thumb 52.9% on the grasp holdout) and the scaler used by the force analysis. Used by `analysis/` as the contact classifier and the source of the scaler. |
| `grasp_trained_force/` | The **offline grasp-trained force model** from the paper analysis: a force-only neural network trained on the grasp data's training split, using the deployed model's scaler. It is a research candidate and is **not** deployed. | The grasp-trained force results on the standard split: MAE 0.210 N, RMSE 0.312 N, r = 0.916. (The leave-one-scenario-out numbers come from seven models refitted on the fly by `analysis/`; they are not stored.) |

Where each number is recorded: `analysis/RESULTS_LEDGER.md`. To regenerate
the force model, run `python analysis/train_force_candidate.py`; its
output gives the same predictions and error figures as the stored file
(checked when this repository was assembled).

Caveats worth knowing:

* The deployed contact classifier was itself trained on the grasp data's
  training split, so its accuracy on the grasp holdout is a fair held-out
  test only because the same split is reproduced exactly (same seed). It is
  not an independent dataset.
* The 20.1% and 0.2% figures are offline tests on held-out no-touch hand
  positions, not live hardware tests. The older model was briefly reverted to
  in July after live testing showed reduced light-touch recall with the newer
  one; see `software/model_v1/true_zero_dataset/` for that history.
* `SHA256SUMS.txt` lists the checksum of every file here.
