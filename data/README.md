# Data

The datasets are plain CSV files; the folder `closed_loop/` holds two live recordings as JSON and CSV (see
its own README). The data are licensed CC BY 4.0 (`LICENSE-HARDWARE-DOCS-DATA.md` in the repository root).
"Tick" means one sensor reading. Each fingertip has
five magnetometers (S1 to S5), each with X, Y, Z axes. Columns starting with
`d` (for example `dS1X`) are the reading minus a resting baseline, and
`S1_mag` to `S5_mag` are the field strengths.

## Files included

| File | Rows | Size | What it is | How it was produced |
|---|---|---|---|---|
| `master_characterisation.csv` | 202,196 | 41.6 MB | Calibration-rig recordings for the first fingertip unit: raw sensor readings with the known press force, tilt (`Theta`) and direction (`Phi`). | Each recording session is saved as a JSON file and then appended here, one row per tick, by `software/recording_analysis.py`. |
| `master_characterisation_2.csv` | 233,784 | 48.6 MB | The same, for the second fingertip unit. | Same. |
| `master_characterisation_3.csv` | 50,485 | 12.4 MB | The same, for a third unit; a supplement covering the back region of the fingertip. | Same. |
| `load_cell_dataset.csv` | 18,067 | 8.1 MB | Grasp data from the real hand: 140 grasp repetitions in 7 finger combinations (`scenario`), at light, medium and firm grip levels plus no-touch "control" repetitions. Each row pairs the finger's sensor readings with the force measured by a separate load cell. Only the finger facing the load cell is labelled in each repetition. **This is the only input to the paper analysis.** | `software/model_v1/grasping_dashboard/build_load_cell_dataset.py`, from the grasp-session recordings. Only verified repetitions and the steady holding phase are used. |
| `selfclosure_contact_dataset.csv` | 8,424 | 3.7 MB | "No touch" examples with the fingers closed to 0, 25, 50, 75 and 100% on the hand, so the classifier learns that a closing finger on its own is not contact. | `software/model_v1/contact_classifier_fix/build_selfclosure_contact_dataset.py`, from the crosstalk recordings. |
| `true_zero_dataset.csv` | 119,400 | 55.4 MB | "No touch" examples with all four fingers moving at once over 500 position combinations. Force is zero by construction. | `software/model_v1/true_zero_dataset/build_true_zero_dataset.py`, from the continuous no-touch recordings. |
| `crosstalk_dataset.csv` | 900 | 0.1 MB | How moving one finger changes the readings of the others (one row per source finger, position, affected finger and channel). | `software/model_v1/crosstalk/build_crosstalk_dataset.py`. |
| `combined_dataset.csv` | 450 | 0.07 MB | The same measured with two fingers moving together, used to test whether crosstalk effects simply add up. | `software/model_v1/crosstalk/build_combined_dataset.py`. |
| `crosstalk_lookup_dataset.csv` | 6,000 | 0.6 MB | Crosstalk measured over many random combinations of finger positions, for the nearest-neighbour lookup model. | `software/model_v1/crosstalk_lookup/build_lookup_dataset.py`. |

Also included: `closed_loop/`, three files from two live grip recordings of the Force-Target Grip controller
on the real hand (21 July and 8 September 2026): `slip_detection_live_verification.json`,
`eflesh_test_20260908T121329Z grasp test_meta.json` and `eflesh_test_20260908T121329Z grasp test.csv`. They
are not used by the paper analysis scripts; the counts derived from them are in the last section of
`analysis/RESULTS_LEDGER.md`. See `closed_loop/README.md` for contents, provenance and checksums.

## Which script reads which file

| File | Read by |
|---|---|
| `master_characterisation*.csv` | `software/model_v1/build_delta_dataset_backregion.py` |
| `load_cell_dataset.csv` | `analysis/*.py` (through `analysis/common.py`) and `software/model_v1/grasping_dashboard/retrain_contact_with_load_cell_data.py` |
| `selfclosure_contact_dataset.csv` | `contact_classifier_fix/evaluate_contact_fix.py` and `retrain_contact_classifier.py`; `grasping_dashboard/retrain_contact_with_load_cell_data.py`; `true_zero_dataset/retrain_contact_from_true_zero.py`; `production_review/02_architecture_comparison.py` and `09_build_candidate_bundle.py` |
| `true_zero_dataset.csv` | `true_zero_dataset/true_zero_common.py` (and the scripts using it, including `retrain_contact_from_true_zero.py`); `grasping_dashboard/retrain_contact_with_load_cell_data.py` |
| `crosstalk_dataset.csv` | `crosstalk/crosstalk_common.py`, used by `train_crosstalk_model.py` and `evaluate_crosstalk_model.py` |
| `combined_dataset.csv` | `crosstalk/validate_additivity.py` |
| `crosstalk_lookup_dataset.csv` | `crosstalk_lookup/crosstalk_lookup_common.py` |

(Scripts are under `software/model_v1/`.) Apart from the analysis scripts,
which have been updated for this repository, the training scripts still look
for these files at their original locations (next to the script, or under
`model_v1/` and `recordings/`). Copy or point them to the files above to run
them.

## Not included

* **Raw recordings.** The per-session JSON files behind the master CSVs, the
  grasp-session episode files, the crosstalk, true-zero and lookup raw
  captures, and the characterisation videos. Too large, and not needed to
  reproduce the paper numbers.
* **The three `*_backregion.csv` files** (`characterisation_delta_combined_backregion.csv`,
  `train_data_backregion.csv`, `test_data_backregion.csv`; roughly 218, 160 and
  56 MB). They are derived from the master CSVs by
  `build_delta_dataset_backregion.py` and `build_split_backregion.py`, so they
  can be regenerated. The model training scripts and the older `production_review`
  scripts need them.
* **Anything over 100 MB**, and other derived or superseded datasets (for
  example the earlier self-labelled grasp datasets, which were found to
  contain wrong contact labels and were discarded).

**Hosting:** the three `*_backregion.csv` files are available on request, to
be deposited in a data repository after review. Their details, so a copy can
be verified:

| File | Size (bytes) | Data rows | SHA-256 |
|---|---|---|---|
| `characterisation_delta_combined_backregion.csv` | 217,808,777 | 486,465 | `bd7496211548779add41e41d0c202509f017423cd51138016a212abe1246b6fe` |
| `train_data_backregion.csv` | 160,472,102 | 360,404 | `885fb3e03d79938e74adcbc5bf89a78804d0ae186543c2725c8671a022fe0e2f` |
| `test_data_backregion.csv` | 56,100,023 | 126,061 | `4ec6aced5bb7ad60b7c5fab926639aaefb9d2cc5338506e5ee7e2294ed415839` |

Row counts exclude the header line. The raw recordings are likewise available
on request.
