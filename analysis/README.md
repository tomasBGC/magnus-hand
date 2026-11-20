# analysis

The scripts that compute every number and figure for the paper, with their outputs. They read
`data/load_cell_dataset.csv` and `models/deployed/active_model.pkl`, find those from their own location,
and run from any directory. Set `EFLESH_OUT_DIR` to write results, figures and the force model to another
folder instead of overwriting the committed ones. Every result is offline-only (see `RESULTS_LEDGER.md`).

| File or folder | What it is |
|---|---|
| `common.py` | Shared setup: paths, the rep-grouped 80/20 split (seed 42), the force model and metrics. |
| `a_contact_accuracy.py` | (a) Deployed contact classifier accuracy per finger on the standard holdout. |
| `b_force_error.py` | (b) Force MAE, RMSE and correlation per finger, standard split and leave-one-scenario-out, with the constant-mean baseline. |
| `c_level_confusion.py` | (c) Light/medium/firm confusion matrices from predicted force. |
| `d_control_bias.py` | (d) Predicted-force bias on no-touch (control) rows. |
| `e_level_stats.py` | (e) Per-finger, per-level mean and SD of true and predicted force. |
| `f_predictions.py` | (f) Row-level predictions used by the figures. |
| `g_rep_counts.py` | (g) Number of repetitions per finger and grip level. |
| `train_force_candidate.py` | Retrains the grasp-trained force model and saves it to `models/grasp_trained_force/`. |
| `two_level.py` | Light-versus-firm analysis; writes only to `results_two_level/`. |
| `build_ledger.py` | Builds `RESULTS_LEDGER.md` from `results/*.csv`. |
| `fig_style.py`, `fig1_pred_vs_true.py`, `fig2_level_confusion.py`, `fig3_level_means.py` | Shared figure style and the three paper figures. |
| `fig_two_level.py` | The same three figures for the 1.0 N two-level analysis; writes `figures_two_level/`. |
| `results/` | CSV outputs of the scripts above. |
| `results_two_level/` | CSV outputs of `two_level.py`. |
| `figures_two_level/` | PDF figures and PNG previews from `fig_two_level.py`. |
| `RESULTS_LEDGER.md` | Every reported number with its script, input, split, seed and status. |
| `_input_hashes_before.txt` | SHA-256 of the model and dataset the analysis reads, recorded before the scripts were run. |
