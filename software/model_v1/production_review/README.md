# production_review — 2026-07 production-readiness review

**Status: partially promoted and live (2026-07-30).** This review pipeline
itself is standalone and additive: it reads the deployed model, datasets,
and crosstalk artifacts, and writes only inside this folder, never touching
`active_model.pkl` directly. However, two of its outputs (the Phi head
retrained without Theta=0 rows, and the isotonic `contact_calibrator`) were
subsequently reviewed and promoted into `model_v1/active_model.pkl` on
2026-07-17. The Phi head remains live; the isotonic `contact_calibrator` was
removed again on 2026-09-07 when the Contact classifier was retrained (it was
fitted to the old classifier's outputs). The regressor and Contact
classifier in `active_model_review_candidate.pkl` were carried over
byte-identical from the model deployed at review time rather than retrained
here. See `model_v1/README.md`'s "How `active_model.pkl` was built" section
for the current full lineage. Every other candidate in this folder (GBM
heads, rate-of-change features, GBM Contact head, pooled-data calibration)
was evaluated and rejected, and was never deployed.

Start with **`00_REVIEW_SUMMARY.md`** (findings, ranked recommendations,
honest negatives, hardware-data list) and **`DEPLOYMENT_INSTRUCTION.md`**
(the ready-to-paste instruction that deploys the validated items).

| File | What it is |
|---|---|
| `01_baseline_eval.py` / `01_baseline_report.md` | Fine-grained baseline of the deployed model: per unit/theta/phi/recording, force bands, Contact+Phi calibration, training-coverage census |
| `02_architecture_comparison.py` / `02_architecture_report.md` | GBM heads vs deployed MLP heads on the identical split (mixed result — no full swap) |
| `03_scale_invariant_features.py` / `03_scale_invariant_report.md` | Leave-one-unit-out transfer with direction+log-magnitude features (Contact: partial win; Phi: no) |
| `04_calibration.py` / `04_calibration_report.md` | Isotonic calibration + probability-Schmitt contact policy (deployed as recommendation #1/#5) |
| `05_robustness_stress.py` / `05_robustness_report.md` | Sub-guard glitch injection, OOD scale sweep, noise gain, zero-input offset |
| `06_temporal_and_smoothing.py` / `06_temporal_report.md` | Rate-of-change features (rejected) + feature-EMA sweep (optional) at the live 20 Hz cadence |
| `07_crosstalk_ridge.py` / `07_crosstalk_ridge_report.md` | Origin-anchored quadratic-ridge crosstalk backend + repeated-CV validation |
| `08_phi_head_no_theta0.py` / `08_phi_head_report.md` | Candidate Phi head retrained without degenerate theta=0 rows |
| `09_build_candidate_bundle.py` | Assembles + regression-guards `active_model_review_candidate.pkl` |
| `ridge_crosstalk_model.py` + `candidate_crosstalk_ridge_v1.json` | Drop-in 4th crosstalk backend (load_model/predict_shift contract) |
| `candidate_gbm_heads.pkl`, `candidate_phi_no_theta0.pkl`, `candidate_calibrators.pkl` | Candidate artifacts (not deployed) |
| `active_model_review_candidate.pkl` | The deployable bundle (deployed model + new Phi head + contact calibrator) |
