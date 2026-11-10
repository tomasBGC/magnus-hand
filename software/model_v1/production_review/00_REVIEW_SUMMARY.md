# Production-readiness review (2026-07-16/17)

Full review of model_v1's deployed sensing stack (active_model.pkl, the
sensing_gui.py pipeline, and the crosstalk backends) against a
production-on-a-robotic-hand bar: accuracy, robustness, noise, calibration,
and data coverage. All experiments live in this folder (01..09) and are
additive/read-only — **nothing deployed was modified**; every candidate
artifact is a new file. Numbers are from the untouched held-out test set
(`test_data_backregion.csv`, 12 recordings) unless stated otherwise.

## Where the deployed model actually stands (01_baseline_report.md)

The headline numbers previously reported for this model (88.4% Contact,
~0.6 N Force MAE) are Units-1+2-only slices. The full held-out picture:

| Slice | Force MAE | Theta MAE | Phi acc | Contact acc |
|---|---|---|---|---|
| Units 1+2 (deployed-hand regime) | 0.94 N | 12.3–14.9° | 0.64–0.67 | 0.86–0.92 |
| Unit 3 (back-region supplement) | 3.10 N | 7.9° | 0.17 | 0.27 (Contact FN rate 99.5%) |
| theta=90 back region, all units | 2.44 N | 14.3° | 0.42 | 0.46 |
| theta=0 (Phi physically degenerate) | — | — | 0.15 | — |

Other load-bearing baseline facts:

- **Phi confidence was untrustworthy**: 51% accuracy in the 0.95–0.99
  confidence bucket (ECE 0.26). Driven by the degenerate theta<30 rows and
  the back region — in the physically meaningful regime (U1+2, theta>=30)
  Phi is fine (0.84 acc, ECE 0.011).
- **Contact probabilities on U1+2 are decent but uncalibrated at the low
  end** (ECE 0.053); on Unit 3 they are meaningless (ECE 0.73) — the model
  does NOT know when it doesn't know on out-of-scale units.
- **Idle Force bias** +0.35 N mean / p90 0.60 N (why FORCE_DEADBAND_N=0.4
  exists). At a perfect zero delta the regressor still outputs +0.39 N.
- **Data coverage**: 70 of the notional 105 (unit,theta,phi) cells were
  never recorded; every recorded cell is a SINGLE recording; Phi=±45 exists
  only at Unit3/theta=90, one recording each, train-only. Contact training
  is 75%/25% touch/idle.

## Ranked recommendations

Ordered by expected impact vs. implementation risk. "Validated" = held-out
offline evidence in this folder; live A/B is still the final gate where
noted. Items 1–5 are packaged, ready to deploy, and covered step-by-step in
`DEPLOYMENT_INSTRUCTION.md`.

### 1. Contact-probability Schmitt trigger (validated; pipeline change)

Replace `predict()==1` in the live hysteresis with two thresholds on the
classifier's own probability (confirm at P>=0.6 + the existing 0.1 N force
gate, release at P<=0.3, hold state in between). On held-out U1+2 rows this
improves BOTH failure directions simultaneously: idle frames feeding the
confirm streak 0.0633 → 0.0454 (−28%), genuine-press frames pressuring
release 0.1277 → 0.0908 (−29%). Cost: 2–6 pp confirm sensitivity in the
already-weak <1 N band (quantified in 04's addendum; a P_hi=0.8 "calm mode"
exists but costs 13–17 pp there — documented, not default). Risk: LOW —
constants + ~10 lines in `_process_one_finger`, flag-gated.

### 2. Transient spike guard (validated; robustness hole closed)

The only input guard today is |delta|>2000. Injected single-channel spikes
below that (300–1900 raw units) put the deployed model at 13–22% false
confirm-feed frames with 7.6–10.7 N phantom forces (05). Measured genuine
per-tick jumps at the live 20 Hz cadence never exceed 19.4 raw units on
Units 1+2 — including during presses — so a 300-unit/tick jump guard has
~15x margin. One-tick hold semantics (suppress a jump once; accept it if it
persists) bound the worst-case genuine-touch delay at 50 ms. Risk: LOW.

### 3. Phi head without theta=0 + honest Phi display gating (validated)

At theta=0 the azimuth label is physically meaningless; those rows are
label noise (the deployed head scores 15% there — below the 33% chance
floor). Retraining ONLY the Phi head without them (identical architecture,
deployed scaler reused — the exact single-head-swap mechanism the contact
fix already validated): back-region (theta=90) Phi 0.416 → 0.644, meaningful
regime 0.837 → 0.841 (no regression), theta=15 drops (0.49 → 0.31) but sits
below the display gate anyway. Coupled GUI change: show `Phi: --` unless
confirmed contact AND |Theta|>=25° AND max-prob>=0.7 — at that gate,
displayed Phi is 98.8% accurate at ~71% coverage. Candidate bundle built
and regression-guarded in 09 (`active_model_review_candidate.pkl`); Force/
Contact heads byte-identical. Risk: LOW.

### 4. Ridge crosstalk backend for Middle + Index (validated offline; live A/B gate)

Origin-anchored quadratic ridge fit on the 100 directly-measured 4-finger
combinations (corrected target). 5x repeated 5-fold CV across all 100
combos: Middle 46.8 → 41.7 MAE (−11% vs its current pairwise backend),
Index 18.2 → 17.7 (vs current NN-smoothed), Ring tie, Thumb within noise
floor. Structurally it has BOTH properties that decided previous backend
choices: exact zero at the reference position by construction, and
continuity (no Voronoi jumps) — while being fitted on real simultaneous-
4-finger data (no additive assumption). Module + model JSON are packaged
(`ridge_crosstalk_model.py`, `candidate_crosstalk_ridge_v1.json`), drop-in
compatible with `CROSSTALK_CORRECTION_BACKENDS`. Risk: MEDIUM until the
counterbalanced `realistic_session_diagnostic` A/B confirms it live —
offline MAE mispredicted live behaviour once before (plain NN), so the
per-finger switch is explicitly reversible and A/B-gated.

### 5. Surface calibrated Contact confidence (validated, display/logging only)

The bundle now carries an isotonic calibrator (U1+2-fit): test ECE
0.053 → 0.007 on U1+2. Use it for display/diagnostics ("Contact 92%")
rather than decisions (the Schmitt thresholds in #1 were tuned on raw
probabilities and work as-is). Explicitly NOT valid for Unit-3-scale
inputs (its metadata says so). Risk: negligible.

### 6. Optional, default-off: feature-level EMA (alpha 0.4–0.6)

Idle predicted-Force p90 0.61 → 0.43 N (−29%) and touch MAE slightly
IMPROVES (1.13 → 1.09 N), at +120–180 ms release lag (06). Worth a flag for
demos where calm matters more than latency; not a default because the
project has repeatedly and correctly prioritised responsiveness.

### 7. Candidate for a later pass: GBM Force head (keep MLP Theta)

GBM force regression beats the MLP on Force MAE (1.33 → 1.22 all-test;
0.94 → 0.79 on Unit 1), has 24% lower noise gain (0.52 → 0.39 N of Force
jitter per 2.5-unit sensor noise), one-third the zero-input offset
(+0.39 → +0.13 N — headroom to shrink FORCE_DEADBAND_N), saturates
gracefully on OOD input where the MLP extrapolates to −14 N, and needs no
scaler at all (the entire scaler-dilution failure class disappears). Not
in this deployment because it changes the runtime prediction path (two
models where one regressor lives today) and has no live hours; propose as
the next A/B after items 1–5 settle.

## Tested and REJECTED (honest negatives)

- **Full GBM architecture swap**: Theta regresses badly in-domain
  (12.3° → 22.4° on Unit 1) and in-domain Phi is worse; only Force/back
  region improve. No clean win (02).
- **Rate-of-change features**: Force MAE 1.2105 → 1.2021, Contact +0.5 pp —
  not worth adding statefulness to the live feature path (06).
- **Post-hoc Phi confidence calibration**: does not transfer (ECE unchanged
  on the slices that matter — 04). The structural fix is #3's gating.
- **Contact GBM head**: idle FP better (0.055 vs 0.064) but touch FN worse
  (0.139 vs 0.127) — a wash; keep the deployed head + Schmitt (02, 04).
- **Pooled-data isotonic for Contact**: Unit 3 contamination drives the
  calibrator floor to 0.304, above any usable release threshold — a
  once-confirmed contact could NEVER release. Caught in review; calibrator
  refit on U1+2 only (04). A cautionary example of why every one of these
  changes carries a release-health check, not just accuracy numbers.

## What needs NEW hardware data (cannot be fixed in software)

Ranked by expected payoff:

1. **Back-region presses on the DEPLOYED hand's own fingertips** (the
   selfclosure-dataset trick, but with real presses at theta≈90 against the
   load scale). The back region is the single biggest accuracy hole
   (Force MAE 2.44 N, Contact acc 0.46) and Unit 3 — hand-held, 10–40x
   scale, one channel visibly corrupted (dS4Y std 12,531 vs 9.4/9.8 on
   Units 1/2) — is a poor proxy for it.
2. **Phi=±45 coverage beyond one recording at one theta on one unit** —
   currently the ±45 classes are unlearnable-in-general and untestable.
3. **Repeats.** Every (unit,theta,phi) cell is ONE recording, so
   session-to-session variance is invisible to every number in this review.
   2–3 repeat recordings of a handful of cells would bound it.
4. **Light-touch band enrichment (0.3–1 N)**: confirm sensitivity there is
   29–51% even before any threshold change; slow-ramp recordings would
   directly improve the contact boundary.
5. **Crosstalk extremes**: repeated sweeps near 0%/100% for Index<-Middle
   (its documented residual noise source) and lookup combos concentrated in
   the rest-position corners.
6. **Cross-unit transfer** (fingertip swap resilience): tested in 03 —
   scale-invariant features (delta direction + log-magnitude) lift Contact
   transfer to a never-seen unit substantially (train U1+2 → test U3:
   0.537 → 0.724; train U2 → U3: 0.468 → 0.746) at zero in-domain cost
   (0.881 → 0.883), but 72–75% is still nowhere near production-grade, and
   the same features HURT Phi transfer (0.551 → 0.360 for U1+2 → U3). So:
   a promising ingredient for the day a fingertip is swapped, not a
   substitute for collecting a small per-unit calibration set — which
   remains the honest requirement, exactly as contact_fix_report.md's (a2)
   already concluded.

## Noise-source audit: real floors vs. remaining software levers

| Noise source | Status | Remaining software lever? |
|---|---|---|
| Index crosstalk-fit residual at 0/100% extremes | Best remaining single source of false contacts (2 episodes/session, Thumb full-range moves) | #4's ridge backend is the last untried modelling lever; beyond it, this is data-limited (extreme-position repeats) |
| Joint backlash position uncertainty | Deadband tried and correctly reverted (accuracy cost > noise win) | None credible in software; hardware position feedback or approach-direction-aware calibration data |
| Self-motion transit noise | Transit suppression deployed and verified | Scale the ramp duration by the Speed dropdown value (currently fixed at default speed) — small, cheap correctness improvement |
| Self-closure steady-state false contact | Fixed (contact head retrain), re-verified 0.0% across all fingers/phases | — (re-derive per fingertip if hardware is ever swapped) |
| Session baseline drift | low_pass tracking + contamination gating deployed | Surface a "baseline age" staleness indicator; otherwise addressed |
| Model noise amplification (0.52 N per 2.5-unit sensor noise) | Now quantified (05) | #6 EMA (bought with lag) and #7 GBM force head (−24%) are the levers; below that it's the regressor's intrinsic roughness |
| Sub-guard glitch spikes | Open hole today | Closed by #2 |
| Idle-force positive bias (+0.35 N) | Masked by FORCE_DEADBAND_N | #7 would cut the zero-input offset 3x and allow a smaller deadband |

## Architecture verdict (the question asked first)

The current split — one joint Force/Theta MLP regressor, separate Phi and
Contact MLP classifiers, shared per-unit-median scaler, per-finger crosstalk
backends — is fundamentally sound and close to locally optimal for the data
that exists. The full multi-task/shared-network alternative was NOT tested
here (it would re-open the scaler coupling this project already paid for
once), and the tested tree-based alternative wins only on Force robustness
(#7), not across the board. The real accuracy ceiling is data coverage
(back region, ±45 Phi, repeats), not model class. The highest-leverage
changes were all at the DECISION layer (how raw model outputs become
displayed state): thresholds, gating, guards — items 1–5.

## Deployment

`DEPLOYMENT_INSTRUCTION.md` in this folder is the ready-to-paste Claude
Code instruction implementing items 1–5 end-to-end (file-by-file, with
per-step verification and the live A/B gate for the crosstalk switch).
