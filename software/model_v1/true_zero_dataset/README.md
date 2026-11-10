# True-zero dataset (2026-07-17)

Standalone, additive pipeline. Does not modify `model_v1/crosstalk/`,
`model_v1/crosstalk_lookup/`, or the live sensing pipeline -- only reads
`model_v1/crosstalk/crosstalk_common.py` as a library (unmodified) and
imports the CURRENTLY-LIVE `model_v1/sensing_gui.py` to drive its exact
live per-finger processing during data collection (also unmodified).

## Why this exists

Everything up to this point (Schmitt trigger, spike guard, crosstalk
backend choice, auto-rebaseline-on-return-to-open) improved the DECISION
layer around the existing Contact classifier's output. This pipeline
instead collects new, real, measured no-touch ground truth across 500
genuinely simultaneous 4-finger position combinations -- a direct,
data-driven continuation of the same class of fix as
`model_v1/contact_classifier_fix/` (self-closure) and
`model_v1/crosstalk_lookup/` (measured crosstalk), but at far broader
coverage (500 vs. 100 combinations, and measuring the model's actual
DISPLAYED output rather than raw sensor deltas).

## Pipeline

```
sample_combinations.py         (no hardware)  -> combinations_manifest.json
        |
        v
record_true_zero_dataset.py    (hardware)     -> raw/*.json + raw/session_log.json
        |
        v
build_true_zero_dataset.py     (no hardware)  -> true_zero_summary.csv + true_zero_dataset.csv
        |
        v
analyze_true_zero_results.py   (no hardware)  -> true_zero_report.md
        |
        v
retrain_contact_from_true_zero.py (no hardware) -> active_model_true_zero_fix.pkl + retrain_report.md
        |
        v
build_candidate_v3.py           (no hardware)  -> candidate_v3/ (full self-contained candidate)
```

## Safety (Index+Thumb collision)

`true_zero_common.SAFETY_CAPS` enforces a **15%** simultaneous-closure cap
for Index+Thumb -- tighter than `crosstalk_lookup`'s 20% and
`record_crosstalk_combined_round2.py`'s 30% (each is that script's own
considered choice; this module deliberately does not import either, to
avoid inheriting a looser cap). Enforced at every check point, matching
this project's established pattern:

1. `sample_combinations.py` filters every candidate combination before it's
   ever written to the manifest.
2. `record_true_zero_dataset.py` re-checks every manifest entry again, up
   front, before any hardware is touched.
3. It re-checks each combination a third time, immediately before sending
   any servo command for it.
4. Between every combination it returns ALL 4 fingers to the 0% reference
   position before commanding the next one (the same "safe waypoint"
   protocol `crosstalk_lookup` uses, which makes the cap hold structurally
   through transit, not just at the two move endpoints).

Any violation raises `SystemExit` immediately, with no servo command sent.

**Deliberately includes 15 "boundary" combinations** where Index=Thumb=15%
exactly (the closest-without-touching case), Middle/Ring LHS-varied,
interleaved through the session -- the most informative case for the model
to learn "this is still zero-force" from.

## What's measured

Unlike `crosstalk_lookup` (which records raw sensor deltas for offline
fitting), this pipeline drives the LIVE model's actual per-finger
processing (`sensing_gui._process_one_finger`, imported unmodified)
continuously across the whole 500-combination session -- baseline
tracking, periodic recalibration, and the auto-rebaseline-on-return-to-open
fix all carry state across combinations exactly as they would in a real
continuous-use session. At each combination, after settling, it captures a
few seconds of the model's DISPLAYED Force, raw Force, and confirmed
Contact per finger, plus (for training purposes) the exact 20-dim input
feature vector that produced them.

## Results (2026-07-17 run)

All 500 combinations collected (across 9 collection runs -- the recording
process was killed unexpectedly by something outside this pipeline's
control 6 times during the session; each time the hand was immediately
confirmed returned to a safe fully-open state before resuming with
`--resume`, and zero data was lost since each combo's file is written
immediately upon capture, not batched in memory).

**Headline**: 523/2000 (26.2%) (combination, finger) pairs showed nonzero
displayed Force despite genuinely no touch anywhere. See `true_zero_report.md`
for the full breakdown. Per-finger affected rate: Ring 52.8%, Middle 31.8%,
Index 20.0%, **Thumb 0.0%**. Worst Middle cases correlate strongly with Ring
sitting at a high (closed) position -- consistent with this project's own
prior finding that Middle<->Ring is one of the strongest crosstalk pairs.
Worst Index cases correlate with Index's OWN position being near fully-open
-- consistent with the joint-backlash finding from the live candidate_v2
testing session earlier the same day.

**Retrain result** (`retrain_report.md`, Contact classifier only, jig +
self-closure + 400/500 true-zero combos, 100 held out):

| Check | Before | After |
|---|---|---|
| Held-out jig test (genuine touch, regression check) | 77.71% | 78.30% (+0.60pp, no regression) |
| Held-out true-zero combos (should read 0) -- accuracy | 79.86% | 99.80% |
| Held-out true-zero combos -- false-positive rate | 20.14% | **0.20%** (~100x reduction) |
| Per-finger FP: Index | 17.62% | 0.03% |
| Per-finger FP: Middle | 23.50% | 0.02% |
| Per-finger FP: Ring | 39.18% | 0.38% |
| Per-finger FP: Thumb | 0.27% | 0.37% (negligible either way) |

Packaged into `candidate_v3/` (full self-contained copy of the current live
system, Force regressor/Phi classifier/scaler byte-identical, Contact
classifier retrained).

**Update (2026-07-30): promoted, and now live.** This Contact classifier
was promoted to `active_model.pkl` on 2026-07-17. It was briefly reverted
to the pre-true-zero (selfclosure-fix) classifier earlier on 2026-07-20
after live A/B testing showed reduced light-touch recall, then restored on
2026-07-20T18:35:15Z after weighing that tradeoff against the ~100x
multi-finger false-positive reduction shown above; see
`v2_revert_verification_report.md` and
`true_zero_restore_verification_report.md` for that decision. It stayed
the deployed Contact classifier until **2026-09-07**, when the Contact head
was retrained again on top of this dataset (all 500 combinations) plus the
load-cell grasp dataset (`grasping_dashboard/retrain_contact_with_load_cell_data.py`).
The classifier described in this README is kept as `models/after_true_zero/`.
See `model_v1/README.md`'s "How
`active_model.pkl` was built" section for the full up-to-date lineage.

## Known limitations

- Single session -- no repeat trials to check consistency over time
  (thermal drift, etc.), same caveat every single-session dataset in this
  project already carries.
- The "assume target position throughout transit" simplification (see
  `record_true_zero_dataset.py`'s own docstring) means crosstalk-correction
  accuracy DURING transit is approximate; this only affects diagnostics
  during transit, not any captured data point (Contact evaluation is frozen
  during a finger's own transit regardless, via
  `CONTACT_SUPPRESS_DURING_OWN_TRANSIT`).
- Retraining held out combinations, not individual ticks -- correct for
  avoiding leakage, but with 500 combinations an 80/20 split leaves ~100
  combos (400 finger-readings) for held-out evaluation; treat as
  indicative at the current scale, not a tight bound.
