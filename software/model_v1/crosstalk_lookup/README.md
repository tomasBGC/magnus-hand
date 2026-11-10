# Crosstalk lookup -- direct measurement + 3-way approach comparison

Standalone, additive pipeline. Does not modify anything in `model_v1/crosstalk/`
or the live sensing pipeline (`sensing_gui.py`, `hand_control_gui.py`) --
only reads `model_v1/crosstalk/crosstalk_common.py` and `crosstalk_model.py`
as libraries (unmodified), and reuses `model_v1/crosstalk/record_crosstalk.py`'s
connection/servo/capture code the same way `record_crosstalk_combined.py`
already does. See `crosstalk_lookup_common.py`'s module docstring for why
this lives in its own folder rather than extending `model_v1/crosstalk/`.

## Why this exists

`model_v1/crosstalk/`'s existing model is fitted from data where exactly
ONE finger ever moved at a time, and its "shift when 2+ fingers move
together" prediction is an ADDITIVE ASSUMPTION -- validated only for 2
simultaneous movers (see that folder's README.md), never all 4. This
pipeline collects real, directly-measured ground truth across genuinely
random 4-finger combinations (all 4 moving simultaneously) and compares
three ways of using it:

- **(a)** the existing pairwise-additive model, unmodified, as a baseline
- **(b)** nearest-neighbour lookup against the measured data -- no fitting
- **(c)** a small trained regression model over the measured data

## Pipeline

```
sample_combinations.py       (no hardware) -- LHS + safety filter -> combinations_manifest.json
        |
        v
record_crosstalk_lookup.py   (hardware)    -- moves all 4 fingers per combo -> raw/*.json
        |
        v
build_lookup_dataset.py      (no hardware) -- raw/*.json -> crosstalk_lookup_dataset.csv
        |
        v
evaluate_approaches.py       (no hardware) -- 80/20 split, trains (b)/(c), scores a/b/c -> comparison_report.md
```

Run in that order:

```bash
uv run model_v1/crosstalk_lookup/sample_combinations.py
uv run model_v1/crosstalk_lookup/record_crosstalk_lookup.py [--port COM7] [--resume]
uv run model_v1/crosstalk_lookup/build_lookup_dataset.py
uv run model_v1/crosstalk_lookup/evaluate_approaches.py
```

## Safety (Index+Thumb collision) -- read `record_crosstalk_lookup.py`'s
module docstring in full before rerunning this

`crosstalk_lookup_common.SAFETY_CAPS` enforces a **20%** simultaneous-
closure cap for Index+Thumb -- tighter than `record_crosstalk_combined_round2.py`'s
existing 30% cap, because this session's broad random/LHS coverage visits
far more simultaneous-nonzero Index+Thumb combinations than that script's
narrow, deliberately-away-from-the-corner validation batches ever did.
Enforced in code, not just by construction:

1. `sample_combinations.py` filters every candidate combination against
   the cap before it's ever written to the manifest (168 of 268 sampled
   combinations were rejected in the run that produced the 100 used here
   -- see `combinations_manifest.json`'s own `rejected_log`).
2. `record_crosstalk_lookup.py` re-checks every manifest entry again,
   up front, before any hardware is touched.
3. It re-checks each combination a third time, immediately before sending
   any servo command for it.
4. Between every combination it returns ALL 4 fingers to the 0% reference
   position before commanding the next one -- this "safe waypoint"
   protocol is what makes the cap hold structurally THROUGHOUT transit,
   not just at the two move endpoints (there is no servo position
   feedback on this hardware to monitor continuously, so the guarantee
   has to come from the motion protocol itself -- see the recorder's
   module docstring for the full argument).

Any violation at any of these checks raises `SystemExit` immediately, with
no servo command sent for the offending move.

## Output data

- `combinations_manifest.json` -- the 100 sampled+filtered combinations,
  plus every rejected candidate and why.
- `raw/crosstalk_lookup_reference_idx000_mid000_ring000_thumb000.json` --
  one recording, all 4 fingers at 0%, the reference every combination's
  shift is measured against.
- `raw/crosstalk_lookup_combo{N:03d}_idx{I}_mid{M}_ring{R}_thumb{T}.json`
  -- one recording per combination, all 4 fingers' raw frames.
- `raw/session_log.json` -- per-combination capture log (frame counts,
  settle waits) from the recording session.
- `crosstalk_lookup_dataset.csv` -- one row per (combination, finger,
  channel): mean/std raw value and `delta_from_reference`.
- `comparison_report.md` -- the 3-way MAE comparison (auto-generated).

## Not integrated

Nothing here is wired into `sensing_gui.py` or any other live-pipeline
file. This pass only builds and validates all three approaches; see the
comparison report for the results and pick an integration approach as a
separate, deliberate step.
