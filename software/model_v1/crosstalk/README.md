# Cross-finger magnetic crosstalk

Standalone, additive data-collection + modelling pipeline. Does not
modify, import from, or get imported by any existing working file
(`unified_hand_demo.py`, `hand_sensing_gui.py`, `live_predictor_test.py`,
`live_phiclass_predictor.py`, `hand_controller.py`, `eflesh_controller.py`,
any existing model `.pkl`, or any `master_characterisation*.csv`). It only
*reads* `AmazingHand/hand_controller.py` and
`MagneticSensors/eflesh_controller.py` as libraries. **Nothing in this
folder is wired into the live sensing pipeline** -- see "Integration" at
the bottom.

Pipeline, in order:

1. `record_crosstalk.py` -- collects `raw/*.json` (see "Protocol" below).
2. `build_crosstalk_dataset.py` -- flattens `raw/*.json` into `crosstalk_dataset.csv`.
3. `train_crosstalk_model.py` -- fits `crosstalk_model_v1.json` from the CSV.
4. `evaluate_crosstalk_model.py` -- validates the model, writes `crosstalk_model_report.md`.
5. `record_crosstalk_combined.py` -- collects `raw_combined/*.json`, a small
   combined-movement validation set (see "Additivity validation" below).
6. `build_combined_dataset.py` -- flattens `raw_combined/*.json` into `combined_dataset.csv`.
7. `validate_additivity.py` -- tests the additive assumption against real
   simultaneous-movement data, appends its finding to `crosstalk_model_report.md`.

Each script re-reads the previous stage's output file rather than calling
into the previous script directly (same convention as model_v1's own
`build_delta_dataset.py` -> `build_split.py` -> `train_model_*.py`
pipeline), so any stage can be rerun independently. Steps 5-7 import
`record_crosstalk.py`'s connection/servo/capture code directly (not a
copy) rather than duplicating it.

## What this data represents

Each finger's magnetometers can pick up a faint signal from the OTHER
fingers' magnets simply because of how close they are, purely from hand
geometry -- not from any real contact/touch. This is "crosstalk": a
baseline shift on Finger X's sensors caused by Finger Y's servo position,
with nothing touching Finger X at all.

This folder's recordings isolate that effect: one finger moves through 5
positions while the other 3 stay fixed and idle (no contact), so any
change seen on the 3 stationary fingers' sensors at a given position can
only be geometric crosstalk from the one finger that moved.

## Protocol

For each of the 4 fingers (Index, Middle, Ring, Thumb), one at a time:

1. Move that finger through 5 positions: 0%, 25%, 50%, 75%, 100%
   (0% = fully open, 100% = fully closed, 50% = neutral/`middle_pos` --
   same 0-100 scale `model_v1/hand_control_gui.py`'s position readout
   uses).
2. The other 3 fingers stay fixed at the reference position (0%, fully
   open) throughout.
3. At each position, ~4s of raw magnetic frames (all 4 fingers) are
   captured after a servo-settle pause, with no contact happening.

4 fingers x 5 positions = **20 recordings**.

Finger <-> mux group mapping (confirmed in `hand_control_gui.py`):
Index = Finger 1 (group index 0), Middle = Finger 2 (index 1),
Ring = Finger 3 (index 2), Thumb = Finger 4 (index 3).

## Running it

```
python model_v1/crosstalk/record_crosstalk.py [--port COM7] [--baud 2000000]
                                                [--record-s 4.0] [--settle-s 1.5]
```

Run with the project's own venv (same one every other `model_v1` script
uses), from anywhere in the repo -- `AmazingHand`/`MagneticSensors` are
editable-installed, no `sys.path` hacking needed. Prints a live summary
after each recording (frame count + the other 3 fingers' sensor magnitude
range) so you can sanity-check as you go. Ctrl+C aborts safely -- fingers
are always returned to the 0% (open) reference position before exit.

## Output format (`raw/*.json`)

One file per (moving finger, position) pair:

```
raw/crosstalk_movingfinger{N}_pos{P}.json
```

- `N` -- moving finger's 1-indexed Finger/mux-group number (1=Index,
  2=Middle, 3=Ring, 4=Thumb)
- `P` -- position, zero-padded to 3 digits (`000`, `025`, `050`, `075`, `100`)

Each file:

```jsonc
{
  "recording_info": {
    "tool": "record_crosstalk.py",
    "purpose": "...",
    "recorded_at_iso": "...",
    "num_frames": 123,
    "requested_duration_s": 4.0,
    "settle_s": 1.5,
    "servo_speed": 3,
    "moving_finger": {"label": "Index", "group_index": 0, "position_percent": 25},
    "other_fingers": [
      {"label": "Middle", "group_index": 1, "position_percent": 0},
      {"label": "Ring",   "group_index": 2, "position_percent": 0},
      {"label": "Thumb",  "group_index": 3, "position_percent": 0}
    ]
  },
  "frames": [
    {
      "seq": 4821,
      "timestamp": 1752400000.123,
      "groups": [
        {"group_index": 0, "sensors": [{"x": .., "y": .., "z": .., "magnitude": ..}, ...]},
        {"group_index": 1, "sensors": [...]},
        {"group_index": 2, "sensors": [...]},
        {"group_index": 3, "sensors": [...]}
      ]
    },
    ...
  ]
}
```

`frames[].groups[]` always includes all 4 fingers (not just the moving
one), since crosstalk is measured on the OTHER 3. Values are raw
(no baseline subtraction) -- deriving deltas/crosstalk magnitude from
these is left to whatever builds the actual crosstalk model later.

## Combined-movement validation set (`raw_combined/*.json`)

A smaller, separate 15-recording set (2 batches), added specifically to
test the additive-model assumption below (single-finger sweeps alone
can't -- see "Additive assumption"). Two source fingers move to the SAME
position SIMULTANEOUSLY instead of one:

**Batch 1** -- `record_crosstalk_combined.py`, 6 recordings:
- Index + Middle together, at 25%/50%/75% -- Ring and Thumb held fixed at 0%.
- Middle + Ring together, at 25%/50%/75% -- Index and Thumb held fixed at 0%.

**Batch 2** -- `record_crosstalk_combined_round2.py`, 9 recordings (imports
and reuses batch 1's `save_recording()`/`print_summary()`/`FINGER_BY_LABEL`
directly, rather than duplicating them):
- Middle + Thumb together, at 25%/50%/75% -- Index and Ring held fixed at
  0% (closes out Index's own flagged CAUTION pair, see below).
- Index + Ring together, at 25%/50%/75% -- Middle and Thumb held fixed at
  0% (closes out Middle's own flagged CAUTION pair).
- Index + Thumb together, at 10%/20%/30% ONLY -- Middle and Ring held
  fixed at 0%. **Safety-capped**: Index and Thumb physically collide if
  both are closed past ~30% simultaneously (confirmed hardware
  constraint). The cap is enforced in code (`SAFETY_CAPS` in
  `record_crosstalk_combined_round2.py`), checked both up front (before
  any hardware is touched) and immediately before every individual move,
  not just by the choice of positions in that script's `TESTS` list.

Run either with the same `--port`/`--baud`/`--record-s`/`--settle-s`/
`--speed` flags as `record_crosstalk.py` (which both import and reuse
directly, not duplicate). Both batches save into the SAME `raw_combined/`
directory with the same filename convention --
`crosstalk_combined_{sourceA}-{sourceB}_pos{P}.json`, e.g.
`crosstalk_combined_middle-thumb_pos025.json` -- so
`build_combined_dataset.py` (unmodified) picks up all 15 recordings
automatically. Same JSON shape as `raw/*.json`'s `frames[]`, except
`recording_info` has a `"moving_fingers"` list (2 entries) instead of a
single `"moving_finger"`.

This set has no position=0 recording of its own -- `build_combined_dataset.py`
uses the pooled average of the 4 original `raw/*_pos000.json` recordings
as its reference baseline instead (see that script's docstring for why,
and the limitation that implies).

## Crosstalk model (v1)

### Dataset (`build_crosstalk_dataset.py` -> `crosstalk_dataset.csv`)

For each of the 20 recordings and each of the 3 stationary ("affected")
fingers in it, computes the mean and within-recording standard deviation
of each of the 15 raw `S1X..S5Z` channels (same naming convention as
`model_v1`'s own `raw_cols`), then a `delta_from_p0`: that same value
minus the SAME sweep's own position=0 recording (all 4 fingers at the
reference/open position). Using the within-sweep P=0 as the reference
(rather than pooling across all four P=0 recordings) cancels out drift
over the ~2 minutes the full run took, instead of leaking it into the
delta. 900 rows total (20 recordings x 3 affected fingers x 15 channels).

### Model (`train_crosstalk_model.py` -> `crosstalk_model_v1.json`)

For every (affected finger, source finger, channel) triple -- 12 finger
pairs x 15 channels = 180 tiny models -- fits `delta = c1*P + c2*P^2` with
**no intercept term**, i.e. forced through the origin: at the source's own
reference position (P=0), predicted crosstalk shift is exactly zero by
construction, not by data (every target value already IS a delta from
that P=0 point).

Degree (1 = linear, or 2 = quadratic) is chosen **per channel**, not
globally, via leave-one-out cross-validation over the 4 non-zero measured
positions {25, 50, 75, 100}: whichever degree has the lower LOOCV mean
absolute error wins, ties favouring degree 1. This exists because several
channels showed clearly non-monotonic behaviour across the sweep (e.g.
rising then plateauing/dipping rather than a straight ramp) -- a
degree-2-only or degree-1-only rule would have under- or over-fit
depending on the channel.

### Additive assumption -- read this before trusting a prediction

**The model's core assumption is that when 2+ other fingers move
simultaneously, their crosstalk contributions on a given finger simply
ADD:**

```
predicted_shift(finger F, channel) = sum over source fingers S != F of
                                      f_{F,S,channel}(position of S)
```

This is an assumption that the ORIGINAL 20-recording dataset **cannot
verify**: every one of those recordings moved exactly one finger while
the other three sat at the reference position. There is no recording of,
say, Index and Thumb moving together within that set, so nothing there
confirms their two individual effects actually sum linearly rather than
interacting (saturating, partially cancelling, etc.) when combined.

`evaluate_crosstalk_model.py` screens where this matters most: for each
(affected, source) pair it compares the pair's peak effect size (largest
`|delta|` across all 15 channels and 4 non-zero positions) against that
same channel's pure sensor noise floor (its within-recording standard
deviation at P=0) as a signal-to-noise ratio (SNR), and flags the pair
"significant" if SNR >= 3.0. **This is a simple screening heuristic, not
a formal significance test.**

| Affected | Significant sources | Originally flagged | Now validated? |
|---|---|---|---|
| Index  | Middle, Thumb | CAUTION -- 2 significant sources | **Yes -- held within noise** |
| Middle | Index, Ring   | CAUTION -- 2 significant sources | **Yes -- held within noise** |
| Ring   | Index, Middle | CAUTION -- 2 significant sources | **Yes -- held within noise** |
| Thumb  | Index (only)  | OK -- nothing else non-negligible to add | n/a (never flagged) |

Crosstalk strength tracks physical finger adjacency, which is a useful
sanity check: Index<->Middle and Middle<->Ring (physically neighbouring
fingers) show the strongest effects (SNR in the high 20s-high 30s on
their peak channel), while more distant pairs (e.g. Ring<->Thumb) are
mostly within the noise floor. See "Additivity validation" immediately
below for how each CAUTION flag was subsequently tested against real
simultaneous-movement data.

#### Additivity validation (2026-07, `raw_combined/` + `validate_additivity.py`)

Two batches of combined-movement recordings (15 total, see
"Combined-movement validation set" above) were collected specifically to
test this. `validate_additivity.py` compares `predict_shift()`'s additive
prediction against what was actually measured, and cross-references each
tested combination against the CAUTION table above (full detail,
including per-channel numbers, in `crosstalk_model_report.md`'s
"Additivity validation" section, regenerated by rerunning that script
after `build_combined_dataset.py`).

**All 3 originally-flagged CAUTION pairs now have a direct, exact-match
test, and all 3 held within noise:**

| Affected | Its flagged CAUTION pair | Tested via | Result |
|---|---|---|---|
| Index  | Middle, Thumb | Middle+Thumb moved together | **held** (ratio 0.7) |
| Middle | Index, Ring   | Index+Ring moved together   | **held** (ratio 0.6) |
| Ring   | Index, Middle | Index+Middle moved together | **held** (ratio 0.8) |

The 4 partial-match combinations (one flagged source substituted for a
non-flagged one, e.g. Index affected by Middle+Ring) **all held within
noise too** -- 7/7 tests on Index, Middle, and Ring held, with no
exceptions.

**Only Thumb's 3 tests broke down** (Index+Middle, Index+Ring, and
Middle+Ring, all measured on Thumb) -- but Thumb was never flagged
CAUTION to begin with (only Index was individually significant for it),
and the absolute effect sizes are tiny (mean |actual shift| ~2-4 raw
units, comparable to the noise floor already, vs. 20-37 raw units for the
Index/Middle/Ring tests above). This reads as noisy prediction of an
already-small effect, not a large physical interaction, and doesn't
indicate the additive model is unreliable where it actually matters.

**Extra check -- Index+Thumb (10/20/30% only, safety-capped):** measured
on both Middle and Ring. Middle showed a noticeably larger response
(mean |actual| ~22 raw units, comparable in scale to the flagged-pair
effects above) than Ring (~2.7 raw units, near the noise floor) --
plausible given Middle sits physically between Index and Ring, closer to
Index. Both **held within noise** regardless (ratios 1.4 and 1.3).

**Net conclusion: additivity holds well across every case that matters.**
All 3 originally-flagged CAUTION pairs, tested directly and exactly, held
within the individual models' own fit noise -- no evidence of the
saturating/cancelling interaction effects the CAUTION flag was warning
about. The only "breakdown" cases are on Thumb, which was never flagged
in the first place and where the underlying effect is barely above the
noise floor regardless of whether 1 or 2 fingers are moving. This is
still a small dataset (15 combined recordings, 3 positions each) and not
a substitute for validating specific arbitrary position combinations
before depending on this for precise correction -- but there is no longer
an untested flagged-risk pair standing between this model and a
first-pass live integration.

### Fit quality (`evaluate_crosstalk_model.py` -> `crosstalk_model_report.md`)

Two goodness-of-fit numbers per (pair, channel), averaged into the
report's per-pair table:

- **In-sample R²**, over all 5 measured points per channel (0, 25, 50,
  75, 100) -- the standard "how much variance does the fit explain"
  number, but note it's fit on the same points it's scored against.
- **LOOCV MAE**, a genuine held-out check: leaves out one of the 4
  non-zero positions, refits on the remaining 3 (+ the structural origin
  point), predicts the held-out one, and averages the absolute error over
  all 4 folds. This is the honest "how well does it predict a position it
  didn't see" number the fit-quality request asked for -- but with only 4
  points per channel, treat it as indicative, not a tight bound.

An end-to-end sanity check (re-running every one of the 900 dataset rows
through the actual `crosstalk_model.predict_shift()` function a future
integration would call) matched the per-channel fits essentially exactly,
confirming the model is wired correctly through its own prediction API --
this checks plumbing, not the additive-combination assumption above.

### Position sensitivity (2026-07, `analyze_position_sensitivity.py`)

The finger-to-palm joints have mechanical backlash/play -- a commanded
position (e.g. 100%) doesn't always correspond to the exact same true
physical position across repetitions. The correction relies on
**commanded** position as a proxy for true position
(`sensing_gui.py`'s `set_finger_position()`), so if the fitted polynomials
are highly sensitive to small position errors, backlash-scale uncertainty
could itself be an independent source of residual correction error -- a
mechanical fix isn't practical right now, so this was investigated as a
question of whether a software mitigation is warranted.

`analyze_position_sensitivity.py` perturbs each fitted polynomial's
position input by a plausible backlash magnitude and compares the
resulting correction swing against that channel's own sensor noise floor.
Result (full tables in `position_sensitivity_report.md`):

| Backlash | Significant pairs flagged (swing >= noise floor x 3) |
|---|---|
| +/-2% | 0/7 -- safe |
| +/-3% | 1/7 -- Middle<-Index (3.8x) |
| +/-5% | 4/7 -- Index<-Middle (4.5x), Middle<-Index (6.2x), Middle<-Ring (3.6x), Ring<-Middle (4.4x) |
| +/-8% | same 4/7, up to 9.7x |

**Confirmed as a real, meaningful contributor** at plausible backlash
magnitudes (all 4 flagged pairs involve Middle, either as source or
affected finger -- consistent with Middle already being the dominant
crosstalk source everywhere in this model). The sensitivity is worst at
the resting *extremes* (0%/100%, precisely where fingers spend most real
time) and best in the 50-70% middle of the range, where the fitted
quadratics are flattest.

**Mitigation implemented**: `sensing_gui.py`'s `HandSensingEngine.poll()`
now rounds each finger's position estimate to the nearest
`CROSSTALK_POSITION_DEADBAND_PCT` (5%) immediately before it's used by
the crosstalk correction -- applied only at the point of use, not to the
position-ramp estimate itself (the earlier transit-timing fix, see
`CROSSTALK_POSITION_RAMP_S_PER_100PCT`'s comment in `sensing_gui.py`, is
untouched and still tracks a smoothly-interpolating estimate during
transit). Positions within the same +/-2.5%-wide bucket now produce an
identical correction.
This doesn't fix bias from backlash (the true position can still land in
the "wrong" bucket near a boundary) but eliminates the correction-value
*jitter* small continuous position noise would otherwise inject -- the
more directly relevant risk for spurious Contact-threshold crossings.

Live validation (same realistic-session diagnostic as the transit-timing
and stale-baseline fixes): Index's confirmed false-contact episodes,
already reduced from 5 to 2 by the transit-timing fix alone, dropped to
**0** with the deadband added. One new isolated episode appeared on
Middle in that same run (not present in the prior post-fix session) --
plausibly ordinary run-to-run variability rather than a regression (Middle
remained one of the two cleanest fingers across every other session), but
noted rather than dismissed; Ring stayed completely clean across every
session tested.

### Deadband widening investigated and rejected (2026-07-14, follow-up)

Index still showed occasional false-contact noise after the 5% deadband
above, so a natural next question was whether widening
`CROSSTALK_POSITION_DEADBAND_PCT` further (e.g. to 10%), globally or as a
per-pair override just for Index's own sources (Middle, Thumb), would
help further. This was checked analytically first, per instruction not
to keep tuning the deadband blindly.

`analyze_position_sensitivity.py` was extended with
`worst_case_deadband_error()`/`deadband_sweep_for_pair()`, which model the
deadband/quantization step itself rather than a raw perturbation: for a
commanded position, round to each candidate grain, then find the worst
gap between the resulting correction and the true correction needed for
any physically-plausible true position within the backlash range. Swept
`{0%, 5%, 10%}` against Index's two actual crosstalk sources
(Index<-Middle, Index<-Thumb) across all previously-used backlash
magnitudes (full detail in `position_sensitivity_report.md`).

**Result: widening makes it worse, not better.** At every backlash
magnitude tested, for both pairs, worst-case error increases monotonically
with deadband width (e.g. Index<-Middle @ +/-5% backlash: 14.88 with no
deadband, 20.50 at 5%, 28.57 at 10%). Mechanism: a wider bucket reduces
how often the estimate crosses a boundary, but each bucket's center sits
further from the commanded position on average, and that rounding error
stacks on top of (not instead of) the backlash-driven uncertainty the
deadband can't fix in the first place. So no widening was implemented --
`CROSSTALK_POSITION_DEADBAND_PCT` stays at 5%, and no per-pair override
was added, since the same monotonic-worsening pattern held for both of
Index's sources with no exception to build an override around.

This also explains why the deadband still empirically helped earlier
(2->0 episodes) despite this: that benefit was real but was against
tick-to-tick *jitter in the position estimate itself* (small timing-driven
fluctuations from the ramp interpolation, not backlash), which a fixed
grain does genuinely cancel -- a different mechanism from the
backlash-vs-bucket-boundary trade-off modelled here, and one that doesn't
improve by widening the grain further once the jitter itself is already
smaller than one bucket.

A fresh live confirmation run (same realistic-session diagnostic,
unmodified code) put a number on the actual remaining state: Index showed
2 confirmed-contact episodes, both triggered specifically by **Thumb's own
individual full-range (0->100%) move** -- not the smaller 15%
simultaneous all-4 move -- consistent with the earlier finding that
crosstalk sensitivity is worst at the 0%/100% extremes. Ring stayed
completely clean; Middle showed 1 moderate episode (1.05N, triggered by
Index's own full-range move), matching the "possibly ordinary variability"
Middle episode already noted in the section above rather than indicating
a new problem. All 24 recalibration commits in this run fired only during
genuine `confirmed_contact=False` windows (5 of them via the stuck-recovery
path), confirming the transit-ramp and stale-baseline-cascade fixes are
both still intact with no regression.

**Conclusion**: Index's remaining false-contact noise is dominated by the
underlying crosstalk fit's own residual error at extreme (near-0%/100%)
positions -- consistent with Index<-Middle already being flagged as the
noisiest major-pair fit earlier in this document -- not by position
quantization/rounding. Further deadband tuning should not be expected to
resolve it; the next lever, if this is worth pursuing further, would be
improving the underlying polynomial fit itself (more/repeated calibration
data at the extremes) rather than any position-estimate tolerance.

### Deadband reverted (2026-07-15)

Live use after the sections above landed showed Index's *real* Force
readings going noticeably weak/underestimated on firm presses -- a
regression in the crosstalk correction's accuracy, not its noise. Given
the "Deadband widening investigated and rejected" numbers already in this
file (5% deadband roughly doubling Index<-Middle's worst-case and mean
correction error over no deadband, at every backlash magnitude tested),
the 5% deadband itself was the leading suspect rather than the transit-
timing ramp (which had no such documented accuracy cost) or the base
correction/subtraction (which predates both fixes and was accurate before
them). `sensing_gui.py`'s `CROSSTALK_POSITION_DEADBAND_PCT`/
`_quantize_position()` were removed -- `HandSensingEngine.poll()` now
feeds the crosstalk correction the raw, un-quantized ramp-interpolated
position again, same as before this constant existed. The transit-timing
ramp (`CROSSTALK_POSITION_RAMP_S_PER_100PCT`) and the crosstalk correction
itself are untouched. Expected/accepted trade-off: Index's false-contact
noise this deadband suppressed is expected to return to its pre-deadband
level.

### Known limitations

- **Only 20 recordings, 5 positions each, no repeats.** Every fit quality
  number here (R², LOOCV MAE) is based on that same handful of points --
  small enough that noisy individual channels can swing those numbers a
  lot. Treat per-channel numbers as indicative, not precise.
- **All 3 originally-flagged CAUTION pairs now have a direct, exact-match
  combined-movement test (15 recordings total, 2 batches), and all 3 held
  within noise** -- see the "Additivity validation" subsection above. This
  is real evidence the additive assumption is reasonable, not just an
  untested assumption anymore, for the pairs that were flagged risky. It
  is still only 3 positions (25/50/75%, or 10/20/30% for the safety-capped
  Index+Thumb check) per combination, from a single session -- not
  exhaustive coverage of the full position space, and not repeated across
  sessions to check consistency.
- **The combined-validation set has no reference recording of its own**
  -- `build_combined_dataset.py` uses the ORIGINAL sweep session's pooled
  P=0 recordings as its baseline, so any drift between that session and
  the combined-validation sessions (recorded separately, potentially
  hours/days apart, across 2 batches) is not cancelled out the way
  `delta_from_p0` cancels it within a single sweep. A small, unquantified
  extra source of error in `combined_dataset.csv`'s `delta_from_reference`
  values.
- **No temperature/long-term drift characterisation** -- all 20 original
  recordings were captured back-to-back in one session (same for each of
  the 2 combined-validation batches, each in its own separate session);
  whether crosstalk magnitude itself drifts over a longer session or
  across power cycles is unknown.
- **Only 2 fingers were ever moved together at once.** No recording here
  tests 3 or 4 fingers moving simultaneously -- the additive model
  extrapolates to that case (`predict_shift()` sums however many source
  fingers are passed in), but it has only been validated for exactly 2
  simultaneous movers.
- The reference position (0%, fully open) is the ONLY "stationary" state
  tested for the affected fingers -- crosstalk FROM a moving finger was
  only ever measured with the affected finger sitting at 0%. Whether the
  affected finger's own position changes how much crosstalk it picks up
  (rather than just where its baseline sits) is untested.

### Integration

Not done, and deliberately out of scope for this pass. `crosstalk_model.py`
exposes `load_model()` / `predict_shift()` as a small, self-contained
library that a future integration (e.g. inside `sensing_gui.py`'s
baseline/recalibration logic) could call, but nothing currently imports
it outside this folder's own `evaluate_crosstalk_model.py` and
`validate_additivity.py`.
