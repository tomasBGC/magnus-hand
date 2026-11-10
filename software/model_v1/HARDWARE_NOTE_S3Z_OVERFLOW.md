# Hardware/firmware note: channel overflow/wraparound (S3Z, and now Ring's S4X)

**Status: open, unfixed. For hardware/firmware attention. Independent of
the crosstalk-correction modelling work in `crosstalk_lookup/` — flagged
here separately so it doesn't get lost inside that folder's model-fitting
history.**

**2026-08-30 update**: a SECOND, apparently unrelated instance of this same
overflow signature was found on Ring's own sensor 4 X-axis (`dS4X`) during
the genuine-grasp data collection session (`true_zero_dataset/
record_genuine_grasp_dataset.py`) -- see the new section at the bottom of
this file. Unlike S3Z (only glitched under specific crosstalk load, on the
magnet-free Index board specifically), this one glitched on **100% of hold
ticks, in all 36 recorded reps, on the NORMAL magnet-equipped Ring
fingertip, while Ring sat perfectly still at the reference position** --
i.e. persistent for an entire session, not load-triggered. This raises the
first finding's severity: it's evidently not unique to one swapped-out
board, and may be a broader per-unit or per-channel susceptibility worth
checking across all 4 fingers, not just Index/Ring.

## What was found

During the magnet-free Index disturbance characterisation session
(2026-08-30, see `crosstalk_lookup/magnetfree_index_disturbance/report.md`
section 0 for the full writeup and per-recording data), raw channel **S3Z**
(sensor 3, Z-axis) was found to swing by ~253,700 raw units *within a single
~4-second steady-state capture* — toggling between roughly **+126,000** and
**-126,500** frame-to-frame — in **11 of 22 recordings** this session. No
other channel (S1X..S5Z minus S3Z, 14 channels total) ever showed anything
close to this: the largest genuine crosstalk-driven delta measured anywhere
this session was well under 1,000 raw units.

This is consistent with the raw register **crossing a signed-integer
overflow/wraparound boundary near +131,071** (a 17-bit signed range) as the
ambient field from an adjacent finger's magnet (Middle or Ring specifically
— see trigger conditions below) adds onto S3Z's quiescent value and pushes
it just over that boundary.

## Trigger conditions (from this session's data)

- **Never** appears in the idle baseline (all 4 fingers at 0%).
- **Never** appears when only Thumb moves (Thumb-alone sweep, 0-100%, clean
  throughout).
- **Always** appears once Middle or Ring moves off 0%, and gets **more**
  unstable (closer to an even positive/negative frame split within one
  capture, i.e. more of the ~4s capture spent flickering across the
  boundary) as Middle approaches 100% closure.

## Why this was invisible before this session

This is the first time this codebase has measured Index's own 15 raw
channels with a **magnet-free** fingertip in the Index position. Every
earlier measurement (the entire `model_v1/crosstalk/` and
`model_v1/crosstalk_lookup/` history) used a normal, magnet-equipped Index
fingertip, whose own magnet's much larger local field almost certainly
keeps S3Z's quiescent raw value far away from the overflow boundary at
rest — so ordinary crosstalk-sized perturbations from other fingers never
pushed it close enough to wrap. The magnet-free board's quiescent S3Z value
(~125,504 at rest this session) sits much closer to the boundary, so a
comparatively small additional field is enough to tip it over.

**Open question this note does NOT answer**: whether a normal
magnet-equipped Index fingertip's S3Z channel is *also* subject to this
same overflow under some other condition (e.g. a different pose, a
different physical unit, temperature drift shifting the quiescent value
closer to the boundary over time) — this session only characterised the
magnet-free board, so that has not been tested either way.

## What this means for downstream software

- `crosstalk_lookup/analyze_magnetfree_index_disturbance.py` and
  `crosstalk_lookup/build_magnetfree_index_model.py` both detect this
  automatically (any channel whose within-capture max-min range exceeds
  50,000 raw units is flagged and excluded) and treat S3Z as **unreliable
  data this session** — it is never averaged into any comparison or fitting
  target.
- The new Index crosstalk model
  (`crosstalk_lookup/magnetfree_index_model_v1.json`) always predicts a
  **0 shift** for S3Z rather than fitting a model to what would be noise
  dominated by wraparound artifacts.
- No other part of the live sensing pipeline (`sensing_gui.py`'s Force/Phi/
  Contact inference) was checked for its own sensitivity to S3Z
  specifically as part of this session — that's a separate, not-yet-done
  investigation if this channel turns out to matter for anything beyond
  crosstalk correction.

## Suggested next steps (not attempted this session)

1. Confirm whether the *normal* magnet-equipped Index board's S3Z channel
   ever approaches the same boundary, under any pose/condition, on this
   unit and/or other units.
2. If reproducible, this is likely a firmware/data-format issue (raw
   magnetometer register width, sign handling, or a scaling step upstream
   of what's captured as "raw") rather than a one-off ADC glitch, given how
   consistently it reproduces (11/11 flagged recordings show the identical
   ~253,700-unit range and boundary location, not a smaller/random-looking
   fault).
3. If it can be reproduced on demand, capturing the RAW (pre-any-
   processing) register value at the moment of a flip, rather than only the
   post-processing `x/y/z` reading already used throughout this codebase,
   would help distinguish "signed overflow at a fixed bit width" from other
   candidate causes (e.g. a calibration/offset step overflowing separately
   from the raw sensor read).

---

## Second instance: Ring's S4X, persistent for a whole session (2026-08-30)

Found incidentally while collecting genuine multi-finger grasp data
(`true_zero_dataset/record_genuine_grasp_dataset.py`, 36 reps, real
Index+Thumb / Index+Middle / all-4-finger grasps) for Contact-classifier
retraining. Ring's own group, sensor 4, X-axis (`dS4X` in the standard
15-channel-delta feature naming) was flagged `glitched=True` (the SAME
`SANE_DELTA_MAX=2000` guard as the S3Z finding above) on **160/160 hold-
phase ticks, in every one of 36 recorded reps, across the entire ~15-minute
session** -- not intermittent, not load-triggered: Ring sat at the
reference (0%) position, untouched, the whole time in most of these reps,
yet this one channel was glitched from the very first captured tick to the
last.

**Same signature as S3Z**: a bimodal-magnitude delta, e.g. one rep's
values ranged from -52,456 to +104,827 -- large swings between two distant
magnitudes, not a smooth drift, consistent with the same
overflow/wraparound mechanism (crossing a fixed-bit-width signed boundary)
as the original S3Z finding, just on a different finger (Ring, the
*normal*, magnet-equipped fingertip -- not the swapped-out magnet-free
board S3Z was found on) and a different sensor/axis (sensor 4, X, not
sensor 3, Z).

**Why this is more concerning than the first finding**: S3Z only glitched
under specific crosstalk conditions, and only on the magnet-free
replacement board -- plausibly a one-off consequence of that board's
different quiescent value sitting closer to the boundary. Ring's S4X
glitches **constantly, on completely ordinary, currently-in-service
hardware, with nothing touching it and no crosstalk load on it at all**.
That two different (finger, sensor, axis) triples both show the identical
overflow signature suggests this may be a more general susceptibility --
worth checking systematically across all 4 fingers x 5 sensors x 3 axes
(15 channels x 4 fingers = 60 total) for which ones sit close enough to
whatever boundary this is to be at risk, rather than treating each
occurrence as isolated.

**Practical impact on this session's work**: every Ring row was
automatically excluded from the genuine-grasp Contact-classifier training
data (the existing `glitched` exclusion in `build_true_zero_dataset.py`'s
convention, reused here) -- meaning **this data-collection session
contributed zero usable training rows for Ring specifically**, despite
Ring being included in the `all4` scenario recordings. This did not block
the retraining work (Index/Middle/Thumb data was unaffected and
plentiful), but Ring's Contact-classifier behaviour in a real grasp
remains just as uncharacterised after this session as before it.

---

## Root cause identified (2026-08-31): confirmed ADC/register wraparound
## at the current firmware GAIN_SEL/RES/HALLCONF configuration -- NOT a
## wiring/connector fault, NOT an int16-cast firmware logic bug

**Method**: (1) read `MagneticSensors/MCU/lib/MLX90393/MLX90393.cpp`'s
`convertRaw()` and `MagneticSensors/MCU/lib/eflesh/eflesh.cpp`'s
`initGroup()` to find the exact register configuration and conversion
formula actually used; (2) statistically re-analysed the existing 36-rep
`genuine_grasp_raw/*.json` session (already collected, no new Ring
hardware time spent) to check whether the glitch is isolated to one
channel or spread across Ring's board/bus; (3) derived the theoretical
raw-register full-scale rail from (1) and checked it against the observed
swing magnitudes from both this finding and the original S3Z finding.
Same diagnostic approach as the earlier per-unit hardware-fault
triage: isolate whether a fault is channel-specific (chip/config-level)
or shared across a bus/connector (wiring-level) before assuming either.

**Firmware configuration** (`eflesh.cpp::initGroup()`, applied identically
to all 20 sensors on connect): `GAIN_SEL=0x1` (gain factor 4.0),
`RES_XYZ=(2,2,2)` (resolution factor `1<<2=4` on every axis),
`HALLCONF=0x0C` (set in `MLX90393::begin()`, which selects sensitivity
`xy_sens=0.150`, `z_sens=0.242` -- the `hallconf==0xc` branch of
`convertRaw()`, not the default `hallconf==0` branch). For `RES=2`
(the configured value, not `RES=0/1`), `convertRaw()` uses
`(raw - 32768.f) * sens * gain * (1<<res)` -- a textbook-correct
unsigned-to-signed conversion of the MLX90393's native **16-bit** raw
register, not the buggy `int16_t(raw)` reinterpret-cast path (that path
is only reachable at `RES=0/1`, which this firmware never uses). **No
firmware logic bug in the conversion formula.**

**Theoretical full-scale rail from this config**:
- X/Y: `32768 * 0.150 * 4.0 * 4 = 78,643.2` per single reading; since the
  live pipeline's "raw_delta" is `(current reading) - (per-recording
  baseline reading)`, a channel whose raw register wraps from one rail to
  the other between baseline-capture and a later tick can show a **delta**
  up to double that: `157,286.4`.
- Z: `32768 * 0.242 * 4.0 * 4 = 126,877.7` per single reading, double-rail
  delta up to `253,755.4`.

**This matches the observed data to <0.1%, for two independent
instances**:
- Original S3Z finding: swing "~253,700" raw units -- predicted Z
  double-rail is `253,755.4` (0.02% off).
- A previously-unremarked **third instance** found while re-analysing this
  session's data for this investigation: `dS5Y` (Ring board, sensor 5,
  Y-axis) hit the `SANE_DELTA_MAX` guard exactly **once**, at `156,872` --
  predicted X/Y double-rail is `157,286.4` (0.3% off). One-off (not
  persistent, unlike S4X), consistent with sensor 5's Y-axis baseline
  usually sitting well clear of the rail but occasionally drifting close
  enough to touch it.
- Ring's `dS4X` (this note's original second-instance finding): max
  observed delta `104,827.5` -- below the full double-rail max (expected,
  since any single tick only captures whatever partial excursion is
  in progress relative to baseline) but far above every other channel's
  ~300-1,200-unit noise floor, and persistent on **100% of 7,670 Ring
  ticks across all 36 reps** (re-confirmed this session by re-analysing
  the existing JSON captures channel-by-channel).

**Isolation test (rules out wiring/connector)**: re-analysing the same
36-rep session broken out by all 15 of Ring's raw channels shows `dS4X`
alone glitched on 100% of ticks; `dS4Y` and `dS4Z` -- the **other two axes
of the exact same physical chip, read in the same I2C transaction** -- are
completely clean (max ~667/701 raw units), as are all 12 channels on
Ring's other four sensor chips (S1, S2, S3, S5), aside from the single
`dS5Y` one-off above. A loose connector, damaged trace, or bad solder
joint on Ring's board or its mux-channel wiring would be expected to
affect the shared I2C bus for that board (multiple sensors) or at minimum
all axes of one chip's single read transaction (`readMeasurement()` reads
T/X/Y/Z together) with I/O errors or dropped-to-zero readings (see
`collectGroup()`'s `isOK(status)` check, which zeros a *whole* sensor's
t/x/y/z on any I2C error) -- not a clean, deterministic, rail-matching
numeric value confined to one axis of one chip while its own T/Y/Z and
every sibling chip read normally, every single tick, for 15 minutes with
the finger sitting motionless. A live idle re-check was also attempted
this session (COM7) but the connection came up in a degenerate
all-zero-groups state (likely the physical rig currently in an
intermediate/disassembled configuration left over from the magnet-free
board swap work) -- inconclusive, not used as evidence either way; the
conclusion above rests on the firmware-formula match and the existing
36-rep dataset instead.

**Conclusion**: this is a **genuine chip-level / analog-front-end-range
limitation**, specific to individual (physical chip, axis) pairs whose
manufacturing-variance DC offset happens to sit close enough to the
MLX90393's ±32,768-count raw ADC rail, **given the firmware's current
shared GAIN_SEL=1/RES=2/HALLCONF=0x0C configuration** -- not a wiring or
connector fault, and not an int16-cast firmware bug (that code path isn't
even reached at RES=2). Different chips have different intrinsic offsets,
which is why it's S3Z on the (swapped) magnet-free Index board, S4X
persistently and S5Y once on Ring, rather than the same (sensor, axis)
recurring on every board.

**This is fixable without hardware rework**: lowering `GAIN_SEL` (less
analog gain) and/or lowering `RES` for the specific affected axis (per-
axis, via `setResolution(res_x, res_y, res_z)` -- already called per-chip
in `initGroup()`, so a chip-specific override is a small firmware change,
not a redesign) would shrink the raw-count-per-unit-field ratio and pull
the affected channel's quiescent point back off the rail, at the cost of
reduced resolution/dynamic range for legitimate signal on that one axis.
**Not attempted this session** (no firmware changes made, per the
report-before-acting brief) -- proposed next step if picked up: identify
the exact quiescent raw register value for Ring's sensor 4 (would need a
small firmware/debug build that streams the raw `txyzRaw` alongside the
converted value, since the current binary frame protocol only sends
converted floats) to pick the smallest RES/GAIN reduction that restores
headroom without over-shrinking resolution, then re-flash and re-verify
idle before any new Ring data collection.

---

## Fix implemented and validated (2026-08-31)

**Change**: `MagneticSensors/MCU/lib/eflesh/eflesh.cpp::initGroup()` --
added a chip-specific override right after the existing uniform
`setResolution(0x2, 0x2, 0x2)` call: for `groupIdx == 2` (Ring) and
sensor slot `i == 3` (S4, 0-indexed) only, `setResolution(0x0, 0x2, 0x2)`
lowers just that chip's X-axis RES from 2 to 0 (a 4x reduction). `GAIN_SEL`
(the analog front-end stage, shared across a chip's 3 axes) was
deliberately left untouched, and RES is per-axis, so this touches nothing
else: not the other 19 sensors, not this same chip's own Y/Z axes. Built
and uploaded via PlatformIO (`pio run --target upload`, `teensy-gui`
protocol) -- compiled clean, uploaded successfully.

**Validation**: reconnected live (COM7) and ran a fresh idle capture --
2s baseline average, then 15s / 1,576 ticks of Ring sitting untouched
(the same trigger condition that produced a 100% glitch rate before, since
S4X's fault was persistent/positional, not neighbour-motion-triggered like
S3Z). Result: **`S4X` max|delta| = 10.2 raw units across all 1,576 ticks,
zero exceedances of `SANE_DELTA_MAX=2000`** -- in line with every other
Ring channel's ~6-15 unit idle noise floor (previously: 104,827.5 max,
7,670/7,670 ticks glitched). All 14 other Ring channels remained clean
throughout, confirming the targeted fix didn't disturb anything else.

**Caveat**: validated under idle (no neighbour finger motion, no load) --
the same condition the original characterisation showed the fault under,
so this directly replicates the trigger condition. Not yet validated
under an actual loaded/moving-neighbour grasp session; the upcoming
expanded genuine-grasp data collection (which now includes Ring) will
serve as that real-world confirmation -- watch each session's `glitched`
flag rate for Ring specifically as a first check before trusting the
collected rows.
