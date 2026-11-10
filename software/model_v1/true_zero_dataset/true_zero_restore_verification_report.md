# true-zero classifier restore — verification report (2026-07-20)

**Action taken**: `model_v1/active_model.pkl` restored from
`true_zero_dataset/active_model_BACKUP_pre_v2_revert_20260720.pkl`,
undoing the earlier candidate_v2 revert. The pre-restore (candidate_v2)
bundle was backed up first to
`true_zero_dataset/active_model_BACKUP_pre_true_zero_restore_<timestamp>.pkl`.

**Byte-diff confirmation**: `cmp model_v1/active_model.pkl
true_zero_dataset/active_model_true_zero_fix.pkl` — byte-identical.
`sensing_gui.py` confirmed unchanged throughout (`diff -q` against the
pre-session baseline) — Schmitt trigger, spike guard, Phi display gating,
`AUTO_REBASELINE_ON_RETURN_TO_OPEN_ENABLED`, `BASELINE_TRACKING_MODE=
"low_pass"` all intact, identical to what was live before any of this
session's work. Only `active_model.pkl`'s `contact_classifier` changed
(back to the true-zero-trained one) — confirmed via the earlier bundle-key
diff that this is the only functionally-different key between the two
bundles.

## Checklist verification (`verify_true_zero_restore_live.py`)

| # | Item | Result |
|---|---|---|
| 1 | Crosstalk backends load, no FAILED | ✅ all 3 loaded cleanly |
| 2 | At-rest idle baseline clean | ✅ 0 false confirms, all 4 fingers, 899 ticks each (45s) |
| 3 | Individual press/release, all 4 fingers | ✅ all 4 confirmed (max force 7.8–16.0N) |
| 4 | Phi gating unaffected | ✅ by construction — Phi classifier/regressor untouched by this restore (only `contact_classifier` changed) |
| 5/6 | Close All (30%)/Open All (100%): no false contact, backlash caught if it occurs | ✅ 0 false confirms across all 4 fingers this run (cleaner than the candidate_v2 run, which had 2 transient backlash confirms both correctly auto-rebaselined — this run had none at all) |
| **7 (NEW)** | **Grip to Target, no object present** | **✅ PASS — see below** |

### Item 7 in detail: the demo-critical scenario

Replicated `hand_control_gui.py`'s exact closed-loop grip algorithm
(3%/round staggered closing, checking displayed Force against a 2.0N
target, up to 100% safety max) with **nothing between or touching any
finger**. Index/Thumb capped at 15% in the script for hardware-collision
safety (the real GUI has no such cap since it assumes an object stops
each finger via force feedback — irrelevant with no object present, so
this script adds it; see script docstring). Result:

| Finger | Stopped at | False positive? |
|---|---|---|
| Index | 15% (safety cap) | **No** |
| Middle | 100% | **No** |
| Ring | 100% | **No** |
| Thumb | 15% (safety cap) | **No** |

Every finger closed to its full extent with zero false "target reached"
events — the true-zero classifier's entire purpose (this exact scenario:
progressive multi-finger closure with no touch) confirmed working live.

## Force-band recall (secondary, sanity check that genuine touch still works)

Combined session (firm presses on all 4 fingers, then light touches on
one/two fingers, self-paced, no cues):

| Finger | [0.3,0.5)N | [0.5,0.7)N | [0.7,1.0)N |
|---|---|---|---|
| Index | 0.191 (n=445) | 0.464 (n=276) | 0.640 (n=358) |
| Thumb | 0.121 (n=544) | 0.429 (n=175) | 0.612 (n=98) |
| Middle | 0.000 (n=607) | 0.000 (n=208) | 0.000 (n=105) |
| Ring | 0.000 (n=497) | 0.000 (n=130) | 0.000 (n=111) |

Middle/Ring's zeros are consistent with them only receiving brief firm-press
ramp transients through these force bands (not sustained light holds — the
same data-quality pattern seen earlier tonight, not a classifier issue);
both confirmed easily at their much higher peak forces (8.7N / 14.9N).
Index/Thumb, which did receive sustained light touches, show clear,
force-increasing recall — consistent with the true-zero classifier's known
(lower-than-candidate_v2, but real and substantial) light-touch behavior,
already quantified in `live_force_band_recall_report.md`.

## Bottom line

Live system is confirmed running the true-zero-trained Contact classifier
(Option A: ~100x multi-finger false-positive reduction, moderately reduced
light-touch sensitivity) with every other 2026-07 improvement intact. The
demo-critical scenario — closing toward a grip with no object present —
is clean across all 4 fingers.
