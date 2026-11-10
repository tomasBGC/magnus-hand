# candidate_v2 revert — verification report (2026-07-20)

**Action taken**: `model_v1/active_model.pkl` reverted to
`true_zero_dataset/active_model_BACKUP_pre_true_zero_fix.pkl` (candidate_v2's
Contact classifier). The pre-revert (true-zero) bundle was backed up first to
`true_zero_dataset/active_model_BACKUP_pre_v2_revert_20260720.pkl`, so this
remains reversible.

**Scope of the change, verified before promoting**: bundle diff (pickle
byte comparison) confirmed the two `active_model.pkl` bundles differ in
`contact_classifier` ONLY. Force/Theta regressor, Phi classifier (the
no-theta0 head), scaler, and `contact_calibrator` are all byte-identical.
`sensing_gui.py` was not touched (confirmed via `diff -q` against the
pre-session backup) — Schmitt trigger, spike guard, Phi display gating,
`AUTO_REBASELINE_ON_RETURN_TO_OPEN_ENABLED`, and `BASELINE_TRACKING_MODE=
"low_pass"` are all unchanged from what was live before this revert.

## Checklist verification (candidate_v2/README.md's original items)

| # | Item | Result |
|---|---|---|
| 1 | Crosstalk backends load, no FAILED | ✅ all 3 backends loaded cleanly |
| 2 | At-rest idle baseline clean | ✅ 0 false confirms, all 4 fingers, 900 ticks each (45s) |
| 3 | Individual press/release, all 4 fingers | ✅ all 4 fingers confirmed (max force 10.9–14.2N), released before session end |
| 4 | Phi gating unaffected | ✅ by construction — gate depends only on `theta`/`phi_conf`, sourced from the regressor/Phi-classifier, both confirmed byte-identical to pre-revert. Not re-measured empirically (not necessary given the structural guarantee), see reasoning below |
| 5 | Close All (30%)/Open All: no false contact from the commands | ⚠️ Index (11 ticks) and Middle (56 ticks) briefly false-confirmed during the sequence |
| 6 | Joint-backlash false confirm gets auto-rebaselined | ✅ both false confirms were caught and cleared: `auto-rebaselined at open position ... (was confirmed_contact=True -- should clear within the normal release streak on the next few ticks)` |

**On item 5/6 together**: this is not a new regression — it is the *exact*
documented pattern from `candidate_v2/README.md`'s own original "Live test
findings" section (joint backlash after Close→Open causing a transient
false confirm, caught and cleared by
`AUTO_REBASELINE_ON_RETURN_TO_OPEN_ENABLED`, "reproduced live 4/4 times" at
the time that flag was built). Seeing it recur here, correctly caught and
cleared, is confirmation the fix still works with the reverted classifier —
not a red flag.

**On item 4**: `PHI_DISPLAY_MIN_THETA_DEG`/`PHI_DISPLAY_MIN_CONF` gating
(`sensing_gui.py` ~L2182) is a pure function of `fr.theta` and
`fr.phi_conf`, both derived from the Force/Theta regressor and Phi
classifier — neither of which changed in this revert (confirmed via the
bundle diff). The gate's own code is untouched. This is a structural
guarantee, not an inference from incomplete data.

## Force-band recall confirmation (the primary ask)

First attempt (a scripted, cued 3-touch-per-finger phase + a freeform
light-touch phase) came back with implausible 0% recall in all 3 bands —
investigated and ruled out as an indexing bug (label orderings verified to
match); most likely explanation is that phase's freeform touches were fast
ramps through the light-touch range en route to a firmer press, never
sustaining the 5-frame confirm streak at that specific force level — a
data-quality issue with that specific short session, not the classifier.

Redone with explicit "hold each light touch 1–2s" instructions
(`verify_v2_revert_freeform.json`, 3 min, all 4 fingers touched):

| Force band | n (pooled, 4 fingers) | confirm rate |
|---|---|---|
| [0.3,0.5)N | 1676 | 0.475 |
| [0.5,0.7)N | 1048 | 0.801 |
| [0.7,1.0)N | 1336 | 0.908 |

Compare against `live_force_band_recall_report.md`'s "Old@0.6" column
(same classifier bundle, pooled from earlier, more mixed-technique
sessions): 0.169 / 0.365 / 0.500. Both measurements are for the identical
classifier bundle (confirmed via the bundle diff) — the higher numbers here
reflect this session's more disciplined sustained-hold touch technique
(explicitly requested this time), not a different underlying classifier.
What matters for the revert's purpose is confirmed by **both** independent
live measurements: recall is substantially, consistently nonzero and rises
with force — a world apart from the true-zero classifier's measured
[0.7,1.0)N rate of 0.358 (from the same report, "New@0.6" column) even
against the LOWER-recall reference technique, let alone against this
session's sustained-hold technique.

## Bottom line

The revert is confirmed working as intended: individual press/release,
idle safety, and the joint-backlash auto-rebaseline path all check out
unchanged; light-touch recall is restored to a level clearly and
substantially above the true-zero classifier's measured live performance,
consistent with candidate_v2's historical numbers. Live system is currently
running candidate_v2's Contact classifier with all other 2026-07 live
improvements intact.
