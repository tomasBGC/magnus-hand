# Crosstalk backend candidate: origin-anchored quadratic ridge

Fit on 80 train combos, scored on 20 held-out combos against crosstalk_from_others (same split/target as comparison_report_corrected.md).

| Finger | (a) pairwise-additive | (b) NN raw | (c) NN smoothed | (d) ridge quad | ridge alpha |
|---|---|---|---|---|---|
| Index | 35.844 | 19.770 | 19.355 | **17.140** | 0.1 |
| Middle | 43.353 | 50.138 | 47.606 | **39.995** | 3.0 |
| Ring | 34.996 | 47.246 | 38.754 | **32.962** | 1.0 |
| Thumb | 3.203 | 4.333 | 3.349 | **3.465** | 0.3 |
| **Overall** | 29.349 | 30.372 | 27.266 | **23.390** | |

Origin check: prediction at all-others-at-0% is exactly 0 by construction (all 9 features vanish).

Verified numerically for all 4 fingers.

Candidate model saved to candidate_crosstalk_ridge_v1.json (not wired into sensing_gui.py).
## Robustness check: 5x repeated 5-fold CV over all 100 combos

Single-split results above could be split luck; this repeats the comparison
across 25 held-out evaluations (every combo held out 5 times):

| Finger | pairwise-additive | NN smoothed | ridge quad |
|---|---|---|---|
| Index | 37.898 | 18.172 | **17.743** |
| Middle | 46.778 | 43.391 | **41.679** |
| Ring | 32.159 | 35.578 | **32.044** |
| Thumb | 4.715 | **3.771** | 4.098 |

Honest read: ridge is best or tied-best for Index/Middle/Ring, and the
Middle win over its current deployed backend (pairwise, 46.8 -> 41.7,
-11%) is the largest single improvement available. Ring is a tie with its
current backend. Thumb differences are all within its ~2.4-unit noise
floor. Index's offline gain over its current NN-smoothed backend is small
(18.2 -> 17.7); the structural properties (exact origin anchor, continuity,
trained on real 4-finger simultaneous data) are the stronger argument.
Offline MAE has already been shown to mispredict live behaviour once
(plain NN's live regression) -- any backend switch must be gated on the
counterbalanced realistic_session_diagnostic A/B before deployment.
