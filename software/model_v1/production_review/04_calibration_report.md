# Confidence calibration report

## Contact classifier

Isotonic calibrator fit on train-set P(contact) (caveat: train probs are mildly optimistic since the MLP saw those rows), evaluated on the held-out test set.

| Slice | n | ECE raw | ECE calibrated |
|---|---|---|---|
| All test | 126061 | 0.1741 | 0.1372 |
| Units 1+2 | 103349 | 0.0534 | 0.0072 |
| Unit 3 | 22712 | 0.7289 | 0.7289 |
| theta=90 | 40751 | 0.4935 | 0.4552 |

### Probability-Schmitt policy (Units 1+2 rows)

Current live rule: hard predict()==1 AND Force>0.1N counts toward the confirm streak. Candidate rule: calibrated P(contact) >= P_hi to confirm, <= P_lo to release.

Three health metrics per policy: idle-confirm rate (drives FALSE contact -- lower is better), touch-release rate (drives dropped/flickering contact during a real press -- lower is better), and idle-release rate (must be HIGH, or a once-confirmed contact can never release).

- Current rule: idle-confirm 0.0633; touch-release 0.1277; idle-release 0.9367
- Schmitt(calibrated) P_hi=0.6/P_lo=0.3: idle-confirm 0.0952; touch-release 0.0451; idle-release 0.6358
- Schmitt(calibrated) P_hi=0.7/P_lo=0.3: idle-confirm 0.0662; touch-release 0.0451; idle-release 0.6358
- Schmitt(calibrated) P_hi=0.8/P_lo=0.2: idle-confirm 0.0402; touch-release 0.0194; idle-release 0.4142
- Schmitt(calibrated) P_hi=0.8/P_lo=0.3: idle-confirm 0.0402; touch-release 0.0451; idle-release 0.6358
- Schmitt(calibrated) P_hi=0.9/P_lo=0.5: idle-confirm 0.0242; touch-release 0.0969; idle-release 0.8697
- Schmitt(raw) P_hi=0.6/P_lo=0.3: idle-confirm 0.0454; touch-release 0.0908; idle-release 0.8493
- Schmitt(raw) P_hi=0.7/P_lo=0.3: idle-confirm 0.0335; touch-release 0.0908; idle-release 0.8493
- Schmitt(raw) P_hi=0.8/P_lo=0.2: idle-confirm 0.0242; touch-release 0.0596; idle-release 0.7203
- Schmitt(raw) P_hi=0.8/P_lo=0.3: idle-confirm 0.0242; touch-release 0.0908; idle-release 0.8493
- Schmitt(raw) P_hi=0.9/P_lo=0.5: idle-confirm 0.0169; touch-release 0.1274; idle-release 0.9362

## Phi classifier (confidence -> P(correct))

| Slice | n | ECE raw | ECE calibrated |
|---|---|---|---|
| All test | 126061 | 0.2624 | 0.2663 |
| Units 1+2, theta>=30 | 71185 | 0.0111 | 0.0125 |
| theta=90 | 40751 | 0.4655 | 0.4683 |
| theta<30 | 32164 | 0.4602 | 0.4667 |

Interpretation note: even a perfect calibrator cannot fix Phi at theta<30 -- the azimuth is physically degenerate there and the honest output is a LOW confidence, which is what calibration should produce.

Calibrators saved to model_v1/production_review/candidate_calibrators.pkl (not deployed).
### Addendum: touch-confirm sensitivity check (Units 1+2 touch rows)

Confirm-side sensitivity by true-force band (fraction of genuine-touch frames that would feed the confirm streak):

| Band (N) | Current rule | Schmitt(raw) 0.6/0.3 | Schmitt(raw) 0.8/0.3 |
|---|---|---|---|
| [0.3,0.6) | 0.287 | 0.235 | 0.153 |
| [0.6,1.0) | 0.506 | 0.450 | 0.339 |
| [1.0,2.0) | 0.828 | 0.798 | 0.723 |
| [2.0,100) | 0.957 | 0.952 | 0.939 |

Honest trade-off: raising P_hi buys idle stability but costs light-touch (<1N) confirm sensitivity, which is already the weakest band. P_hi=0.6/P_lo=0.3 is the balanced recommendation: idle-confirm 0.0633->0.0454 (-28%), touch-release 0.1277->0.0908 (-29%), light-touch confirm cost ~2-6pp (vs ~13-17pp at P_hi=0.8). P_hi=0.8 remains a documented calm-mode option if false contacts matter more than light-touch latency in a given demo.