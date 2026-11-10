# Architecture comparison: deployed MLP heads vs GBM heads

Same split as evaluate_model_forcefix_v2.py: train=360404 rows, test=126061 rows (12 held-out recordings). GBM heads use RAW features -- no scaler exists in the GBM path at all.

Training times: gbm_force=34.1s, gbm_theta=48.7s, gbm_phi=188.7s, gbm_phi_nz=144.9s, gbm_contact=100.3s

### All test rows (n=126061)

| Metric | Deployed (MLP) | GBM | GBM Phi (no theta=0) |
|---|---|---|---|
| Force MAE (N) | 1.3288 | 1.2170 | -- |
| Theta MAE (deg) | 12.6300 | 18.2446 | -- |
| Phi acc | 0.5707 | 0.5991 | 0.6327 |
| Contact acc | 0.7771 | 0.7798 | -- |

### Unit 1 (n=49054)

| Metric | Deployed (MLP) | GBM | GBM Phi (no theta=0) |
|---|---|---|---|
| Force MAE (N) | 0.9449 | 0.7895 | -- |
| Theta MAE (deg) | 12.2875 | 22.4055 | -- |
| Phi acc | 0.6445 | 0.5932 | 0.6119 |
| Contact acc | 0.9247 | 0.9224 | -- |

### Unit 2 (n=54295)

| Metric | Deployed (MLP) | GBM | GBM Phi (no theta=0) |
|---|---|---|---|
| Force MAE (N) | 0.9351 | 0.9380 | -- |
| Theta MAE (deg) | 14.9253 | 21.6064 | -- |
| Phi acc | 0.6714 | 0.6123 | 0.6779 |
| Contact acc | 0.8555 | 0.8447 | -- |

### Unit 3 (n=22712)

| Metric | Deployed (MLP) | GBM | GBM Phi (no theta=0) |
|---|---|---|---|
| Force MAE (N) | 3.0994 | 2.8076 | -- |
| Theta MAE (deg) | 7.8828 | 1.2213 | -- |
| Phi acc | 0.1707 | 0.5803 | 0.5698 |
| Contact acc | 0.2706 | 0.3166 | -- |

### Back region (theta=90, all units) (n=40751)

| Metric | Deployed (MLP) | GBM | GBM Phi (no theta=0) |
|---|---|---|---|
| Force MAE (N) | 2.4448 | 2.3107 | -- |
| Theta MAE (deg) | 14.3244 | 12.7607 | -- |
| Phi acc | 0.4161 | 0.5671 | 0.5715 |
| Contact acc | 0.4640 | 0.4752 | -- |

### Theta >= 30 (Phi physically meaningful) (n=93897)

| Metric | Deployed (MLP) | GBM | GBM Phi (no theta=0) |
|---|---|---|---|
| Force MAE (N) | 1.5098 | 1.3873 | -- |
| Theta MAE (deg) | 12.3936 | 15.6901 | -- |
| Phi acc | 0.6758 | 0.7167 | 0.7184 |
| Contact acc | 0.7261 | 0.7299 | -- |

### Units 1+2, theta >= 30 (n=71185)

| Metric | Deployed (MLP) | GBM | GBM Phi (no theta=0) |
|---|---|---|---|
| Force MAE (N) | 1.0026 | 0.9341 | -- |
| Theta MAE (deg) | 13.8328 | 20.3064 | -- |
| Phi acc | 0.8370 | 0.7603 | 0.7658 |
| Contact acc | 0.8714 | 0.8618 | -- |

### Contact false-positive rate on idle rows, Units 1+2 (drives live false-contact noise)

- Deployed: 0.0638
- GBM:      0.0551

### Contact false-negative rate on touch rows, Units 1+2

- Deployed: 0.1274
- GBM:      0.1393

### Self-closure regression check (GBM contact head)

- Training-fit accuracy on ALL self-closure rows (Contact=0 expected): 0.9998 (deployed fix scored 0.9967 training-fit)

### Idle-row predicted-Force distribution, Units 1+2 (true Force < 0.3N)

- Deployed: mean=+0.349N  p50=+0.190  p90=+0.604  p99=+4.807
- GBM: mean=+0.285N  p50=+0.126  p90=+0.509  p99=+4.883

Candidate heads saved to model_v1/production_review/candidate_gbm_heads.pkl (not deployed).