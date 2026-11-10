# Candidate Phi head: MLP retrained without theta=0 rows

| Slice | n | Deployed Phi acc | Candidate Phi acc |
|---|---|---|---|
| All test | 126061 | 0.5707 | 0.6604 |
| Units 1+2 | 103349 | 0.6586 | 0.6820 |
| Units 1+2, theta>=30 | 71185 | 0.8370 | 0.8405 |
| Units 1+2, theta in {15,30} | 19072 | 0.6426 | 0.5465 |
| theta=90 (all units) | 40751 | 0.4161 | 0.6437 |
| theta=0 (degenerate labels; expect ~chance for both) | 21259 | 0.1501 | 0.3411 |

Per-theta accuracy (Units 1+2):

| Theta | n | Deployed | Candidate |
|---|---|---|---|
| 0 | 21259 | 0.1501 | 0.3411 |
| 15 | 10905 | 0.4856 | 0.3121 |
| 30 | 8167 | 0.8523 | 0.8596 |
| 45 | 12934 | 0.9242 | 0.9182 |
| 60 | 22186 | 0.8566 | 0.8534 |
| 75 | 9859 | 0.8704 | 0.8654 |
| 90 | 18039 | 0.7251 | 0.7466 |

Confidence gating (Units 1+2, theta>=30): accuracy and coverage above a max-prob threshold:

| Threshold | Deployed acc @ coverage | Candidate acc @ coverage |
|---|---|---|
| 0.5 | 0.9505 @ 78.9% | 0.9505 @ 79.7% |
| 0.7 | 0.9882 @ 70.3% | 0.9881 @ 71.2% |
| 0.9 | 0.9981 @ 65.2% | 0.9984 @ 66.1% |

Candidate head saved to model_v1/production_review/candidate_phi_no_theta0.pkl (not deployed).