"""(a) Deployed contact classifier accuracy per finger on the standard holdout."""
import pandas as pd

import common as c

df = c.load_dataset()
b = c.load_bundle()
_, hold = c.standard_split(df)
pred = b["contact_classifier"].predict(b["scaler"].transform(hold[c.INPUT_COLS].values))
hold = hold.assign(correct=(pred == hold["Contact"].values))
by_finger = {f: s for f, s in hold.groupby("finger")}
rows = [dict(finger=f, n=len(by_finger[f]), accuracy=by_finger[f].correct.mean()) for f in c.FINGERS]
rows.append(dict(finger="ALL", n=len(hold), accuracy=hold.correct.mean()))
out = pd.DataFrame(rows)
print(out.to_string(index=False))
print(f"saved {c.save_csv(out, 'a_contact_accuracy.csv')}")
print("Check vs earlier:")
for f, e in zip(c.FINGERS, [0.990, 0.931, 0.977, 0.529]):
    c.compare(f, out.set_index("finger").loc[f, "accuracy"], e, 0.02)
