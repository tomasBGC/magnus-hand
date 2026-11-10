"""Real-time inference wrapper around force_direction_model_v1_contact.pkl.

Loads the model bundle once (via LivePhiClassPredictor.__init__), then can
be called repeatedly and cheaply as new sensor frames arrive.

The bundle holds THREE learners sharing one StandardScaler (Force and Theta
are two outputs of a single joint MLPRegressor, not separate networks):
  - regressor:          MLPRegressor, 20 inputs -> [Force (N), Theta (deg)]
  - classifier:          MLPClassifier, 20 inputs -> Phi class in {-90, 0, +90}
  - contact_classifier:  MLPClassifier, 20 inputs -> Contact class in {0, 1}

contact_classifier is a dedicated no-contact detector, trained with balanced
class weighting on an explicit Contact label (Force < 0.3N -> 0, else -> 1)
-- see model_v1/train_model_contact.py. It replaces inferring "no contact"
indirectly from the Force regression output, which was wrong on ~45% of
genuinely idle rows (the Force regressor's own low-force resolution is
coarse and Theta on idle rows is a leaky recording-level label, not a
physically meaningful direction).

The 20 inputs are 15 delta features (dS1X..dS5Z, baseline-subtracted) plus
5 per-sensor magnitudes of those deltas (S1_mag..S5_mag). No ForceDirection
/ hysteresis feature and no bootstrap pass are needed here (unlike the v3
hysteresis predictor) -- every input is derivable from the current frame
alone, so predict() is a single forward pass per learner.
"""
import numpy as np
import joblib

DEFAULT_MODEL_PATH = "force_direction_model_v1_contact.pkl"


class LivePhiClassPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        bundle = joblib.load(model_path)
        self.regressor = bundle["regressor"]
        self.classifier = bundle["classifier"]
        self.contact_classifier = bundle["contact_classifier"]
        self.scaler = bundle["scaler"]
        self.input_cols = bundle["input_cols"]
        self.delta_cols = bundle["delta_cols"]
        self.magnitude_cols = bundle["magnitude_cols"]
        self.regression_output_cols = bundle["regression_output_cols"]
        self.phi_classes = bundle["phi_classes"]
        self.contact_classes = bundle["contact_classes"]
        self.contact_force_threshold = bundle["contact_force_threshold"]

        # delta_cols are named "dS1X".."dS5Z"; raw_cols are the corresponding
        # live sensor field names "S1X".."S5Z" (strip the leading "d").
        self.raw_cols = [c[1:] for c in self.delta_cols]
        self.sensor_triplets = [self.raw_cols[i:i + 3] for i in range(0, 15, 3)]

    def _build_feature_row(self, raw_delta_values: dict) -> np.ndarray:
        deltas = [raw_delta_values[raw_col] for raw_col in self.raw_cols]
        magnitudes = [
            np.sqrt(sum(raw_delta_values[c] ** 2 for c in triplet))
            for triplet in self.sensor_triplets
        ]
        return np.array(deltas + magnitudes, dtype=float).reshape(1, -1)

    def predict(self, raw_delta_values: dict) -> dict:
        """Predict Force (N), Theta (deg), a discrete Phi class (deg), and a
        binary Contact class from one instant of baseline-subtracted sensor
        data.

        raw_delta_values: dict with keys S1X, S1Y, S1Z, ..., S5X, S5Y, S5Z
            (raw magnetic readings for the current moment, MINUS the
            per-channel resting baseline -- same convention as the delta
            features this model was trained on).

        Returns: {
            "Force": float, "Theta": float,
            "Phi": int (one of -90, 0, 90),
            "Phi_probs": {-90: float, 0: float, 90: float},
            "Contact": int (0 = no contact, 1 = contact),
            "Contact_probs": {0: float, 1: float},
        }

        Callers should gate the no-contact display override on "Contact",
        not on a Force threshold -- that's the whole point of having a
        dedicated classifier for it (see module docstring).
        """
        features = self._build_feature_row(raw_delta_values)
        features_scaled = self.scaler.transform(features)

        force_pred, theta_pred = self.regressor.predict(features_scaled)[0]
        phi_pred = self.classifier.predict(features_scaled)[0]
        phi_probs_arr = self.classifier.predict_proba(features_scaled)[0]
        phi_probs = {cls: float(p) for cls, p in zip(self.classifier.classes_, phi_probs_arr)}

        contact_pred = self.contact_classifier.predict(features_scaled)[0]
        contact_probs_arr = self.contact_classifier.predict_proba(features_scaled)[0]
        contact_probs = {
            cls: float(p) for cls, p in zip(self.contact_classifier.classes_, contact_probs_arr)
        }

        return {
            "Force": float(force_pred),
            "Theta": float(theta_pred),
            "Phi": int(phi_pred),
            "Phi_probs": phi_probs,
            "Contact": int(contact_pred),
            "Contact_probs": contact_probs,
        }
