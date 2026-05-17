"""Closed-form logistic-regression predictor (pure standard library).

The classifier is a scikit-learn ``Pipeline([StandardScaler, LogisticRegression])``
trained offline by ``ml/train.py``. At *runtime*, however, this module does NOT
import scikit-learn, numpy or joblib and does NOT unpickle anything. Instead it
reads the model's parameters from a plain JSON file and evaluates the prediction
directly with the closed-form expression:

    z_i  = (x_i - mean_i) / scale_i            # StandardScaler
    logit = b + Σ_i  w_i · z_i                 # LogisticRegression
    p     = 1 / (1 + e^(-logit))               # sigmoid

This has three big advantages for a parity-critical project:

1. **No library-version risk.** A pickled sklearn estimator only loads under a
   compatible sklearn/numpy build. A JSON of floats loads anywhere, forever.
2. **No code execution on load.** Unpickling can run arbitrary code; reading
   JSON cannot. Safer to ship.
3. **Trivial Python/JS parity.** The TypeScript port (`web/src/predict.ts`)
   implements the *same three lines*, reading the *same numbers* from
   `web/src/generated/model.ts`. There is no transpiled model to drift.

`models/model_meta.json` is the single source of truth and is committed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .features import FEATURE_NAMES, feature_vector

# <repo>/phishdetect/model.py -> parents[1] is <repo>.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "model_meta.json"


@dataclass(frozen=True)
class LogisticModel:
    """An immutable logistic-regression model loaded from ``model_meta.json``.

    Attributes
    ----------
    feature_names:
        Ordered feature names the coefficients are aligned to. Must equal
        :data:`phishdetect.features.FEATURE_NAMES`.
    coef:
        One weight per feature (the ``LogisticRegression.coef_`` row).
    intercept:
        The model bias term (``LogisticRegression.intercept_``).
    mean:
        Per-feature mean from the fitted ``StandardScaler``.
    scale:
        Per-feature standard deviation from the fitted ``StandardScaler``.
    threshold:
        Decision threshold on the probability for the positive (phishing) class.
    metrics:
        The held-out evaluation metrics recorded at training time.
    """

    feature_names: list[str]
    coef: list[float]
    intercept: float
    mean: list[float]
    scale: list[float]
    threshold: float
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        """Validate the parameter shapes are mutually consistent."""
        n = len(self.feature_names)
        if not (len(self.coef) == len(self.mean) == len(self.scale) == n):
            raise ValueError(
                "model_meta.json is inconsistent: feature_names, coef, mean and "
                f"scale must all have the same length (got {n}, {len(self.coef)}, "
                f"{len(self.mean)}, {len(self.scale)})."
            )
        if self.feature_names != FEATURE_NAMES:
            raise ValueError(
                "model_meta.json feature order does not match "
                "phishdetect.features.FEATURE_NAMES — the model is stale; "
                "re-run ml/train.py."
            )

    def standardize(self, raw: list[float]) -> list[float]:
        """Apply the fitted ``StandardScaler`` to a raw feature vector.

        A zero ``scale`` (a constant feature in training) maps to a zero
        standardized value, matching scikit-learn's ``StandardScaler`` which
        replaces a zero scale with 1.0 before dividing.
        """
        out: list[float] = []
        for x, m, s in zip(raw, self.mean, self.scale, strict=True):
            denom = s if s != 0.0 else 1.0
            out.append((x - m) / denom)
        return out

    def logit(self, raw: list[float]) -> float:
        """Return the raw logit ``b + Σ wᵢ·zᵢ`` for a raw feature vector."""
        z = self.standardize(raw)
        total = self.intercept
        for w, zi in zip(self.coef, z, strict=True):
            total += w * zi
        return total

    def predict_proba(self, raw: list[float]) -> float:
        """Return the phishing probability in ``[0, 1]`` for a raw feature vector."""
        return _sigmoid(self.logit(raw))

    def predict_url(self, url: str) -> float:
        """Convenience: extract features from ``url`` and return the probability."""
        return self.predict_proba(feature_vector(url))

    def contributions(self, raw: list[float]) -> list[tuple[str, float]]:
        """Return per-feature signed contributions to the logit.

        Each pair is ``(feature_name, w_i · z_i)``. A positive value pushes the
        verdict toward "phishing", a negative value toward "safe". This is what
        the UI shows as "top ML contributions".
        """
        z = self.standardize(raw)
        return [
            (name, w * zi) for name, w, zi in zip(self.feature_names, self.coef, z, strict=True)
        ]


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid ``1 / (1 + e^-x)``.

    Splitting on the sign of ``x`` avoids ``math.exp`` overflow for large
    magnitudes (the TypeScript port uses the identical guard).
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def load_model(path: str | Path | None = None) -> LogisticModel:
    """Load the logistic-regression model from a ``model_meta.json`` file.

    Parameters
    ----------
    path:
        Path to the JSON metadata file. Defaults to ``models/model_meta.json``
        at the repository root.

    Returns
    -------
    LogisticModel
        The validated, immutable model ready for prediction.

    Raises
    ------
    FileNotFoundError
        If the metadata file does not exist (run ``ml/train.py`` to create it).
    """
    model_path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found at {model_path}. Run `python ml/train.py` "
            "to train the model and write models/model_meta.json."
        )
    data = json.loads(model_path.read_text(encoding="utf-8"))
    return LogisticModel(
        feature_names=list(data["feature_names"]),
        coef=[float(c) for c in data["coef"]],
        intercept=float(data["intercept"]),
        mean=[float(m) for m in data["mean"]],
        scale=[float(s) for s in data["scale"]],
        threshold=float(data["threshold"]),
        metrics={k: float(v) for k, v in data.get("metrics", {}).items()},
    )
