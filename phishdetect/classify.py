"""End-to-end classification: features -> heuristics + model -> blended verdict.

This module ties the three halves of the detector together and produces the
single structured :class:`ClassificationResult` consumed by the CLI and mirrored
by the website's ``predict.ts``.

The blended score
-----------------
The ML model contributes a data-driven probability; the heuristic engine
contributes a transparent, explainable score. Neither is perfect alone, so the
final score is a fixed weighted blend:

    final = ML_WEIGHT * (ml_probability * 100) + HEURISTIC_WEIGHT * heuristic_score

with ``ML_WEIGHT = 0.60`` and ``HEURISTIC_WEIGHT = 0.40``. The model carries the
larger weight because it was fitted on tens of thousands of real URLs, while the
heuristics provide a sanity floor and the human-readable "why".

Bands
-----
The 0-100 final score maps to a verdict band:

    score <  35   -> "Safe"
    35 <= score < 65 -> "Suspicious"
    score >= 65   -> "Dangerous"

These exact weights and thresholds are duplicated in ``web/src/predict.ts`` so
the browser and the CLI always agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .features import FEATURE_NAMES, extract_features_from_parsed
from .heuristics import HeuristicResult, RuleHit, evaluate
from .model import LogisticModel, load_model
from .urlparse import parse_url

# --- Blend weights and band thresholds (mirrored in web/src/predict.ts) ----------
ML_WEIGHT = 0.60
HEURISTIC_WEIGHT = 0.40

BAND_SUSPICIOUS_AT = 35.0
BAND_DANGEROUS_AT = 65.0

# How many signed ML contributions the result carries (the UI shows the top N).
TOP_CONTRIBUTIONS = 6


@dataclass(frozen=True)
class FeatureContribution:
    """A single feature's signed push on the ML logit.

    A positive ``contribution`` moved the verdict toward "phishing"; a negative
    one toward "safe".
    """

    name: str
    value: float
    contribution: float


@dataclass(frozen=True)
class ClassificationResult:
    """The complete, structured outcome of classifying one URL.

    Attributes
    ----------
    url:
        The input URL string (trimmed, as provided).
    final_score:
        The blended phishing-risk score, 0-100 (rounded to one decimal).
    band:
        ``"Safe"``, ``"Suspicious"`` or ``"Dangerous"``.
    ml_probability:
        The model's raw phishing probability, 0-1.
    heuristic_score:
        The transparent heuristic engine's score, 0-100.
    features:
        The full 20-feature dict.
    contributions:
        The top signed ML feature contributions, largest magnitude first.
    reasons:
        The triggered heuristic rules (human-readable explanations).
    """

    url: str
    final_score: float
    band: str
    ml_probability: float
    heuristic_score: int
    features: dict[str, float] = field(default_factory=dict)
    contributions: list[FeatureContribution] = field(default_factory=list)
    reasons: list[RuleHit] = field(default_factory=list)

    @property
    def is_dangerous(self) -> bool:
        """True if the verdict band is ``"Dangerous"``."""
        return self.band == "Dangerous"


def _band_for(score: float) -> str:
    """Map a 0-100 final score to its verdict band."""
    if score >= BAND_DANGEROUS_AT:
        return "Dangerous"
    if score >= BAND_SUSPICIOUS_AT:
        return "Suspicious"
    return "Safe"


def classify(url: str, model: LogisticModel | None = None) -> ClassificationResult:
    """Classify a single URL and return a structured :class:`ClassificationResult`.

    The pipeline is: deterministic parse -> 20 features -> heuristic rules
    (score + reasons) -> ML probability -> blended final score -> band.

    Parameters
    ----------
    url:
        Any URL string. A missing scheme is handled by the shared parser.
    model:
        A pre-loaded :class:`LogisticModel`. If omitted, the default model
        (``models/model_meta.json``) is loaded once per call — pass an explicit
        model when classifying in bulk to avoid repeated disk reads.

    Returns
    -------
    ClassificationResult
        The full verdict, breakdown and explanations.
    """
    mdl = model if model is not None else load_model()

    parsed = parse_url(url)
    features = extract_features_from_parsed(parsed)
    raw_vector = [features[name] for name in FEATURE_NAMES]

    heuristics: HeuristicResult = evaluate(features)
    ml_probability = mdl.predict_proba(raw_vector)

    final_score = ML_WEIGHT * (ml_probability * 100.0) + HEURISTIC_WEIGHT * float(heuristics.score)
    final_score = max(0.0, min(100.0, final_score))

    signed = mdl.contributions(raw_vector)
    signed.sort(key=lambda pair: abs(pair[1]), reverse=True)
    contributions = [
        FeatureContribution(name=name, value=features[name], contribution=contrib)
        for name, contrib in signed[:TOP_CONTRIBUTIONS]
    ]

    return ClassificationResult(
        url=parsed.original,
        final_score=round(final_score, 1),
        band=_band_for(final_score),
        ml_probability=ml_probability,
        heuristic_score=heuristics.score,
        features=features,
        contributions=contributions,
        reasons=list(heuristics.hits),
    )


def result_to_dict(result: ClassificationResult) -> dict[str, object]:
    """Convert a :class:`ClassificationResult` to a JSON-serialisable ``dict``.

    Used by the CLI's ``--json`` flag and by the parity tooling.
    """
    return {
        "url": result.url,
        "final_score": result.final_score,
        "band": result.band,
        "ml_probability": round(result.ml_probability, 6),
        "heuristic_score": result.heuristic_score,
        "features": {k: result.features[k] for k in FEATURE_NAMES},
        "contributions": [
            {
                "name": c.name,
                "value": c.value,
                "contribution": round(c.contribution, 6),
            }
            for c in result.contributions
        ],
        "reasons": [
            {
                "rule_id": h.rule_id,
                "points": h.points,
                "reason": h.reason,
                "severity": h.severity,
            }
            for h in result.reasons
        ],
    }
