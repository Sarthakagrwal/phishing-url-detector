"""Tests for the closed-form logistic-regression predictor.

Checks that the model loads from ``model_meta.json``, produces probabilities in
``[0, 1]``, ranks an obvious phishing URL above a known-good one, and that its
parameter arrays are aligned to the fixed feature order.
"""

from __future__ import annotations

import math

from phishdetect.features import FEATURE_NAMES, feature_vector
from phishdetect.model import load_model


def test_model_loads_from_json() -> None:
    """The model loads from models/model_meta.json with consistent shapes."""
    model = load_model()
    n = len(FEATURE_NAMES)
    assert len(model.coef) == n
    assert len(model.mean) == n
    assert len(model.scale) == n
    assert model.feature_names == FEATURE_NAMES


def test_probabilities_are_in_unit_interval() -> None:
    """Every prediction is a probability in [0, 1]."""
    model = load_model()
    urls = [
        "https://github.com",
        "http://192.168.0.1/login/verify/account",
        "http://a-b-c-secure.gq@1.2.3.4/webscr",
        "https://example.com",
        "",
    ]
    for url in urls:
        p = model.predict_proba(feature_vector(url))
        assert 0.0 <= p <= 1.0


def test_known_good_url_scores_low() -> None:
    """https://github.com — an obvious legitimate site — scores well below 0.5."""
    model = load_model()
    assert model.predict_proba(feature_vector("https://github.com")) < 0.35


def test_blatant_phishing_url_scores_high() -> None:
    """A blatant phishing URL scores well above 0.5."""
    model = load_model()
    url = "http://secure-login-verify-account.paypal-update.gq/webscr/signin"
    assert model.predict_proba(feature_vector(url)) > 0.65


def test_phishing_ranks_above_legitimate() -> None:
    """A phishing URL's probability exceeds a known-good URL's probability."""
    model = load_model()
    legit = model.predict_proba(feature_vector("https://www.wikipedia.org"))
    phish = model.predict_proba(feature_vector("http://203.0.113.9/account/verify/login.php"))
    assert phish > legit


def test_predict_url_matches_predict_proba() -> None:
    """The predict_url convenience equals predict_proba on the same vector."""
    model = load_model()
    url = "https://example.com/login"
    assert model.predict_url(url) == model.predict_proba(feature_vector(url))


def test_closed_form_matches_manual_sigmoid() -> None:
    """The model's probability equals a hand-computed sigmoid of the logit."""
    model = load_model()
    raw = feature_vector("http://login-secure.example.xyz/verify")
    # Recompute logit + sigmoid by hand from the stored parameters.
    logit = model.intercept
    for x, w, m, s in zip(raw, model.coef, model.mean, model.scale, strict=True):
        denom = s if s != 0.0 else 1.0
        logit += w * ((x - m) / denom)
    expected = 1.0 / (1.0 + math.exp(-logit))
    assert abs(model.predict_proba(raw) - expected) < 1e-12


def test_contributions_sum_relates_to_logit() -> None:
    """Per-feature contributions plus the intercept reconstruct the logit."""
    model = load_model()
    raw = feature_vector("http://secure-login.example.tk/account")
    contributions = model.contributions(raw)
    assert len(contributions) == len(FEATURE_NAMES)
    reconstructed = model.intercept + sum(c for _, c in contributions)
    assert abs(reconstructed - model.logit(raw)) < 1e-12


def test_model_metrics_are_recorded_and_plausible() -> None:
    """The trained metrics are present and within a sane range."""
    model = load_model()
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in model.metrics
        assert 0.0 <= model.metrics[key] <= 1.0
    # a useful model clears these floors on the PhiUSIIL data
    assert model.metrics["accuracy"] > 0.7
    assert model.metrics["roc_auc"] > 0.8
