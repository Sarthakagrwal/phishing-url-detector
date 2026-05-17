"""End-to-end tests for :func:`phishdetect.classify.classify`.

Verifies the structured result shape, the Safe/Suspicious/Dangerous band
thresholds, the blend arithmetic, and that obvious legitimate / phishing URLs
land in the expected band.
"""

from __future__ import annotations

from phishdetect.classify import (
    BAND_DANGEROUS_AT,
    BAND_SUSPICIOUS_AT,
    HEURISTIC_WEIGHT,
    ML_WEIGHT,
    classify,
    result_to_dict,
)
from phishdetect.features import FEATURE_NAMES


def test_result_has_full_structure() -> None:
    """The result carries every documented field with the right types."""
    result = classify("https://example.com/login")
    assert result.url == "https://example.com/login"
    assert 0.0 <= result.final_score <= 100.0
    assert result.band in {"Safe", "Suspicious", "Dangerous"}
    assert 0.0 <= result.ml_probability <= 1.0
    assert 0 <= result.heuristic_score <= 100
    assert list(result.features.keys()) == FEATURE_NAMES
    assert 0 < len(result.contributions) <= 6


def test_contributions_sorted_by_magnitude() -> None:
    """Top ML contributions are ordered by descending absolute value."""
    result = classify("http://secure-login.example.tk/verify/account")
    magnitudes = [abs(c.contribution) for c in result.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_blend_arithmetic_is_correct() -> None:
    """final_score equals the documented 60/40 ML/heuristic blend."""
    result = classify("http://login-verify.example.xyz/account")
    expected = (
        ML_WEIGHT * (result.ml_probability * 100.0) + HEURISTIC_WEIGHT * result.heuristic_score
    )
    assert abs(result.final_score - round(expected, 1)) < 0.05


def test_band_matches_thresholds() -> None:
    """The band is consistent with the score-to-band thresholds."""
    for url in [
        "https://github.com",
        "https://www.google.com/search?q=test",
        "http://192.168.0.1/login",
        "http://secure-login-verify.paypal-update.gq/webscr",
        "https://en.wikipedia.org/wiki/Security",
    ]:
        result = classify(url)
        if result.final_score >= BAND_DANGEROUS_AT:
            assert result.band == "Dangerous"
        elif result.final_score >= BAND_SUSPICIOUS_AT:
            assert result.band == "Suspicious"
        else:
            assert result.band == "Safe"


def test_known_good_url_is_safe() -> None:
    """An obvious legitimate HTTPS site is classified Safe."""
    assert classify("https://github.com").band == "Safe"


def test_blatant_phishing_url_is_dangerous() -> None:
    """A blatant phishing URL is classified Dangerous."""
    result = classify("http://secure-login-verify-account.paypal-update.gq/webscr/signin")
    assert result.band == "Dangerous"
    assert result.is_dangerous is True


def test_raw_ip_login_is_dangerous() -> None:
    """A login page hosted on a raw IP address is classified Dangerous."""
    assert classify("http://203.0.113.42/account/verify.php").band == "Dangerous"


def test_result_to_dict_is_json_serialisable() -> None:
    """result_to_dict produces a complete, plain-data structure."""
    import json

    payload = result_to_dict(classify("https://example.com/login"))
    # round-trips through JSON without error
    restored = json.loads(json.dumps(payload))
    assert restored["band"] in {"Safe", "Suspicious", "Dangerous"}
    assert len(restored["features"]) == 20
    assert "contributions" in restored
    assert "reasons" in restored


def test_scheme_less_url_is_handled() -> None:
    """A scheme-less URL classifies without error (http:// is synthesised)."""
    result = classify("example.com")
    assert result.url == "example.com"
    assert result.band in {"Safe", "Suspicious", "Dangerous"}


def test_classify_accepts_a_preloaded_model() -> None:
    """Passing an explicit model yields the same verdict as the default."""
    from phishdetect.model import load_model

    model = load_model()
    a = classify("http://login.example.xyz/verify", model=model)
    b = classify("http://login.example.xyz/verify")
    assert a.final_score == b.final_score
    assert a.band == b.band
