"""Tests for the transparent heuristic rule engine (:mod:`phishdetect.heuristics`).

Each rule is checked to fire exactly when expected, contribute the documented
points, and carry a sensible human-readable reason. The total score is verified
to be the clamped sum of fired rules.
"""

from __future__ import annotations

from phishdetect.features import extract_features
from phishdetect.heuristics import MAX_SCORE, evaluate


def _fired_ids(url: str) -> set[str]:
    """Return the set of heuristic rule ids that fired for ``url``."""
    return {hit.rule_id for hit in evaluate(extract_features(url)).hits}


def _hit(url: str, rule_id: str):
    """Return the :class:`RuleHit` for ``rule_id`` on ``url`` (or None)."""
    for hit in evaluate(extract_features(url)).hits:
        if hit.rule_id == rule_id:
            return hit
    return None


def test_ip_host_rule() -> None:
    """The raw-IP rule fires for an IP host and explains itself clearly."""
    assert "ip_host" in _fired_ids("http://203.0.113.5/x")
    hit = _hit("http://203.0.113.5/x", "ip_host")
    assert hit is not None
    assert hit.points == 30
    assert hit.severity == "high"
    assert "raw IP address" in hit.reason
    assert "ip_host" not in _fired_ids("https://example.com")


def test_at_symbol_rule() -> None:
    """The @-symbol rule fires when the URL contains an @."""
    assert "at_symbol" in _fired_ids("http://paypal.com@evil.example/x")
    hit = _hit("http://paypal.com@evil.example/x", "at_symbol")
    assert hit is not None and "'@'" in hit.reason
    assert "at_symbol" not in _fired_ids("https://example.com/x")


def test_punycode_and_homograph_are_mutually_exclusive() -> None:
    """An xn-- host triggers punycode (not homograph); an IDN host the reverse."""
    puny = _fired_ids("http://xn--pple-43d.com/x")
    assert "punycode" in puny
    assert "homograph" not in puny

    idn = _fired_ids("https://göögle.com")
    assert "homograph" in idn
    assert "punycode" not in idn


def test_suspicious_tld_rule() -> None:
    """The suspicious-TLD rule fires for abused TLDs only."""
    assert "suspicious_tld" in _fired_ids("http://thing.xyz")
    assert "suspicious_tld" in _fired_ids("http://thing.zip/x")
    assert "suspicious_tld" not in _fired_ids("https://example.com")


def test_shortener_rule() -> None:
    """The shortener rule fires for known URL-shortening hosts."""
    assert "shortener" in _fired_ids("https://bit.ly/abc")
    assert "shortener" in _fired_ids("http://tinyurl.com/abc")
    assert "shortener" not in _fired_ids("https://example.com")


def test_no_https_rule() -> None:
    """The no-HTTPS rule fires for plain http URLs."""
    assert "no_https" in _fired_ids("http://example.com")
    assert "no_https" not in _fired_ids("https://example.com")


def test_many_subdomains_rule() -> None:
    """The deep-subdomain rule fires only at 3+ subdomains."""
    assert "many_subdomains" in _fired_ids("https://a.b.c.example.com")
    assert "many_subdomains" not in _fired_ids("https://www.example.com")


def test_suspicious_keyword_rules() -> None:
    """One keyword triggers the single variant; two+ the multi variant."""
    single = _fired_ids("https://example.com/login")
    assert "suspicious_keyword_single" in single
    assert "suspicious_keywords" not in single

    multi = _fired_ids("https://example.com/login/verify/account")
    assert "suspicious_keywords" in multi
    assert "suspicious_keyword_single" not in multi


def test_many_hyphens_rule() -> None:
    """The hyphen-stuffing rule fires for 3+ hyphens in the host."""
    assert "many_hyphens" in _fired_ids("http://a-b-c-d.example.com")
    assert "many_hyphens" not in _fired_ids("https://a-b.example.com")


def test_digit_heavy_host_rule_excludes_plain_ip() -> None:
    """The digit-heavy rule fires for digit-stuffed hosts but not plain IPs."""
    assert "digit_heavy_host" in _fired_ids("http://1234567890ab.com")
    # a raw IP must not also be flagged as a digit-heavy host
    assert "digit_heavy_host" not in _fired_ids("http://192.168.1.1")


def test_double_slash_and_long_url_rules() -> None:
    """The double-slash and long-URL rules fire on their triggers."""
    assert "double_slash_path" in _fired_ids("http://example.com//x")
    long_url = "http://example.com/" + "a" * 120
    assert "long_url" in _fired_ids(long_url)


def test_clean_url_scores_zero() -> None:
    """A clean HTTPS URL with no tells scores zero with no reasons."""
    result = evaluate(extract_features("https://github.com"))
    assert result.score == 0
    assert result.hits == []


def test_score_is_clamped_sum_of_points() -> None:
    """The reported score equals the fired points, clamped to [0, 100]."""
    result = evaluate(
        extract_features("http://secure-login-verify-account.paypal-update.gq@198.51.100.9/webscr")
    )
    raw_sum = sum(hit.points for hit in result.hits)
    assert result.score == min(MAX_SCORE, raw_sum)
    assert 0 <= result.score <= MAX_SCORE
    assert result.score > 0  # this URL clearly trips several rules


def test_each_rule_has_a_nonempty_reason_and_valid_severity() -> None:
    """Every fired rule carries a non-empty reason and a known severity."""
    result = evaluate(extract_features("http://a-b-c-d.secure-login.gq@1.2.3.4/verify/account"))
    assert result.hits, "expected several rules to fire for this URL"
    for hit in result.hits:
        assert hit.reason.strip() != ""
        assert hit.severity in {"low", "medium", "high"}
        assert hit.points > 0
