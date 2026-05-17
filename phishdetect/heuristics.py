"""Transparent heuristic rule engine.

The machine-learning model gives a single probability but cannot *explain*
itself in plain language. This module is the explainable half of the detector:
a fixed set of human-readable rules, each of which inspects the 20-feature
vector and, when it fires, contributes points and a sentence a non-expert can
understand ("Contains an '@' symbol — used to hide the real destination").

The points sum to a heuristic risk score clamped to 0-100. The rule set and
the exact point values are mirrored verbatim by `web/src/heuristics.ts`, so the
website shows the identical explanations.

Design notes
------------
- Rules operate ONLY on the feature dict — never on the raw URL — so the
  heuristic score is a deterministic function of the same 20 numbers the model
  sees. This keeps Python/JS parity trivial.
- Each rule is a small pure function returning ``None`` (did not fire) or a
  :class:`RuleHit`. Adding a rule is purely additive.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Maximum heuristic score. The raw point sum is clamped to this.
MAX_SCORE = 100


@dataclass(frozen=True)
class RuleHit:
    """A single triggered heuristic rule.

    Attributes
    ----------
    rule_id:
        Stable short identifier (used by tests and the UI; never shown raw).
    points:
        Risk points this rule contributes (positive integer).
    reason:
        Human-readable explanation suitable for display to a non-expert.
    severity:
        One of ``"low"`` / ``"medium"`` / ``"high"`` — drives the UI dot colour.
    """

    rule_id: str
    points: int
    reason: str
    severity: str


# A rule is a function feature-dict -> RuleHit | None.
Rule = Callable[[dict[str, float]], RuleHit | None]


def _rule_ip_host(f: dict[str, float]) -> RuleHit | None:
    """Fires when the host is a raw IP address instead of a domain name."""
    if f["has_ip_host"] >= 1.0:
        return RuleHit(
            "ip_host",
            30,
            "Uses a raw IP address instead of a domain name — legitimate sites "
            "almost always use a named domain.",
            "high",
        )
    return None


def _rule_at_symbol(f: dict[str, float]) -> RuleHit | None:
    """Fires when the URL contains an ``@`` (everything before it is ignored)."""
    if f["has_at_symbol"] >= 1.0:
        return RuleHit(
            "at_symbol",
            25,
            "Contains an '@' symbol — a browser ignores everything before it, "
            "so the real destination can be hidden.",
            "high",
        )
    return None


def _rule_punycode(f: dict[str, float]) -> RuleHit | None:
    """Fires on Punycode (``xn--``) labels — a classic homograph attack."""
    if f["has_punycode"] >= 1.0:
        return RuleHit(
            "punycode",
            22,
            "Host uses Punycode ('xn--') — often used to spoof a trusted brand "
            "with look-alike characters.",
            "high",
        )
    return None


def _rule_homograph(f: dict[str, float]) -> RuleHit | None:
    """Fires when the host contains non-ASCII (look-alike) characters."""
    if f["has_homograph"] >= 1.0 and f["has_punycode"] < 1.0:
        return RuleHit(
            "homograph",
            20,
            "Host contains non-ASCII characters that can visually imitate a "
            "trusted domain (homograph attack).",
            "high",
        )
    return None


def _rule_suspicious_tld(f: dict[str, float]) -> RuleHit | None:
    """Fires when the top-level domain is one heavily abused for phishing."""
    if f["suspicious_tld"] >= 1.0:
        return RuleHit(
            "suspicious_tld",
            18,
            "Uses a top-level domain frequently associated with abuse "
            "(e.g. .zip, .xyz, .top, .tk).",
            "medium",
        )
    return None


def _rule_shortener(f: dict[str, float]) -> RuleHit | None:
    """Fires for known URL shorteners, which hide the true destination."""
    if f["is_shortener"] >= 1.0:
        return RuleHit(
            "shortener",
            15,
            "This is a URL-shortening service — the real destination is hidden "
            "and cannot be inspected before clicking.",
            "medium",
        )
    return None


def _rule_no_https(f: dict[str, float]) -> RuleHit | None:
    """Fires when the URL is not served over HTTPS."""
    if f["is_https"] < 1.0:
        return RuleHit(
            "no_https",
            10,
            "Does not use HTTPS — traffic is not encrypted and the site's "
            "identity is not verified.",
            "low",
        )
    return None


def _rule_many_subdomains(f: dict[str, float]) -> RuleHit | None:
    """Fires when there are many subdomains (used to bury a brand name)."""
    if f["num_subdomains"] >= 3.0:
        return RuleHit(
            "many_subdomains",
            14,
            "Has an unusually deep chain of subdomains — phishing pages bury a "
            "trusted brand name inside a long host to look legitimate.",
            "medium",
        )
    return None


def _rule_suspicious_keywords(f: dict[str, float]) -> RuleHit | None:
    """Fires when several phishing trigger-words appear in the URL."""
    n = int(f["num_suspicious_keywords"])
    if n >= 2:
        return RuleHit(
            "suspicious_keywords",
            16,
            f"Contains {n} security/banking trigger words (login, verify, "
            "account, bank, …) — common bait in phishing links.",
            "medium",
        )
    if n == 1:
        return RuleHit(
            "suspicious_keyword_single",
            7,
            "Contains a security/banking trigger word (login, verify, account, "
            "…) — mildly suspicious on its own.",
            "low",
        )
    return None


def _rule_double_slash_path(f: dict[str, float]) -> RuleHit | None:
    """Fires on a ``//`` inside the path — sometimes used for open redirects."""
    if f["has_double_slash_in_path"] >= 1.0:
        return RuleHit(
            "double_slash_path",
            10,
            "Path contains '//' — sometimes used to smuggle a second URL or "
            "trigger an open redirect.",
            "low",
        )
    return None


def _rule_many_hyphens(f: dict[str, float]) -> RuleHit | None:
    """Fires when the hostname has many hyphens (e.g. ``secure-login-bank``)."""
    if f["num_hyphens"] >= 3.0:
        return RuleHit(
            "many_hyphens",
            12,
            "Hostname is stuffed with hyphens — a pattern used to assemble "
            "fake brand names like 'secure-login-yourbank-com'.",
            "medium",
        )
    return None


def _rule_long_url(f: dict[str, float]) -> RuleHit | None:
    """Fires for very long URLs, often used to hide a deceptive host."""
    if f["url_length"] >= 100.0:
        return RuleHit(
            "long_url",
            10,
            "The URL is very long — excessive length is often used to push the "
            "deceptive part out of view in the address bar.",
            "low",
        )
    return None


def _rule_digit_heavy_host(f: dict[str, float]) -> RuleHit | None:
    """Fires when the host is unusually digit-heavy (and not a plain IP)."""
    if f["has_ip_host"] < 1.0 and f["digit_ratio_host"] >= 0.30:
        return RuleHit(
            "digit_heavy_host",
            12,
            "The hostname is unusually full of digits — legitimate brand "
            "domains are rarely mostly numbers.",
            "medium",
        )
    return None


# The ordered rule set. Order only affects the display order of reasons.
RULES: list[Rule] = [
    _rule_ip_host,
    _rule_at_symbol,
    _rule_punycode,
    _rule_homograph,
    _rule_suspicious_tld,
    _rule_shortener,
    _rule_many_subdomains,
    _rule_suspicious_keywords,
    _rule_many_hyphens,
    _rule_digit_heavy_host,
    _rule_double_slash_path,
    _rule_long_url,
    _rule_no_https,
]


@dataclass(frozen=True)
class HeuristicResult:
    """The outcome of running every heuristic rule.

    Attributes
    ----------
    score:
        Integer risk score 0-100 (the clamped sum of fired rules' points).
    hits:
        The :class:`RuleHit` objects that fired, in rule order.
    """

    score: int
    hits: list[RuleHit]


def evaluate(features: dict[str, float]) -> HeuristicResult:
    """Run every heuristic rule against ``features`` and return the result.

    Parameters
    ----------
    features:
        A 20-feature dict as produced by :func:`phishdetect.features.extract_features`.

    Returns
    -------
    HeuristicResult
        ``score`` is the point sum clamped to ``[0, 100]``; ``hits`` lists every
        rule that fired with its human-readable reason.
    """
    hits: list[RuleHit] = []
    total = 0
    for rule in RULES:
        hit = rule(features)
        if hit is not None:
            hits.append(hit)
            total += hit.points
    score = max(0, min(MAX_SCORE, total))
    return HeuristicResult(score=score, hits=hits)
