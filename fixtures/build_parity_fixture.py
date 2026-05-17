"""Generate ``fixtures/parity_urls.json`` — the Python/JS parity fixture.

This script is the single producer of the parity fixture. It runs the *Python*
feature extractor and classifier over a hand-curated set of ~50 URLs (legit,
phishing, IP hosts, punycode, shorteners, ``@``-tricks, no-scheme, IDN,
query-heavy, edge cases) and records, for each URL, the exact expected feature
vector AND the expected final blended score / band / ML probability.

Both test suites then consume this one file:
- the pytest suite (``tests/test_parity.py``) re-derives the values in Python,
- the vitest suite (``web/src/parity.test.ts``) re-derives them in TypeScript.

If the Python and TypeScript implementations ever diverge, one of those suites
goes red — which is exactly the guarantee this project is built around.

Run ``python fixtures/build_parity_fixture.py`` to (re)generate the fixture.
The committed JSON is the source of truth; do not edit it by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from phishdetect.classify import classify  # noqa: E402  (after sys.path setup)
from phishdetect.features import FEATURE_NAMES, extract_features  # noqa: E402
from phishdetect.model import load_model  # noqa: E402

OUTPUT = _REPO_ROOT / "fixtures" / "parity_urls.json"

# A deliberately varied URL set. Each tuple is (url, short note for humans).
PARITY_URLS: list[tuple[str, str]] = [
    # --- Clearly legitimate -----------------------------------------------------
    ("https://github.com", "legit: bare HTTPS domain"),
    ("https://www.google.com/search?q=phishing", "legit: query param"),
    ("https://en.wikipedia.org/wiki/Phishing", "legit: one subdomain"),
    ("https://mail.google.com/mail/u/0/#inbox", "legit: subdomain + fragment"),
    ("https://stackoverflow.com/questions/12345/how-to", "legit: deep path"),
    ("https://www.amazon.co.uk/dp/B08N5WRWNW", "legit: multi-label eTLD"),
    ("https://news.ycombinator.com/item?id=1", "legit: query"),
    ("https://docs.python.org/3/library/urllib.parse.html", "legit: docs path"),
    ("http://example.com", "legit-ish: plain HTTP, no path"),
    ("https://api.github.com/repos/torvalds/linux", "legit: api subdomain"),
    ("https://www.cloudflare.com", "legit: www domain"),
    ("https://developer.mozilla.org/en-US/docs/Web", "legit: dev subdomain"),
    # --- No scheme (http:// should be prepended) -------------------------------
    ("example.com", "no-scheme: bare domain"),
    ("www.bbc.co.uk/news", "no-scheme: multi-label eTLD with path"),
    ("github.com/anthropics", "no-scheme: domain + path"),
    # --- IP-literal hosts ------------------------------------------------------
    ("http://192.168.0.1/admin", "phishing-ish: private IPv4"),
    ("http://203.0.113.42/login.php", "phishing: IPv4 + login"),
    ("https://[2001:db8::1]/path", "edge: IPv6 literal host"),
    ("http://8.8.8.8:8080/verify/account", "phishing: IPv4 with port"),
    # --- @-trick (real host is after the last @) ------------------------------
    ("http://www.paypal.com@malicious.example/login", "phishing: @-trick"),
    ("https://account.apple.com@198.51.100.7/verify", "phishing: @-trick to IP"),
    ("http://user:pass@evil-site.tk/confirm", "phishing: userinfo + bad TLD"),
    # --- Punycode / IDN homograph ---------------------------------------------
    ("http://xn--pple-43d.com/signin", "phishing: punycode (аpple)"),
    ("https://xn--80ak6aa92e.com", "edge: punycode host"),
    ("http://paурal.com/login", "phishing: Cyrillic homograph"),
    ("https://göögle.com", "edge: non-ASCII host"),
    # --- URL shorteners --------------------------------------------------------
    ("https://bit.ly/3xYzAbc", "shortener: bit.ly"),
    ("http://tinyurl.com/yckabcde", "shortener: tinyurl"),
    ("https://t.co/abcdEFGH", "shortener: t.co"),
    ("https://is.gd/abc123", "shortener: is.gd"),
    # --- Suspicious TLDs -------------------------------------------------------
    ("http://free-prize-winner.xyz", "phishing: .xyz TLD"),
    ("http://secure-update.zip/account", "phishing: .zip TLD"),
    ("https://login-verification.top/bank", "phishing: .top TLD"),
    ("http://account-confirm.tk", "phishing: .tk TLD"),
    # --- Keyword-stuffed / hyphen-stuffed phishing -----------------------------
    (
        "http://secure-login-update-account-verify.com/webscr",
        "phishing: keyword + hyphen stuffed",
    ),
    (
        "http://www.paypal.com.secure-login.confirm-account.evil.com/signin",
        "phishing: many subdomains burying brand",
    ),
    (
        "http://bankofamerica-online-banking-login.verify-account.gq/",
        "phishing: hyphens + bad TLD + keywords",
    ),
    ("http://appleid.apple.com.verify-account.ml/login", "phishing: subdomain spoof"),
    # --- Query-heavy -----------------------------------------------------------
    (
        "https://shop.example.com/cart?item=1&qty=2&ref=email&utm=spring&id=99",
        "legit-ish: many query params",
    ),
    (
        "http://track.example.click/r?u=login&t=verify&a=account&x=1",
        "phishing: query + bad TLD + keywords",
    ),
    # --- Double slash in path / open-redirect style ---------------------------
    ("http://example.com//redirect//login", "edge: double slash in path"),
    (
        "https://safe-looking-domain.com/redirect?next=http://evil.com",
        "phishing-ish: embedded redirect URL",
    ),
    # --- Digit-heavy host ------------------------------------------------------
    ("http://1234567890login.com/verify", "phishing: digit-heavy host"),
    ("http://0nline-b4nk1ng-2024.com/secure", "phishing: leetspeak digits"),
    # --- Long URL --------------------------------------------------------------
    (
        "http://example.com/" + "a" * 130 + "/login",
        "phishing-ish: very long URL",
    ),
    # --- Edge cases ------------------------------------------------------------
    ("https://example.com./trailing-dot", "edge: trailing dot in host"),
    ("HTTP://EXAMPLE.COM/UPPER", "edge: uppercase scheme + host"),
    ("https://example.com:443/path?a=b#frag", "edge: explicit port + frag"),
    ("ftp://files.example.org/pub", "edge: non-http scheme"),
    ("https://a.b.c.d.e.f.example.com/deep", "edge: very deep subdomains"),
    ("https://localhost:3000/dashboard", "edge: localhost dev URL"),
]


def main() -> int:
    """Build the parity fixture from the Python implementation. Returns 0."""
    model = load_model()
    entries: list[dict[str, object]] = []

    for url, note in PARITY_URLS:
        features = extract_features(url)
        result = classify(url, model=model)
        entries.append(
            {
                "url": url,
                "note": note,
                "features": {name: features[name] for name in FEATURE_NAMES},
                "ml_probability": round(result.ml_probability, 10),
                "heuristic_score": result.heuristic_score,
                "final_score": result.final_score,
                "band": result.band,
                "reason_ids": [hit.rule_id for hit in result.reasons],
            }
        )

    payload = {
        "_comment": (
            "AUTO-GENERATED by fixtures/build_parity_fixture.py. The single "
            "source of truth for Python<->TypeScript parity. Both the pytest "
            "suite and the vitest suite assert against this file. Do not edit "
            "by hand; regenerate with `python fixtures/build_parity_fixture.py`."
        ),
        "feature_names": list(FEATURE_NAMES),
        "model_metrics": model.metrics,
        "count": len(entries),
        "urls": entries,
    }

    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[parity] wrote {OUTPUT.relative_to(_REPO_ROOT)} with {len(entries)} URLs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
