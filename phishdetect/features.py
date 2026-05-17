"""URL feature extraction — 20 deterministic, string-only features.

Every feature is computed purely from the URL *string* (after the shared
deterministic parse in :mod:`phishdetect.urlparse`). No DNS, no HTTP, no
network of any kind. That is what lets the identical logic run client-side in
the browser (`web/src/features.ts` is a faithful port of this module).

The 20 features, in fixed order (see :data:`FEATURE_NAMES`)
-----------------------------------------------------------
=== ============================ ===================================================
#   name                         definition
=== ============================ ===================================================
1   url_length                   length of the input URL string (pre-scheme-synthesis)
2   hostname_length              length of the parsed host
3   path_length                  length of the path component (incl. leading ``/``)
4   num_dots                     count of ``.`` in the whole URL string
5   num_hyphens                  count of ``-`` in the hostname
6   num_subdomains               host labels in excess of the registrable domain
7   has_ip_host                  1 if the host is an IPv4/IPv6 literal, else 0
8   has_at_symbol                1 if the URL string contains ``@``, else 0
9   num_query_params             number of ``&``-separated query parameters
10  has_punycode                 1 if any host label starts ``xn--``, else 0
11  has_homograph                1 if the host has any non-ASCII character, else 0
12  is_https                     1 if the scheme is ``https``, else 0
13  num_digits_in_host           count of digits ``0-9`` in the host
14  digit_ratio_host             num_digits_in_host / hostname_length (0 if host empty)
15  suspicious_tld               1 if the eTLD is in :data:`SUSPICIOUS_TLDS`, else 0
16  is_shortener                 1 if the host is a known URL shortener, else 0
17  num_suspicious_keywords      count of phishing keywords found in the URL (lowercased)
18  has_double_slash_in_path     1 if ``//`` appears inside the path, else 0
19  num_special_chars            count of ``% = ? & _`` in the whole URL string
20  tld_length                   length of the eTLD string (e.g. ``co.uk`` -> 5)
=== ============================ ===================================================

The function :func:`extract_features` returns an ordered ``dict`` whose keys
are exactly :data:`FEATURE_NAMES`; :func:`feature_vector` returns the values as
a plain ``list[float]`` in that order (the form the model consumes).
"""

from __future__ import annotations

from .suffix_list import KNOWN_TLDS, MULTI_LABEL_SUFFIXES
from .urlparse import ParsedURL, parse_url

# --- Fixed, ordered feature list ------------------------------------------------
# The model's coefficients are aligned to THIS order. Never reorder; only append.
FEATURE_NAMES: list[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_subdomains",
    "has_ip_host",
    "has_at_symbol",
    "num_query_params",
    "has_punycode",
    "has_homograph",
    "is_https",
    "num_digits_in_host",
    "digit_ratio_host",
    "suspicious_tld",
    "is_shortener",
    "num_suspicious_keywords",
    "has_double_slash_in_path",
    "num_special_chars",
    "tld_length",
]

# Top-level domains disproportionately abused for phishing / malware. Mirrored
# verbatim in the TypeScript port.
SUSPICIOUS_TLDS: frozenset[str] = frozenset(
    {
        "zip",
        "mov",
        "xyz",
        "top",
        "tk",
        "gq",
        "ml",
        "cf",
        "ga",
        "country",
        "kim",
        "work",
        "click",
        "link",
        "gdn",
        "loan",
        "download",
        "review",
        "stream",
        "racing",
        "win",
        "bid",
        "party",
        "date",
        "men",
        "cricket",
        "science",
        "accountant",
        "trade",
    }
)

# Well-known URL-shortening services. A shortener hides the true destination,
# which is a phishing-relevant signal in its own right. Mirrored in the port.
URL_SHORTENERS: frozenset[str] = frozenset(
    {
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "cutt.ly",
        "rebrand.ly",
        "bit.do",
        "shorturl.at",
        "rb.gy",
        "tiny.cc",
        "t.ly",
        "soo.gd",
        "s.id",
        "lnkd.in",
        "db.tt",
        "qr.ae",
        "adf.ly",
        "tr.im",
        "v.gd",
    }
)

# Keywords commonly used in phishing URLs to imitate a trusted action/brand.
# Counted as the number of these substrings present in the lowercased URL.
SUSPICIOUS_KEYWORDS: list[str] = [
    "login",
    "secure",
    "account",
    "verify",
    "update",
    "bank",
    "confirm",
    "signin",
    "password",
    "webscr",
    "ebayisapi",
]


def _is_ipv4(host: str) -> bool:
    """Return True if ``host`` is a dotted-decimal IPv4 literal."""
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if part == "" or len(part) > 3:
            return False
        if not all("0" <= c <= "9" for c in part):
            return False
        if int(part) > 255:
            return False
    return True


def _is_ipv6(host: str) -> bool:
    """Return True if ``host`` is a bracketed IPv6 literal (``[...]``)."""
    if not (host.startswith("[") and host.endswith("]")):
        return False
    inner = host[1:-1]
    if inner == "":
        return False
    # Must look like hex groups separated by ':' (with optional '::' compression
    # and an optional embedded IPv4 tail). A light structural check is enough.
    if "::" in inner:
        if inner.count("::") > 1:
            return False
    allowed = "0123456789abcdefABCDEF:."
    return all(c in allowed for c in inner) and ":" in inner


def _is_ip_host(host: str) -> bool:
    """Return True if the host is an IPv4 or IPv6 literal."""
    return _is_ipv4(host) or _is_ipv6(host)


def _host_is_ascii(host: str) -> bool:
    """Return True if every character of the host is ASCII (code point < 128)."""
    return all(ord(c) < 128 for c in host)


def split_registrable_domain(host: str) -> tuple[str, str, str]:
    """Split a host into ``(subdomains, registrable_domain, etld)``.

    Uses the bundled :data:`MULTI_LABEL_SUFFIXES` / :data:`KNOWN_TLDS`. The
    registrable domain is the eTLD plus exactly one label to its left ("eTLD+1").

    Examples
    --------
    ``www.example.co.uk`` -> ``("www", "example.co.uk", "co.uk")``
    ``mail.google.com``   -> ``("mail", "google.com", "com")``
    ``example.com``       -> ``("", "example.com", "com")``

    For an IP-literal host or a single-label host there is no registrable
    domain; the eTLD is returned empty.
    """
    if host == "" or _is_ip_host(host):
        return "", host, ""

    labels = host.split(".")
    if len(labels) < 2:
        # A bare single label has no public suffix split.
        return "", host, ""

    # Prefer the longest matching multi-label suffix.
    etld = ""
    if len(labels) >= 3:
        two = ".".join(labels[-2:])
        if two in MULTI_LABEL_SUFFIXES:
            etld = two
    if etld == "":
        last = labels[-1]
        # Whether or not ``last`` is a recognised TLD, treat it as the eTLD:
        # unknown new gTLDs still behave as single-label suffixes.
        etld = last
        _ = KNOWN_TLDS  # referenced for documentation/validation parity

    etld_labels = etld.count(".") + 1
    if len(labels) <= etld_labels:
        # Host *is* the suffix (rare, e.g. literally "co.uk") — no domain part.
        return "", host, etld

    registrable = ".".join(labels[-(etld_labels + 1) :])
    subdomain_labels = labels[: len(labels) - (etld_labels + 1)]
    return ".".join(subdomain_labels), registrable, etld


def _count_query_params(query: str) -> int:
    """Return the number of parameters in a raw query string.

    Parameters are the non-empty pieces obtained by splitting on ``&`` (and
    also ``;``, a legacy separator). An empty query has zero parameters.
    """
    if query == "":
        return 0
    count = 0
    for chunk in query.replace(";", "&").split("&"):
        if chunk != "":
            count += 1
    return count


def extract_features_from_parsed(parsed: ParsedURL) -> dict[str, float]:
    """Compute the 20-feature dict from an already-:func:`parse_url` result."""
    url_str = parsed.original
    url_lower = url_str.lower()
    host = parsed.host

    host_labels = host.split(".") if host else []
    _subdomains, _registrable, etld = split_registrable_domain(host)
    num_subdomains = len(_subdomains.split(".")) if _subdomains != "" else 0

    digits_in_host = sum(1 for c in host if "0" <= c <= "9")
    hostname_length = len(host)
    digit_ratio = (digits_in_host / hostname_length) if hostname_length > 0 else 0.0

    has_punycode = 1.0 if any(lbl.startswith("xn--") for lbl in host_labels) else 0.0
    has_homograph = 0.0 if _host_is_ascii(host) else 1.0

    num_keywords = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    num_special = sum(1 for c in url_str if c in "%=?&_")

    features: dict[str, float] = {
        "url_length": float(len(url_str)),
        "hostname_length": float(hostname_length),
        "path_length": float(len(parsed.path)),
        "num_dots": float(url_str.count(".")),
        "num_hyphens": float(host.count("-")),
        "num_subdomains": float(num_subdomains),
        "has_ip_host": 1.0 if _is_ip_host(host) else 0.0,
        "has_at_symbol": 1.0 if "@" in url_str else 0.0,
        "num_query_params": float(_count_query_params(parsed.query)),
        "has_punycode": has_punycode,
        "has_homograph": has_homograph,
        "is_https": 1.0 if parsed.scheme == "https" else 0.0,
        "num_digits_in_host": float(digits_in_host),
        "digit_ratio_host": digit_ratio,
        "suspicious_tld": 1.0 if etld in SUSPICIOUS_TLDS else 0.0,
        "is_shortener": 1.0 if host in URL_SHORTENERS else 0.0,
        "num_suspicious_keywords": float(num_keywords),
        "has_double_slash_in_path": 1.0 if "//" in parsed.path else 0.0,
        "num_special_chars": float(num_special),
        "tld_length": float(len(etld)),
    }
    # Guard: dict order must equal FEATURE_NAMES order.
    assert list(features.keys()) == FEATURE_NAMES
    return features


def extract_features(url: str) -> dict[str, float]:
    """Extract the 20-feature dict from a raw URL string.

    The returned dict's keys are exactly :data:`FEATURE_NAMES`, in order.
    """
    return extract_features_from_parsed(parse_url(url))


def feature_vector(url: str) -> list[float]:
    """Return the 20 feature values as an ordered ``list[float]``.

    This is the exact form the logistic-regression model consumes.
    """
    feats = extract_features(url)
    return [feats[name] for name in FEATURE_NAMES]
