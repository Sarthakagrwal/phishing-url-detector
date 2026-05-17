"""Bundled public-suffix subset for the registrable-domain / TLD split.

The full Public Suffix List (publicsuffix.org) has thousands of entries and
changes constantly. This project bundles a small, frozen subset instead, for
three reasons:

1. **Parity.** The exact same bytes must be used by the Python extractor and
   by the TypeScript port. A frozen list checked into the repo guarantees that
   — there is no network fetch and no "which PSL version?" ambiguity.
2. **Determinism.** Feature extraction must be fully reproducible so that the
   committed parity fixtures stay valid.
3. **Scope.** The detector only needs a *good enough* eTLD split to count
   subdomains and measure the TLD. It does not need every obscure suffix.

The subset covers the common single-label TLDs plus the multi-label public
suffixes most often abused in phishing (``co.uk``, ``com.br`` …). Anything not
in the list falls back to "treat the final dot-label as the TLD", which is the
correct behaviour for the overwhelming majority of real URLs.

`MULTI_LABEL_SUFFIXES` and `KNOWN_TLDS` are the single source of truth; the
TypeScript port (`web/src/suffixList.ts`) mirrors these exact values, and a
parity test fails if they ever drift.
"""

from __future__ import annotations

# Multi-label public suffixes (e.g. a host ``foo.co.uk`` has registrable
# domain ``foo.co.uk`` and eTLD ``co.uk``). Kept deliberately small — only the
# widely used ones. Each entry is a dotted, lowercase suffix.
MULTI_LABEL_SUFFIXES: frozenset[str] = frozenset(
    {
        # United Kingdom
        "co.uk",
        "org.uk",
        "me.uk",
        "ltd.uk",
        "plc.uk",
        "net.uk",
        "sch.uk",
        "ac.uk",
        "gov.uk",
        # Australia
        "com.au",
        "net.au",
        "org.au",
        "edu.au",
        "gov.au",
        "id.au",
        # Brazil
        "com.br",
        "net.br",
        "org.br",
        "gov.br",
        # India
        "co.in",
        "net.in",
        "org.in",
        "gen.in",
        "firm.in",
        "ac.in",
        "edu.in",
        "gov.in",
        # New Zealand
        "co.nz",
        "net.nz",
        "org.nz",
        "govt.nz",
        "ac.nz",
        # South Africa
        "co.za",
        "org.za",
        "net.za",
        # Japan
        "co.jp",
        "or.jp",
        "ne.jp",
        "ac.jp",
        "go.jp",
        # Other commonly seen
        "com.cn",
        "net.cn",
        "org.cn",
        "gov.cn",
        "com.mx",
        "com.tr",
        "com.sg",
        "com.hk",
        "com.tw",
        "co.kr",
        "co.id",
        "co.th",
        "com.my",
        "com.ph",
        "com.ar",
        "com.co",
        "com.ua",
        "com.pl",
        "com.ru",
    }
)

# Single-label TLDs the detector recognises. Anything outside this set is still
# treated as a TLD (the final dot-label), so the list need not be exhaustive —
# it just helps validate hosts and is shared verbatim with the TS port.
KNOWN_TLDS: frozenset[str] = frozenset(
    {
        # Generic
        "com",
        "org",
        "net",
        "edu",
        "gov",
        "mil",
        "int",
        "info",
        "biz",
        "name",
        "pro",
        "mobi",
        "io",
        "co",
        "ai",
        "app",
        "dev",
        "tech",
        "online",
        "site",
        "store",
        "blog",
        "cloud",
        "shop",
        "live",
        "news",
        # Suspicious / heavily abused (also flagged by heuristics)
        "xyz",
        "top",
        "zip",
        "mov",
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
        # Country-code (single label)
        "us",
        "uk",
        "ca",
        "au",
        "de",
        "fr",
        "nl",
        "ru",
        "cn",
        "jp",
        "in",
        "br",
        "it",
        "es",
        "se",
        "no",
        "fi",
        "ch",
        "at",
        "be",
        "pl",
        "pt",
        "gr",
        "ie",
        "nz",
        "za",
        "mx",
        "ar",
        "kr",
        "sg",
        "hk",
        "tw",
        "id",
        "th",
        "my",
        "ph",
        "vn",
        "tr",
        "ua",
        "cz",
        "ro",
        "hu",
        "il",
        "eu",
    }
)
