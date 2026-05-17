"""Tests for the 20-feature URL extractor (:mod:`phishdetect.features`).

Each of the 20 features is exercised on crafted URLs covering the cases a
phishing detector must get right: IP hosts, the ``@`` trick, Punycode, IDN
homographs, shorteners, scheme-less input, query-heavy URLs and odd TLDs.
"""

from __future__ import annotations

from phishdetect.features import (
    FEATURE_NAMES,
    extract_features,
    feature_vector,
    split_registrable_domain,
)


def test_feature_names_are_fixed_and_unique() -> None:
    """The 20 feature names are unique and the vector matches their order."""
    assert len(FEATURE_NAMES) == 20
    assert len(set(FEATURE_NAMES)) == 20
    feats = extract_features("https://example.com")
    assert list(feats.keys()) == FEATURE_NAMES
    assert feature_vector("https://example.com") == [feats[name] for name in FEATURE_NAMES]


def test_length_features() -> None:
    """url_length, hostname_length and path_length measure the right parts."""
    url = "https://example.com/path/to/page"
    f = extract_features(url)
    assert f["url_length"] == len(url)
    assert f["hostname_length"] == len("example.com")
    assert f["path_length"] == len("/path/to/page")


def test_url_length_uses_original_not_synthesised_scheme() -> None:
    """A scheme-less URL's length is measured before http:// is prepended."""
    assert extract_features("example.com")["url_length"] == len("example.com")


def test_num_dots_counts_whole_url_num_hyphens_counts_host() -> None:
    """num_dots spans the entire URL; num_hyphens is host-only."""
    f = extract_features("https://a-b.sub.example.com/x-y-z?p=1")
    assert f["num_dots"] == 3
    assert f["num_hyphens"] == 1  # only 'a-b' in the host


def test_num_subdomains() -> None:
    """num_subdomains counts host labels beyond the registrable domain."""
    assert extract_features("https://example.com")["num_subdomains"] == 0
    assert extract_features("https://www.example.com")["num_subdomains"] == 1
    assert extract_features("https://a.b.c.example.com")["num_subdomains"] == 3
    # multi-label eTLD: example.co.uk is the registrable domain
    assert extract_features("https://www.example.co.uk")["num_subdomains"] == 1


def test_has_ip_host_ipv4_and_ipv6() -> None:
    """has_ip_host detects IPv4 and bracketed IPv6 literals."""
    assert extract_features("http://192.168.1.1/x")["has_ip_host"] == 1.0
    assert extract_features("http://8.8.8.8")["has_ip_host"] == 1.0
    assert extract_features("https://[2001:db8::1]/x")["has_ip_host"] == 1.0
    assert extract_features("https://example.com")["has_ip_host"] == 0.0
    # out-of-range octet is not a valid IPv4
    assert extract_features("http://999.1.1.1")["has_ip_host"] == 0.0


def test_has_at_symbol() -> None:
    """has_at_symbol fires for an @ anywhere in the URL."""
    assert extract_features("http://safe.com@evil.com/x")["has_at_symbol"] == 1.0
    assert extract_features("https://example.com/x")["has_at_symbol"] == 0.0


def test_num_query_params() -> None:
    """num_query_params counts &-separated parameters."""
    assert extract_features("https://example.com")["num_query_params"] == 0
    assert extract_features("https://example.com?a=1")["num_query_params"] == 1
    assert extract_features("https://example.com?a=1&b=2&c=3")["num_query_params"] == 3


def test_has_punycode_and_has_homograph() -> None:
    """Punycode (xn--) and non-ASCII hosts are detected independently."""
    puny = extract_features("http://xn--pple-43d.com")
    assert puny["has_punycode"] == 1.0
    assert puny["has_homograph"] == 0.0  # xn-- labels are pure ASCII

    idn = extract_features("https://göögle.com")
    assert idn["has_homograph"] == 1.0
    assert idn["has_punycode"] == 0.0

    clean = extract_features("https://example.com")
    assert clean["has_punycode"] == 0.0
    assert clean["has_homograph"] == 0.0


def test_is_https() -> None:
    """is_https is 1 only for the https scheme."""
    assert extract_features("https://example.com")["is_https"] == 1.0
    assert extract_features("http://example.com")["is_https"] == 0.0
    # scheme-less input gets http:// -> not https
    assert extract_features("example.com")["is_https"] == 0.0


def test_digit_features() -> None:
    """num_digits_in_host and digit_ratio_host measure host digits."""
    f = extract_features("http://abc123def456.com")
    assert f["num_digits_in_host"] == 6
    assert f["digit_ratio_host"] == 6 / len("abc123def456.com")
    assert extract_features("https://example.com")["digit_ratio_host"] == 0.0


def test_suspicious_tld_and_shortener() -> None:
    """suspicious_tld and is_shortener flag known-bad TLDs / shorteners."""
    assert extract_features("http://free-prize.xyz")["suspicious_tld"] == 1.0
    assert extract_features("http://thing.zip/x")["suspicious_tld"] == 1.0
    assert extract_features("https://example.com")["suspicious_tld"] == 0.0
    assert extract_features("https://bit.ly/abc")["is_shortener"] == 1.0
    assert extract_features("https://tinyurl.com/abc")["is_shortener"] == 1.0
    assert extract_features("https://example.com")["is_shortener"] == 0.0


def test_num_suspicious_keywords() -> None:
    """num_suspicious_keywords counts phishing trigger words in the URL."""
    assert extract_features("http://x.com/login/verify/account")["num_suspicious_keywords"] == 3
    assert extract_features("https://example.com")["num_suspicious_keywords"] == 0


def test_has_double_slash_in_path() -> None:
    """has_double_slash_in_path fires for '//' inside the path."""
    assert extract_features("http://example.com//redirect")["has_double_slash_in_path"] == 1.0
    assert extract_features("http://example.com/redirect")["has_double_slash_in_path"] == 0.0


def test_num_special_chars_and_tld_length() -> None:
    """num_special_chars counts % = ? & _ ; tld_length measures the eTLD."""
    f = extract_features("https://example.com/x?a=b&c_d=e")
    assert f["num_special_chars"] == 5  # ? = & _ =
    assert extract_features("https://example.com")["tld_length"] == 3
    assert extract_features("https://example.co.uk")["tld_length"] == 5


def test_split_registrable_domain() -> None:
    """The eTLD split handles single- and multi-label public suffixes."""
    assert split_registrable_domain("www.example.com") == (
        "www",
        "example.com",
        "com",
    )
    assert split_registrable_domain("a.b.example.co.uk") == (
        "a.b",
        "example.co.uk",
        "co.uk",
    )
    assert split_registrable_domain("example.com") == ("", "example.com", "com")
    # an IP host has no registrable domain / eTLD
    assert split_registrable_domain("192.168.1.1") == ("", "192.168.1.1", "")


def test_no_scheme_url_is_parsed_consistently() -> None:
    """A scheme-less URL is parsed as if http:// had been prepended."""
    f = extract_features("www.bbc.co.uk/news")
    assert f["hostname_length"] == len("www.bbc.co.uk")
    assert f["is_https"] == 0.0
    assert f["path_length"] == len("/news")


def test_at_trick_host_is_after_last_at() -> None:
    """For an @-trick URL the host is the part after the LAST @."""
    f = extract_features("http://www.paypal.com@malicious.example/login")
    assert f["has_at_symbol"] == 1.0
    # the real host is malicious.example, not paypal.com
    assert f["hostname_length"] == len("malicious.example")


def test_all_features_are_numeric() -> None:
    """Every extracted feature value is an int/float (model-ready)."""
    for url in [
        "https://example.com",
        "http://192.168.1.1/login",
        "ftp://files.example.org/pub",
        "",
    ]:
        for value in extract_features(url).values():
            assert isinstance(value, (int, float))
