"""A small, deterministic URL parser shared (in spirit) with the TS port.

Why not just use :mod:`urllib.parse` (Python) and the ``URL`` API (browser)?
Because the two diverge on exactly the inputs a phishing detector cares about:
hosts containing ``@``, missing schemes, IPv6 literals, trailing dots, and
percent-encoding. To get *byte-identical* feature vectors on both sides, this
module implements one explicit, well-documented parsing algorithm; the
TypeScript file ``web/src/urlparse.ts`` mirrors it step for step.

The algorithm (applied identically in both languages)
-----------------------------------------------------
1. Trim leading/trailing ASCII whitespace.
2. If the string has no ``scheme://`` prefix, prepend ``http://``. (A bare
   ``mailto:`` or ``javascript:`` style scheme without ``//`` is also given an
   ``http://`` prefix — the detector only reasons about web URLs.)
3. Split off the scheme (the text before ``://``), lowercased.
4. The remainder up to the first ``/``, ``?`` or ``#`` is the *authority*.
5. Within the authority, anything before the **last** ``@`` is userinfo; the
   rest is the host[:port]. (Browsers and ``urllib`` agree the last ``@``
   wins — attackers exploit this with ``safe.com@evil.com``.)
6. Split host from port on the last ``:`` that is not inside ``[...]`` (IPv6).
7. The host is lowercased and any single trailing dot is removed.
8. After the authority: path is up to ``?``/``#``; query is between ``?`` and
   ``#``; fragment is after ``#``.

The result is a :class:`ParsedURL`. No network access is ever performed.
"""

from __future__ import annotations

from dataclasses import dataclass

# Characters that terminate the authority component.
_AUTHORITY_TERMINATORS = ("/", "?", "#")


@dataclass(frozen=True)
class ParsedURL:
    """The deterministic decomposition of a URL string.

    Attributes
    ----------
    original:
        The input string after whitespace trimming, *before* a scheme was
        synthesised. Feature extraction measures lengths against this so that
        adding ``http://`` to a scheme-less URL does not inflate ``url_length``.
    href:
        The full URL actually parsed (i.e. with a synthesised scheme if one was
        added).
    scheme:
        Lowercased scheme without ``://`` (e.g. ``"https"``).
    userinfo:
        Text before the last ``@`` in the authority (``""`` if absent).
    host:
        Lowercased host with any trailing dot removed. For IPv6 the surrounding
        brackets are kept (``"[::1]"``).
    port:
        Port string if present, else ``""``.
    path:
        Path component including its leading ``/`` (``""`` if absent).
    query:
        Raw query string without the leading ``?`` (``""`` if absent).
    fragment:
        Raw fragment without the leading ``#`` (``""`` if absent).
    had_scheme:
        ``True`` if the input already carried a ``scheme://`` prefix.
    """

    original: str
    href: str
    scheme: str
    userinfo: str
    host: str
    port: str
    path: str
    query: str
    fragment: str
    had_scheme: bool


def _has_scheme(text: str) -> bool:
    """Return True if ``text`` starts with a ``scheme://`` prefix.

    A scheme is ``ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )`` immediately
    followed by ``://`` (RFC 3986 §3.1, restricted to the hierarchical form).
    """
    idx = text.find("://")
    if idx <= 0:
        return False
    scheme = text[:idx]
    first = scheme[0]
    if not (("a" <= first <= "z") or ("A" <= first <= "Z")):
        return False
    for ch in scheme:
        ok = ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch in "+-."
        if not ok:
            return False
    return True


def _split_authority_port(host_port: str) -> tuple[str, str]:
    """Split a ``host[:port]`` string into ``(host, port)``.

    The split is on the last ``:`` that is not inside an IPv6 ``[...]`` literal.
    If the colon is not followed by an all-digit port it is treated as part of
    the host (so a malformed authority does not silently lose characters).
    """
    if host_port.startswith("["):
        close = host_port.find("]")
        if close != -1:
            host = host_port[: close + 1]
            rest = host_port[close + 1 :]
            if rest.startswith(":"):
                return host, rest[1:]
            return host, ""
    colon = host_port.rfind(":")
    if colon == -1:
        return host_port, ""
    candidate_port = host_port[colon + 1 :]
    if candidate_port != "" and all("0" <= c <= "9" for c in candidate_port):
        return host_port[:colon], candidate_port
    return host_port, ""


def parse_url(raw: str) -> ParsedURL:
    """Parse ``raw`` into a :class:`ParsedURL` using the shared algorithm.

    URLs with no ``scheme://`` prefix have ``http://`` prepended before
    parsing, exactly as the TypeScript port does, so the two implementations
    always see the same authority/path split.
    """
    original = raw.strip()

    had_scheme = _has_scheme(original)
    href = original if had_scheme else "http://" + original

    # 1. Scheme.
    scheme_end = href.find("://")
    scheme = href[:scheme_end].lower()
    after_scheme = href[scheme_end + 3 :]

    # 2. Authority = up to the first terminator.
    authority_end = len(after_scheme)
    for term in _AUTHORITY_TERMINATORS:
        pos = after_scheme.find(term)
        if pos != -1 and pos < authority_end:
            authority_end = pos
    authority = after_scheme[:authority_end]
    rest = after_scheme[authority_end:]

    # 3. Userinfo splits on the LAST '@'.
    at = authority.rfind("@")
    if at != -1:
        userinfo = authority[:at]
        host_port = authority[at + 1 :]
    else:
        userinfo = ""
        host_port = authority

    # 4. Host / port.
    host, port = _split_authority_port(host_port)
    host = host.lower()
    if host.endswith(".") and not host.endswith("]"):
        host = host[:-1]

    # 5. Path / query / fragment.
    query = ""
    fragment = ""
    path = rest

    hash_pos = path.find("#")
    if hash_pos != -1:
        fragment = path[hash_pos + 1 :]
        path = path[:hash_pos]

    q_pos = path.find("?")
    if q_pos != -1:
        query = path[q_pos + 1 :]
        path = path[:q_pos]

    return ParsedURL(
        original=original,
        href=href,
        scheme=scheme,
        userinfo=userinfo,
        host=host,
        port=port,
        path=path,
        query=query,
        fragment=fragment,
        had_scheme=had_scheme,
    )
