"""Optional, CLI-only domain-age enrichment via WHOIS.

Newly registered domains are strongly over-represented among phishing sites, so
the *age* of a domain is a useful extra signal. Determining it, however,
requires a network WHOIS query — which the core detector deliberately avoids so
that the identical logic can run offline in the browser.

This module is therefore **entirely optional and CLI-only**:

- ``python-whois`` is imported lazily, inside the function, so the package
  installs and runs fine without the ``[whois]`` extra.
- Any failure (package missing, network down, registry rate-limit, unparyseable
  response) is swallowed and reported as a graceful "unavailable" result — it
  never raises and never blocks a verdict.

It does NOT feed into the model or the heuristic score; it is presented purely
as informational context next to the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .urlparse import parse_url


@dataclass(frozen=True)
class WhoisInfo:
    """The outcome of a best-effort WHOIS lookup.

    Attributes
    ----------
    available:
        ``True`` only if a creation date was successfully obtained.
    domain:
        The registrable domain that was queried.
    creation_date:
        The domain's registration date, if known.
    age_days:
        Whole days between ``creation_date`` and now, if known.
    registrar:
        The registrar name, if the WHOIS record exposed one.
    note:
        A short human-readable status (e.g. why the lookup was unavailable).
    """

    available: bool
    domain: str
    creation_date: datetime | None = None
    age_days: int | None = None
    registrar: str | None = None
    note: str = ""

    @property
    def is_newly_registered(self) -> bool:
        """True if the domain is known to be younger than 90 days."""
        return self.age_days is not None and self.age_days < 90


def _first_datetime(value: object) -> datetime | None:
    """Normalise a ``python-whois`` date field to a single ``datetime``.

    ``python-whois`` returns either a single ``datetime`` or a list of them
    (registries differ). Naive datetimes are assumed to be UTC.
    """
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def lookup_domain_age(url: str) -> WhoisInfo:
    """Best-effort WHOIS lookup for the registrable domain of ``url``.

    This function never raises. On any problem it returns a :class:`WhoisInfo`
    with ``available=False`` and an explanatory ``note``.

    Parameters
    ----------
    url:
        Any URL string; only its host is used.

    Returns
    -------
    WhoisInfo
        Domain-age information, or a graceful "unavailable" result.
    """
    parsed = parse_url(url)
    host = parsed.host
    if host == "":
        return WhoisInfo(False, "", note="No host to look up.")

    # An IP-literal host has no WHOIS domain record in the usual sense.
    if any(ch.isdigit() for ch in host) and all(c.isdigit() or c == "." for c in host):
        return WhoisInfo(False, host, note="Host is a raw IP address — no domain WHOIS.")

    try:
        import whois  # type: ignore  # lazy import — optional [whois] extra
    except ImportError:
        return WhoisInfo(
            False,
            host,
            note="WHOIS lookup needs the optional 'python-whois' package "
            "(install with the [whois] extra).",
        )

    try:
        record = whois.whois(host)  # network call
    except Exception as exc:  # noqa: BLE001 — degrade gracefully on ANY failure
        return WhoisInfo(False, host, note=f"WHOIS query failed: {exc}")

    creation = _first_datetime(getattr(record, "creation_date", None))
    if creation is None:
        return WhoisInfo(False, host, note="WHOIS record had no usable creation date.")

    now = datetime.now(UTC)
    age_days = max(0, (now - creation).days)
    registrar = getattr(record, "registrar", None)
    if isinstance(registrar, list):
        registrar = registrar[0] if registrar else None

    return WhoisInfo(
        available=True,
        domain=host,
        creation_date=creation,
        age_days=age_days,
        registrar=registrar if isinstance(registrar, str) else None,
        note="",
    )
