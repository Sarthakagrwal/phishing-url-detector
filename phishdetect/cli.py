"""Command-line interface for the phishing-URL detector.

Usage
-----
    phishdetect https://example.com           # single URL, coloured verdict
    phishdetect example.com --json            # machine-readable JSON
    phishdetect --batch urls.txt              # one URL per line
    phishdetect example.com --whois           # add optional WHOIS domain age

The CLI prints a coloured verdict banner, the blended score, the triggered
heuristic reasons and the top signed ML feature contributions. Colour is
emitted only when stdout is a TTY (and never with ``--json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .classify import ClassificationResult, classify, result_to_dict
from .model import LogisticModel, load_model

# --- ANSI colour helpers --------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_COLOURS = {
    "Safe": "\033[32m",  # green
    "Suspicious": "\033[33m",  # amber
    "Dangerous": "\033[31m",  # red
}


def _supports_colour(stream: object) -> bool:
    """Return True if ANSI colour should be emitted to ``stream``."""
    return bool(getattr(stream, "isatty", lambda: False)())


def _paint(text: str, colour: str, enabled: bool) -> str:
    """Wrap ``text`` in an ANSI ``colour`` code when ``enabled``."""
    if not enabled:
        return text
    return f"{colour}{text}{_RESET}"


def _band_bar(score: float, enabled: bool) -> str:
    """Render a 20-cell text meter for a 0-100 ``score``."""
    filled = int(round(score / 100.0 * 20))
    filled = max(0, min(20, filled))
    bar = "#" * filled + "-" * (20 - filled)
    return f"[{bar}]"


def _format_human(result: ClassificationResult, *, colour: bool, whois: bool) -> str:
    """Build the full coloured, human-readable report for one result."""
    band_colour = _COLOURS.get(result.band, "")
    lines: list[str] = []

    lines.append("")
    lines.append(_paint(f"  URL   {result.url}", _BOLD, colour))
    verdict = (
        f"  VERDICT  {result.band.upper()}   "
        f"score {result.final_score:.1f}/100   "
        f"{_band_bar(result.final_score, colour)}"
    )
    lines.append(_paint(verdict, band_colour + _BOLD, colour))
    lines.append(
        _paint(
            f"  model phishing probability: {result.ml_probability * 100:.1f}%   "
            f"heuristic score: {result.heuristic_score}/100",
            _DIM,
            colour,
        )
    )

    # Triggered heuristic reasons.
    lines.append("")
    if result.reasons:
        lines.append(_paint("  Why this score (heuristic signals):", _BOLD, colour))
        for hit in result.reasons:
            mark = {"high": "!!", "medium": " !", "low": "  "}.get(hit.severity, "  ")
            sev_colour = {
                "high": _COLOURS["Dangerous"],
                "medium": _COLOURS["Suspicious"],
                "low": _COLOURS["Safe"],
            }.get(hit.severity, "")
            lines.append(
                _paint(f"   {mark} ", sev_colour, colour) + f"(+{hit.points}) {hit.reason}"
            )
    else:
        lines.append(_paint("  No heuristic rules triggered.", _COLOURS["Safe"], colour))

    # Top ML contributions.
    lines.append("")
    lines.append(_paint("  Top model feature contributions:", _BOLD, colour))
    lines.append(_paint("   (+ pushes toward phishing, - pushes toward safe)", _DIM, colour))
    for contrib in result.contributions:
        sign = "+" if contrib.contribution >= 0 else "-"
        c_colour = _COLOURS["Dangerous"] if contrib.contribution >= 0 else _COLOURS["Safe"]
        bar_len = int(round(min(abs(contrib.contribution), 3.0) / 3.0 * 18))
        bar = "#" * bar_len
        lines.append(
            f"   {contrib.name:<24} value={contrib.value:<8.3g} "
            + _paint(f"{sign}{abs(contrib.contribution):.3f} {bar}", c_colour, colour)
        )

    # Optional WHOIS enrichment.
    if whois:
        lines.append("")
        lines.append(_paint("  Domain age (WHOIS — informational only):", _BOLD, colour))
        from .whois_enrich import lookup_domain_age

        info = lookup_domain_age(result.url)
        if info.available:
            created = (
                info.creation_date.date().isoformat()
                if info.creation_date is not None
                else "unknown"
            )
            age_note = f"{info.age_days} days old (registered {created})"
            if info.is_newly_registered:
                age_note += _paint("  -- recently registered!", _COLOURS["Dangerous"], colour)
            lines.append(f"   {age_note}")
            if info.registrar:
                lines.append(_paint(f"   registrar: {info.registrar}", _DIM, colour))
        else:
            lines.append(_paint(f"   {info.note}", _DIM, colour))

    lines.append("")
    return "\n".join(lines)


def _read_batch(path: Path) -> list[str]:
    """Read a batch file: one URL per line, ignoring blanks and ``#`` comments."""
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            urls.append(stripped)
    return urls


def _classify_all(urls: Sequence[str], model: LogisticModel) -> list[ClassificationResult]:
    """Classify a sequence of URLs against a single pre-loaded model."""
    return [classify(url, model=model) for url in urls]


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``phishdetect`` command."""
    parser = argparse.ArgumentParser(
        prog="phishdetect",
        description=(
            "Score a URL for phishing risk by blending transparent heuristic "
            "rules with a logistic-regression classifier. Analysis is based on "
            "the URL string alone — no page is fetched."
        ),
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="the URL to analyse (omit when using --batch)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of a coloured report",
    )
    parser.add_argument(
        "--batch",
        metavar="FILE",
        type=Path,
        help="classify every URL in FILE (one URL per line)",
    )
    parser.add_argument(
        "--whois",
        action="store_true",
        help="also show optional WHOIS domain-age info (needs network + [whois] extra)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable coloured output even on a terminal",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"phishdetect {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.

    Exit codes
    ----------
    0   ran successfully (a "Dangerous" verdict is still a successful run)
    1   bad usage (no URL and no ``--batch``) or a file/model error
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.url is None and args.batch is None:
        parser.error("provide a URL, or use --batch FILE")

    try:
        model = load_model()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Collect the URLs to classify.
    urls: list[str] = []
    if args.url is not None:
        urls.append(args.url)
    if args.batch is not None:
        if not args.batch.is_file():
            print(f"error: batch file not found: {args.batch}", file=sys.stderr)
            return 1
        urls.extend(_read_batch(args.batch))

    if not urls:
        print("error: no URLs to classify.", file=sys.stderr)
        return 1

    results = _classify_all(urls, model)

    if args.json:
        payload: object = (
            result_to_dict(results[0])
            if len(results) == 1 and args.batch is None
            else [result_to_dict(r) for r in results]
        )
        print(json.dumps(payload, indent=2))
        return 0

    colour = _supports_colour(sys.stdout) and not args.no_color
    for result in results:
        print(_format_human(result, colour=colour, whois=args.whois))

    # A brief summary line for batch runs.
    if len(results) > 1:
        dangerous = sum(1 for r in results if r.band == "Dangerous")
        suspicious = sum(1 for r in results if r.band == "Suspicious")
        safe = len(results) - dangerous - suspicious
        print(
            f"  Summary: {len(results)} URLs -> "
            f"{safe} safe, {suspicious} suspicious, {dangerous} dangerous.\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
