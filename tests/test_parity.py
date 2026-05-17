"""Parity tests against the shared fixture ``fixtures/parity_urls.json``.

The fixture stores, for ~50 diverse URLs, the expected feature vector AND the
expected blended score / band that the Python implementation produces. This
suite re-derives those values in Python and asserts they still match — so if a
change to the extractor or model silently shifts behaviour, it is caught here.

The vitest suite (`web/src/*.test.ts`) consumes the *same* file: the two
together guarantee Python and TypeScript stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phishdetect.classify import classify
from phishdetect.features import FEATURE_NAMES, extract_features
from phishdetect.model import load_model

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "parity_urls.json"


def _load_fixture() -> dict:
    """Load and return the parsed parity fixture."""
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


_FIXTURE = _load_fixture()
_ENTRIES = _FIXTURE["urls"]
_IDS = [f"{e['note']} :: {e['url'][:48]}" for e in _ENTRIES]


def test_fixture_exists_and_is_populated() -> None:
    """The fixture file exists and contains a meaningful number of URLs."""
    assert _FIXTURE_PATH.is_file()
    assert _FIXTURE["count"] == len(_ENTRIES)
    assert len(_ENTRIES) >= 50
    assert _FIXTURE["feature_names"] == FEATURE_NAMES


@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_feature_vector_matches_fixture(entry: dict) -> None:
    """Each URL's extracted feature vector matches the stored expectation."""
    got = extract_features(entry["url"])
    for name in FEATURE_NAMES:
        assert got[name] == pytest.approx(entry["features"][name], abs=1e-9), (
            f"{entry['url']} :: feature {name}"
        )


@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_classification_matches_fixture(entry: dict) -> None:
    """Each URL's ML probability, score, band and reasons match the fixture."""
    model = load_model()
    result = classify(entry["url"], model=model)
    assert result.ml_probability == pytest.approx(entry["ml_probability"], abs=1e-9), (
        f"{entry['url']} :: ml_probability"
    )
    assert result.heuristic_score == entry["heuristic_score"], f"{entry['url']} :: heuristic_score"
    assert result.final_score == pytest.approx(entry["final_score"], abs=1e-6), (
        f"{entry['url']} :: final_score"
    )
    assert result.band == entry["band"], f"{entry['url']} :: band"
    assert [h.rule_id for h in result.reasons] == entry["reason_ids"], (
        f"{entry['url']} :: reason_ids"
    )


def test_fixture_covers_the_required_url_categories() -> None:
    """The fixture spans the categories the brief requires (IP, punycode, …)."""
    urls = [e["url"] for e in _ENTRIES]
    joined = " ".join(urls)
    # IP-host URL present
    assert any(extract_features(u)["has_ip_host"] == 1.0 for u in urls)
    # punycode present
    assert "xn--" in joined
    # shortener present
    assert any(extract_features(u)["is_shortener"] == 1.0 for u in urls)
    # @-trick present
    assert any("@" in u for u in urls)
    # scheme-less URL present
    assert any("://" not in u for u in urls)
    # both bands of verdict present
    bands = {e["band"] for e in _ENTRIES}
    assert "Safe" in bands and "Dangerous" in bands
