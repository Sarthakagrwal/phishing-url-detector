"""Tests for the ``phishdetect`` command-line interface (:mod:`phishdetect.cli`).

Exercises single-URL output, ``--json``, ``--batch`` and the error paths,
driving :func:`phishdetect.cli.main` directly and capturing stdout/stderr.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phishdetect.cli import main


def test_single_url_human_output(capsys: pytest.CaptureFixture[str]) -> None:
    """A single URL prints a human-readable verdict block."""
    code = main(["https://github.com", "--no-color"])
    out = capsys.readouterr().out
    assert code == 0
    assert "VERDICT" in out
    assert "https://github.com" in out
    assert "feature contributions" in out


def test_json_output_is_valid(capsys: pytest.CaptureFixture[str]) -> None:
    """--json emits a single valid JSON object with the expected keys."""
    code = main(["http://192.168.0.1/login/verify", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["band"] in {"Safe", "Suspicious", "Dangerous"}
    assert len(payload["features"]) == 20
    assert "ml_probability" in payload
    assert "reasons" in payload


def test_dangerous_url_is_flagged(capsys: pytest.CaptureFixture[str]) -> None:
    """A blatant phishing URL is reported as Dangerous."""
    code = main(["http://secure-login-verify.paypal-update.gq/webscr", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["band"] == "Dangerous"


def test_batch_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--batch classifies every non-comment line and emits a JSON array."""
    batch = tmp_path / "urls.txt"
    batch.write_text(
        "# a comment line\nhttps://github.com\n\nhttp://203.0.113.9/account/verify.php\n",
        encoding="utf-8",
    )
    code = main(["--batch", str(batch), "--json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert len(payload) == 2  # comment and blank line are skipped


def test_no_arguments_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Invoking with neither a URL nor --batch exits non-zero with a message."""
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0
    assert "URL" in capsys.readouterr().err


def test_missing_batch_file_is_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-existent --batch file exits with code 1 and an error message."""
    code = main(["--batch", "/nonexistent/path/urls.txt"])
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_batch_summary_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A multi-URL human-mode batch prints a summary count line."""
    batch = tmp_path / "urls.txt"
    batch.write_text(
        "https://github.com\nhttp://203.0.113.9/login/verify.php\n",
        encoding="utf-8",
    )
    code = main(["--batch", str(batch), "--no-color"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Summary:" in out
    assert "2 URLs" in out
