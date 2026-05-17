"""phishdetect — phishing-URL risk detection.

A small, dependency-free library that scores any URL for phishing risk by
combining a transparent heuristic rule engine with a logistic-regression
classifier. Every feature is derived from the URL *string* alone (no network,
no DNS, no page fetch), so the identical logic also runs fully in the browser.

Public surface
--------------
- :func:`phishdetect.classify.classify` — the end-to-end entry point.
- :mod:`phishdetect.features` — the 20-feature extractor.
- :mod:`phishdetect.heuristics` — the transparent rule engine.
- :mod:`phishdetect.model` — the closed-form logistic-regression predictor.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
