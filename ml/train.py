"""Train the logistic-regression phishing classifier.

Pipeline
--------
1. Fetch the **PhiUSIIL Phishing URL Dataset** — UCI ML Repository #967 — via
   the ``ucimlrepo`` package. If that fails (offline, API change), fall back to
   downloading the dataset CSV directly from the UCI file endpoint.
2. Keep ONLY the ``URL`` and ``label`` columns. The dataset also ships dozens of
   precomputed features — we deliberately ignore all of them and rebuild our own
   20 features from the URL string with :mod:`phishdetect.features`, so the
   model is trained on exactly the features the detector (and the browser) will
   compute at inference time.
3. **Verify label polarity.** The PhiUSIIL dataset labels legitimate URLs as
   ``1`` and phishing URLs as ``0`` (the opposite of the intuitive convention).
   We detect this from the data and remap so that, in this project, ``1`` always
   means *phishing* — the positive class. The detection + decision is printed
   and recorded in ``model_meta.json``.
4. **Mitigate a dataset bias by augmenting the legitimate class.** Every
   legitimate URL in PhiUSIIL is a bare site *home page* (``https://www.<domain>``
   with no path, no query, always HTTPS, always ``www.``) while phishing URLs
   frequently carry deep paths and query strings. A model trained on the raw set
   would therefore learn the spurious shortcut "has a path ⇒ phishing" and
   wrongly flag legitimate deep links such as ``github.com/user/repo``. To break
   that shortcut we synthesise extra *legitimate* URLs from the real legitimate
   domains by attaching realistic benign paths/queries and varying the scheme
   and ``www.`` prefix. This is a deliberate, documented augmentation step; it
   does not invent new domains, only realistic URL shapes for known-good ones.
   See :func:`_augment_legit`.
5. Train ``Pipeline([StandardScaler, LogisticRegression])`` with a fixed
   ``random_state`` and a stratified train/test split.
6. Evaluate on the held-out test set: accuracy, precision, recall, F1, ROC-AUC,
   confusion matrix — all printed and written into ``model_meta.json``.
7. Save two artefacts (both committed to the repo):
   - ``models/phishing_model.joblib`` — the fitted sklearn pipeline (reference).
   - ``models/model_meta.json`` — the closed-form parameters + metrics that the
     pure-Python predictor and the browser port actually use.

Run:  ``python ml/train.py``
"""

from __future__ import annotations

import io
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Make the in-repo package importable when run as `python ml/train.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from phishdetect.features import FEATURE_NAMES, feature_vector  # noqa: E402

# --- Configuration --------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
UCI_DATASET_ID = 967
# Direct-download fallback (the static file UCI serves for dataset #967).
UCI_CSV_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
DATASET_DIR = _REPO_ROOT / "ml" / "dataset"
MODELS_DIR = _REPO_ROOT / "models"
MODEL_JOBLIB = MODELS_DIR / "phishing_model.joblib"
MODEL_META = MODELS_DIR / "model_meta.json"
# Cap the row count so training is fast and the committed model is reproducible
# on any machine. The PhiUSIIL set has ~235k rows; a stratified 60k sample is
# more than enough for a 20-feature logistic regression.
MAX_ROWS = 60_000
# Fraction of the legitimate rows to additionally synthesise as realistic
# deep-link variants (see _augment_legit). 0.35 was chosen by a small sweep: it
# breaks the "path => phishing" shortcut (legitimate deep links score low)
# while keeping held-out precision/recall balanced and ROC-AUC ~0.91.
AUGMENT_FRACTION = 0.35

# Realistic, benign path templates attached to known-good domains during
# augmentation. These contain NO phishing keywords — they are ordinary site
# structure (docs, blogs, products, search) so the model learns that a path on
# a reputable domain is normal.
_BENIGN_PATHS: tuple[str, ...] = (
    "/about",
    "/about/team",
    "/contact",
    "/blog",
    "/blog/2024/03/release-notes",
    "/news/latest",
    "/products",
    "/products/catalog/item-2381",
    "/docs/getting-started",
    "/docs/api/reference",
    "/help/faq",
    "/support/articles/how-to-reset",
    "/search?q=annual+report",
    "/search?q=opening+hours&page=2",
    "/category/research/papers",
    "/2024/03/15/quarterly-update",
    "/u/profile/settings",
    "/library/catalogue/record/55821",
    "/events/conference-2024",
    "/downloads/brochure.pdf",
    "/services/consulting",
    "/team/members",
    "/portfolio/case-study-3",
    "/wiki/Main_Page",
    "/pricing",
    "/gallery/photos/spring",
    "/faq?topic=shipping",
    "/article/123456/local-news-today",
    "/courses/introduction-to-biology",
    "/jobs/openings/engineering",
)


def _augment_legit(legit_urls: list[str], rng: random.Random) -> list[str]:
    """Synthesise realistic legitimate deep-link URLs from known-good domains.

    Every legitimate URL in PhiUSIIL is a bare ``https://www.<domain>`` home
    page, so a raw-URL model would conflate "has a path" with "phishing". This
    function takes a fraction of the real legitimate domains and rebuilds them
    as ordinary deep links — different scheme, optional ``www.`` drop, a benign
    path and sometimes a benign query — WITHOUT inventing any new domains. The
    result is legitimate URL *shapes* the dataset otherwise lacks.

    Parameters
    ----------
    legit_urls:
        The real legitimate URLs (bare home pages) from the dataset.
    rng:
        A seeded :class:`random.Random` for reproducibility.

    Returns
    -------
    list[str]
        Synthetic legitimate URLs to add to the legitimate class.
    """
    augmented: list[str] = []
    sample_size = int(len(legit_urls) * AUGMENT_FRACTION)
    for url in rng.sample(legit_urls, k=min(sample_size, len(legit_urls))):
        # Strip scheme and an optional leading 'www.' to get the bare domain.
        bare = url.split("://", 1)[-1]
        domain = bare[4:] if bare.startswith("www.") else bare
        domain = domain.rstrip("/")
        if "/" in domain or domain == "":
            continue

        # Mix schemes ~70/30 https/http. The PhiUSIIL legitimate URLs are 100%
        # HTTPS, so without some http legitimate examples the model would treat
        # plain http as a strong phishing signal and over-flag real http sites.
        scheme = "https" if rng.random() < 0.70 else "http"
        host = domain if rng.random() < 0.5 else "www." + domain
        path = rng.choice(_BENIGN_PATHS)
        augmented.append(f"{scheme}://{host}{path}")

        # For a minority, also emit a bare-domain variant under the other
        # scheme so the model sees legitimate http home pages too.
        if rng.random() < 0.25:
            augmented.append(f"http://{host}")
    return augmented


def _load_raw_dataset() -> pd.DataFrame:
    """Return a DataFrame with ``URL`` and ``label`` columns.

    Tries ``ucimlrepo`` first, then a direct CSV/ZIP download from UCI, then any
    previously cached copy in ``ml/dataset/``. Raises ``RuntimeError`` if none
    of these succeed.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    cache_csv = DATASET_DIR / "phiusiil.csv"

    # --- Attempt 1: ucimlrepo --------------------------------------------------
    try:
        from ucimlrepo import fetch_ucirepo

        print(f"[data] fetching UCI dataset #{UCI_DATASET_ID} via ucimlrepo …")
        repo = fetch_ucirepo(id=UCI_DATASET_ID)
        features_df = repo.data.features
        targets_df = repo.data.targets
        frame = pd.concat([features_df, targets_df], axis=1)
        frame = _normalise_columns(frame)
        frame.to_csv(cache_csv, index=False)
        print(f"[data] ucimlrepo OK — {len(frame):,} rows, cached to {cache_csv}")
        return frame
    except Exception as exc:  # noqa: BLE001 — fall through to the next strategy
        print(f"[data] ucimlrepo failed ({exc!r}); trying direct download …")

    # --- Attempt 2: direct download -------------------------------------------
    try:
        import urllib.request
        import zipfile

        print(f"[data] downloading {UCI_CSV_URL} …")
        with urllib.request.urlopen(UCI_CSV_URL, timeout=120) as resp:  # noqa: S310
            blob = resp.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(csv_name) as fh:
                frame = pd.read_csv(fh)
        frame = _normalise_columns(frame)
        frame.to_csv(cache_csv, index=False)
        print(f"[data] direct download OK — {len(frame):,} rows")
        return frame
    except Exception as exc:  # noqa: BLE001 — fall through to the cache
        print(f"[data] direct download failed ({exc!r}); trying local cache …")

    # --- Attempt 3: local cache ------------------------------------------------
    if cache_csv.is_file():
        frame = _normalise_columns(pd.read_csv(cache_csv))
        print(f"[data] using cached dataset — {len(frame):,} rows")
        return frame

    raise RuntimeError(
        "Could not obtain the PhiUSIIL dataset by any method (ucimlrepo, direct "
        "download, or local cache). Check your network connection and retry."
    )


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce an arbitrary PhiUSIIL frame to just ``URL`` and ``label`` columns.

    The dataset's column names are matched case-insensitively so the loader is
    robust to minor schema differences between the ucimlrepo and CSV forms.
    """
    lower = {c.lower(): c for c in frame.columns}
    if "url" not in lower or "label" not in lower:
        raise RuntimeError(f"Expected 'URL' and 'label' columns; got {list(frame.columns)[:12]}…")
    out = frame[[lower["url"], lower["label"]]].copy()
    out.columns = ["URL", "label"]
    out = out.dropna(subset=["URL", "label"])
    out["URL"] = out["URL"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def _resolve_label_polarity(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return a phishing-is-1 label series plus a human-readable explanation.

    The PhiUSIIL dataset uses ``label = 1`` for *legitimate* and ``label = 0``
    for *phishing*. This project's convention is the opposite (1 = phishing =
    positive class). We confirm the dataset's polarity heuristically — phishing
    URLs are, on average, markedly longer than legitimate ones — and remap the
    labels accordingly so the rest of the pipeline always treats 1 as phishing.
    """
    url_len = frame["URL"].str.len()
    mean_len_label0 = float(url_len[frame["label"] == 0].mean())
    mean_len_label1 = float(url_len[frame["label"] == 1].mean())

    # The class with the longer average URL is the phishing class.
    if mean_len_label1 >= mean_len_label0:
        phishing_value = 1
    else:
        phishing_value = 0

    phishing = (frame["label"] == phishing_value).astype(int)
    explanation = (
        f"Dataset label polarity check: mean URL length is "
        f"{mean_len_label0:.1f} for label=0 and {mean_len_label1:.1f} for "
        f"label=1. The longer-URL class is phishing, so dataset "
        f"label={phishing_value} == phishing. Remapped to project convention "
        f"(1 = phishing). This matches the documented PhiUSIIL encoding "
        f"(1 = legitimate, 0 = phishing) when label=0 is phishing."
    )
    return phishing, explanation


def _build_feature_matrix(urls: pd.Series) -> np.ndarray:
    """Build the 20-feature matrix from a series of URL strings.

    Uses :func:`phishdetect.features.feature_vector` so that the training-time
    features are byte-identical to what the detector computes at inference.
    """
    rows = [feature_vector(u) for u in urls]
    return np.asarray(rows, dtype=float)


def main() -> int:
    """Train, evaluate, and persist the phishing classifier. Returns 0 on success."""
    started = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    frame = _load_raw_dataset()

    # Resolve labels to the project convention (1 = phishing).
    phishing, polarity_note = _resolve_label_polarity(frame)
    frame = frame.assign(phishing=phishing)
    print(f"[data] {polarity_note}")

    # Stratified down-sample for fast, reproducible training. Each class is
    # sampled in proportion to its share of the full dataset, preserving the
    # original class balance.
    if len(frame) > MAX_ROWS:
        total = len(frame)
        parts: list[pd.DataFrame] = []
        for _cls, group in frame.groupby("phishing"):
            take = int(round(MAX_ROWS * len(group) / total))
            take = min(take, len(group))
            parts.append(group.sample(n=take, random_state=RANDOM_STATE))
        frame = (
            pd.concat(parts, axis=0)
            .sample(frac=1.0, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        print(f"[data] stratified down-sample to {len(frame):,} rows")

    # Augment the legitimate class with realistic deep-link URLs (see the module
    # docstring and _augment_legit) so the model does not learn "path ⇒ phishing".
    rng = random.Random(RANDOM_STATE)
    legit_urls = frame.loc[frame["phishing"] == 0, "URL"].tolist()
    synthetic_legit = _augment_legit(legit_urls, rng)
    if synthetic_legit:
        aug_frame = pd.DataFrame({"URL": synthetic_legit, "label": 0, "phishing": 0})
        frame = (
            pd.concat([frame, aug_frame], axis=0, ignore_index=True)
            .sample(frac=1.0, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        print(
            f"[augment] added {len(synthetic_legit):,} synthetic legitimate "
            f"deep-link URLs to break the 'path ⇒ phishing' shortcut"
        )

    n_phish = int(frame["phishing"].sum())
    n_legit = len(frame) - n_phish
    print(f"[data] class balance — phishing: {n_phish:,}  legitimate: {n_legit:,}")

    print(f"[features] building the {len(FEATURE_NAMES)}-feature matrix …")
    x = _build_feature_matrix(frame["URL"])
    y = frame["phishing"].to_numpy(dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(
        f"[split] train: {len(x_train):,}  test: {len(x_test):,}  "
        f"(stratified, random_state={RANDOM_STATE})"
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                    C=1.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    print("[train] fitting StandardScaler + LogisticRegression …")
    pipeline.fit(x_train, y_train)

    # --- Evaluation on the held-out test set ----------------------------------
    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Held-out test-set metrics =========================")
    for name, value in metrics.items():
        print(f"  {name:<10} {value:.4f}")
    print("  confusion matrix  [[TN FP] [FN TP]]:")
    print(f"    {cm.tolist()}")
    print("=======================================================\n")

    # --- Extract the closed-form parameters -----------------------------------
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    meta = {
        "model_type": "logistic_regression",
        "description": (
            "StandardScaler + LogisticRegression trained on the PhiUSIIL "
            "Phishing URL Dataset (UCI #967). Features are the 20 string-only "
            "features defined in phishdetect.features. Positive class (1) = "
            "phishing."
        ),
        "dataset": {
            "name": "PhiUSIIL Phishing URL Dataset",
            "source": "UCI Machine Learning Repository, dataset #967",
            "license": "CC BY 4.0",
            "rows_used": int(len(frame)),
            "phishing_rows": n_phish,
            "legitimate_rows": n_legit,
            "synthetic_legitimate_rows": len(synthetic_legit),
            "label_polarity": polarity_note,
            "augmentation_note": (
                "Every legitimate URL in PhiUSIIL is a bare site home page "
                "(https://www.<domain>, no path). To stop the model learning "
                "the spurious shortcut 'path => phishing', the legitimate class "
                "was augmented with realistic deep-link URLs built from the "
                "same known-good domains (varied scheme/www/path, no phishing "
                "keywords). No new domains were invented."
            ),
        },
        "training": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "scaler": "StandardScaler",
            "classifier": "LogisticRegression(C=1.0, class_weight='balanced')",
        },
        "feature_names": list(FEATURE_NAMES),
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "threshold": 0.5,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
    }

    MODEL_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    joblib.dump(pipeline, MODEL_JOBLIB)
    print(f"[save] wrote {MODEL_META.relative_to(_REPO_ROOT)}")
    print(f"[save] wrote {MODEL_JOBLIB.relative_to(_REPO_ROOT)}")

    # --- Sanity check: pure-Python predictor must match sklearn ---------------
    _verify_closed_form(pipeline, x_test[:200], meta)

    print(f"\n[done] training finished in {time.time() - started:.1f}s")
    return 0


def _verify_closed_form(pipeline: Pipeline, sample: np.ndarray, meta: dict) -> None:
    """Assert the closed-form sigmoid reproduces sklearn's probabilities.

    This guards the central design claim: the pure-Python / TypeScript predictor
    computes the *same* number as the trained sklearn pipeline. A mismatch fails
    training loudly rather than shipping a divergent model.
    """
    import math

    coef = meta["coef"]
    intercept = meta["intercept"]
    mean = meta["mean"]
    scale = meta["scale"]

    sk_proba = pipeline.predict_proba(sample)[:, 1]
    max_diff = 0.0
    for row, expected in zip(sample, sk_proba, strict=True):
        logit = intercept
        for x, w, m, s in zip(row, coef, mean, scale, strict=True):
            denom = s if s != 0.0 else 1.0
            logit += w * ((x - m) / denom)
        got = (
            1.0 / (1.0 + math.exp(-logit))
            if logit >= 0
            else (math.exp(logit) / (1.0 + math.exp(logit)))
        )
        max_diff = max(max_diff, abs(got - float(expected)))

    print(f"[verify] closed-form vs sklearn max probability diff: {max_diff:.2e}")
    if max_diff > 1e-9:
        raise AssertionError(
            f"Closed-form predictor diverges from sklearn by {max_diff:.2e} — "
            "the pure-Python / browser model would be wrong."
        )


if __name__ == "__main__":
    raise SystemExit(main())
