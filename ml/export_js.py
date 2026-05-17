"""Generate ``web/src/generated/model.ts`` from ``models/model_meta.json``.

The website runs the *same* logistic-regression model as the Python CLI, fully
in the browser. Rather than ship the JSON and parse it at runtime, this script
bakes the model parameters into a typed TypeScript constant so the bundler can
tree-shake and type-check it like any other source.

This is NOT a model-to-code transpiler (no m2cgen, no decision-tree codegen).
It only copies a handful of floating-point arrays — the closed-form parameters
``coef``, ``intercept``, ``mean``, ``scale`` plus the ordered ``featureNames``
and ``threshold``. ``web/src/predict.ts`` evaluates the closed-form sigmoid over
them, exactly as :mod:`phishdetect.model` does in Python.

Run ``python ml/export_js.py`` after ``ml/train.py``. The generated file is
committed so CI does not need to retrain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from phishdetect.features import FEATURE_NAMES  # noqa: E402

MODEL_META = _REPO_ROOT / "models" / "model_meta.json"
OUTPUT_TS = _REPO_ROOT / "web" / "src" / "generated" / "model.ts"


def _fmt_number_array(values: list[float], indent: str) -> str:
    """Format a list of floats as a multi-line TS numeric-array literal."""
    parts = [repr(float(v)) for v in values]
    body = ", ".join(parts)
    return f"[\n{indent}  {body},\n{indent}]"


def _fmt_string_array(values: list[str], indent: str) -> str:
    """Format a list of strings as a multi-line TS string-array literal."""
    quoted = [f'"{v}"' for v in values]
    lines = []
    for i in range(0, len(quoted), 4):
        lines.append(indent + "  " + ", ".join(quoted[i : i + 4]) + ",")
    return "[\n" + "\n".join(lines) + f"\n{indent}]"


def main() -> int:
    """Read the trained model metadata and write the typed TS constant. Returns 0."""
    if not MODEL_META.is_file():
        print(
            f"error: {MODEL_META} not found — run `python ml/train.py` first.",
            file=sys.stderr,
        )
        return 1

    meta = json.loads(MODEL_META.read_text(encoding="utf-8"))

    feature_names = list(meta["feature_names"])
    if feature_names != FEATURE_NAMES:
        print(
            "error: model_meta.json feature order does not match "
            "phishdetect.features.FEATURE_NAMES — retrain the model.",
            file=sys.stderr,
        )
        return 1

    metrics = meta.get("metrics", {})
    coef = [float(c) for c in meta["coef"]]
    mean = [float(m) for m in meta["mean"]]
    scale = [float(s) for s in meta["scale"]]
    intercept = float(meta["intercept"])
    threshold = float(meta["threshold"])

    metrics_lines = ",\n".join(
        f"  {key}: {float(value)!r}" for key, value in sorted(metrics.items())
    )

    content = f"""/**
 * AUTO-GENERATED — do not edit by hand.
 *
 * Source: models/model_meta.json
 * Generator: ml/export_js.py
 *
 * These are the parameters of the logistic-regression phishing classifier,
 * trained offline on the PhiUSIIL Phishing URL Dataset (UCI #967) by
 * ml/train.py. The browser runs the *identical* model as the Python CLI by
 * evaluating the closed-form sigmoid over these numbers in src/predict.ts:
 *
 *   z_i   = (x_i - mean_i) / scale_i
 *   logit = intercept + Σ_i coef_i · z_i
 *   p     = 1 / (1 + e^(-logit))
 *
 * Regenerate with:  python ml/export_js.py
 */

export interface PhishingModel {{
  /** Ordered feature names; the coefficients are aligned to this order. */
  readonly featureNames: readonly string[];
  /** Per-feature logistic-regression weights. */
  readonly coef: readonly number[];
  /** Logistic-regression bias term. */
  readonly intercept: number;
  /** Per-feature mean from the fitted StandardScaler. */
  readonly mean: readonly number[];
  /** Per-feature standard deviation from the fitted StandardScaler. */
  readonly scale: readonly number[];
  /** Decision threshold on the phishing probability. */
  readonly threshold: number;
  /** Held-out test-set metrics recorded at training time. */
  readonly metrics: {{
    readonly accuracy: number;
    readonly precision: number;
    readonly recall: number;
    readonly f1: number;
    readonly roc_auc: number;
  }};
}}

export const MODEL: PhishingModel = {{
  featureNames: {_fmt_string_array(feature_names, "  ")},
  coef: {_fmt_number_array(coef, "  ")},
  intercept: {intercept!r},
  mean: {_fmt_number_array(mean, "  ")},
  scale: {_fmt_number_array(scale, "  ")},
  threshold: {threshold!r},
  metrics: {{
{metrics_lines},
  }},
}};
"""

    OUTPUT_TS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TS.write_text(content, encoding="utf-8")
    print(f"[export] wrote {OUTPUT_TS.relative_to(_REPO_ROOT)}")
    print(f"[export] {len(feature_names)} features, accuracy={metrics.get('accuracy', 0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
