/**
 * End-to-end phishing classification — TypeScript port of `phishdetect/classify.py`.
 *
 * Runs entirely in the browser: deterministic parse -> 20 features -> heuristic
 * rules -> closed-form logistic-regression probability -> blended verdict. No
 * network call is ever made; the URL never leaves the page.
 *
 * The blend weights and band thresholds are the SAME constants as `classify.py`,
 * and the model parameters come from the generated `model.ts` — so this and the
 * Python CLI always produce the identical verdict (verified by parity tests).
 */

import {
  extractFeaturesFromParsed,
  FEATURE_NAMES,
  type FeatureName,
  type FeatureVector,
} from './features'
import { MODEL } from './generated/model'
import { evaluate, type RuleHit } from './heuristics'
import { parseUrl } from './urlparse'

// --- Blend weights and band thresholds (identical to classify.py) ----------------
export const ML_WEIGHT = 0.6
export const HEURISTIC_WEIGHT = 0.4
export const BAND_SUSPICIOUS_AT = 35
export const BAND_DANGEROUS_AT = 65
export const TOP_CONTRIBUTIONS = 6

/** The verdict band. */
export type Band = 'Safe' | 'Suspicious' | 'Dangerous'

/** A single feature's signed push on the ML logit. Mirrors FeatureContribution. */
export interface FeatureContribution {
  /** Feature name. */
  readonly name: FeatureName
  /** Raw feature value. */
  readonly value: number
  /** Signed logit contribution (`+` = phishing, `-` = safe). */
  readonly contribution: number
}

/** The complete structured outcome of classifying one URL. */
export interface ClassificationResult {
  /** The input URL string (trimmed). */
  readonly url: string
  /** Blended phishing-risk score 0-100 (rounded to one decimal). */
  readonly finalScore: number
  /** `'Safe' | 'Suspicious' | 'Dangerous'`. */
  readonly band: Band
  /** The model's raw phishing probability 0-1. */
  readonly mlProbability: number
  /** The transparent heuristic engine's score 0-100. */
  readonly heuristicScore: number
  /** The full 20-feature record. */
  readonly features: FeatureVector
  /** Top signed ML contributions, largest magnitude first. */
  readonly contributions: FeatureContribution[]
  /** Triggered heuristic rules (human-readable explanations). */
  readonly reasons: RuleHit[]
}

/**
 * Numerically stable logistic sigmoid `1 / (1 + e^-x)`.
 * The sign split avoids `Math.exp` overflow — identical to the Python `_sigmoid`.
 */
function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x))
  }
  const ex = Math.exp(x)
  return ex / (1 + ex)
}

/** Apply the fitted StandardScaler to a raw feature vector. */
function standardize(raw: number[]): number[] {
  return raw.map((x, i) => {
    const denom = MODEL.scale[i] !== 0 ? MODEL.scale[i] : 1
    return (x - MODEL.mean[i]) / denom
  })
}

/** Return the raw logit `b + Σ wᵢ·zᵢ` for a raw feature vector. */
function logit(raw: number[]): number {
  const z = standardize(raw)
  let total = MODEL.intercept
  for (let i = 0; i < z.length; i += 1) {
    total += MODEL.coef[i] * z[i]
  }
  return total
}

/** Return the phishing probability in `[0, 1]` for a raw feature vector. */
export function predictProba(raw: number[]): number {
  return sigmoid(logit(raw))
}

/** Map a 0-100 final score to its verdict band. */
function bandFor(score: number): Band {
  if (score >= BAND_DANGEROUS_AT) {
    return 'Dangerous'
  }
  if (score >= BAND_SUSPICIOUS_AT) {
    return 'Suspicious'
  }
  return 'Safe'
}

/** Round to one decimal place (matches Python `round(x, 1)`). */
function round1(x: number): number {
  return Math.round(x * 10) / 10
}

/**
 * Classify a single URL and return a structured {@link ClassificationResult}.
 * Faithful port of `phishdetect.classify.classify`.
 */
export function classify(url: string): ClassificationResult {
  const parsed = parseUrl(url)
  const features = extractFeaturesFromParsed(parsed)
  const rawVector = FEATURE_NAMES.map((name) => features[name])

  const heuristics = evaluate(features)
  const mlProbability = predictProba(rawVector)

  let finalScore =
    ML_WEIGHT * (mlProbability * 100) + HEURISTIC_WEIGHT * heuristics.score
  finalScore = Math.max(0, Math.min(100, finalScore))

  // Signed per-feature contributions to the logit, largest magnitude first.
  const z = standardize(rawVector)
  const signed: Array<[FeatureName, number]> = FEATURE_NAMES.map((name, i) => [
    name,
    MODEL.coef[i] * z[i],
  ])
  signed.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
  const contributions: FeatureContribution[] = signed
    .slice(0, TOP_CONTRIBUTIONS)
    .map(([name, contribution]) => ({
      name,
      value: features[name],
      contribution,
    }))

  return {
    url: parsed.original,
    finalScore: round1(finalScore),
    band: bandFor(finalScore),
    mlProbability,
    heuristicScore: heuristics.score,
    features,
    contributions,
    reasons: [...heuristics.hits],
  }
}
