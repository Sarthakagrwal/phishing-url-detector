/**
 * DOM rendering for the PhishGuard page.
 *
 * Pure rendering helpers: each takes plain data and returns/updates DOM. The
 * actual classification happens in `predict.ts`; this module only displays it.
 */

import { FEATURE_NAMES, type FeatureName, type FeatureVector } from './features'
import type { Band, ClassificationResult, FeatureContribution } from './predict'

/** Human-readable labels for each of the 20 features (shown in the table). */
const FEATURE_LABELS: Record<FeatureName, string> = {
  url_length: 'URL length',
  hostname_length: 'Hostname length',
  path_length: 'Path length',
  num_dots: 'Dots in URL',
  num_hyphens: 'Hyphens in hostname',
  num_subdomains: 'Subdomain count',
  has_ip_host: 'Host is a raw IP',
  has_at_symbol: "Contains '@'",
  num_query_params: 'Query parameters',
  has_punycode: "Punycode ('xn--')",
  has_homograph: 'Non-ASCII host chars',
  is_https: 'Uses HTTPS',
  num_digits_in_host: 'Digits in hostname',
  digit_ratio_host: 'Digit ratio of host',
  suspicious_tld: 'Suspicious TLD',
  is_shortener: 'URL shortener',
  num_suspicious_keywords: 'Phishing keywords',
  has_double_slash_in_path: "'//' in path",
  num_special_chars: 'Special characters',
  tld_length: 'TLD length',
}

/** Feature names that are 0/1 booleans — rendered as a Yes/No flag. */
const BOOLEAN_FEATURES: ReadonlySet<FeatureName> = new Set<FeatureName>([
  'has_ip_host',
  'has_at_symbol',
  'has_punycode',
  'has_homograph',
  'is_https',
  'suspicious_tld',
  'is_shortener',
  'has_double_slash_in_path',
])

/** For a boolean feature, whether value=1 is the *risky* state. */
const RISKY_WHEN_ONE: ReadonlySet<FeatureName> = new Set<FeatureName>([
  'has_ip_host',
  'has_at_symbol',
  'has_punycode',
  'has_homograph',
  'suspicious_tld',
  'is_shortener',
  'has_double_slash_in_path',
])

/** Map a band to the theme's risk-colour suffix (`safe` / `warn` / `danger`). */
function bandColour(band: Band): 'safe' | 'warn' | 'danger' {
  if (band === 'Dangerous') {
    return 'danger'
  }
  if (band === 'Suspicious') {
    return 'warn'
  }
  return 'safe'
}

/** A one-line plain-English summary for each band. */
function bandSummary(band: Band): string {
  if (band === 'Dangerous') {
    return 'Strong phishing signals detected. Do not enter credentials or personal data.'
  }
  if (band === 'Suspicious') {
    return 'Some risky traits found. Treat this link with caution and verify the source.'
  }
  return 'No strong phishing signals found. Always stay alert regardless.'
}

/** Escape a string for safe insertion as HTML text content. */
function esc(text: string): string {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

/** Format a feature value compactly (integers plain, ratios to 3 dp). */
function formatValue(name: FeatureName, value: number): string {
  if (name === 'digit_ratio_host') {
    return value.toFixed(3)
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(2)
}

/** Render the big verdict banner (score + band + summary). */
export function renderVerdict(result: ClassificationResult): string {
  const colour = bandColour(result.band)
  return `
    <div class="verdict verdict--${colour}" data-testid="verdict">
      <div class="verdict__score" data-testid="verdict-score">${result.finalScore.toFixed(0)}</div>
      <div class="verdict__body">
        <div class="verdict__label" data-testid="verdict-band">${result.band}</div>
        <div class="verdict__detail">${esc(bandSummary(result.band))}</div>
      </div>
    </div>
    <div class="meter" aria-hidden="true">
      <div class="meter__fill meter__fill--${colour}" style="width:${result.finalScore}%"></div>
    </div>
    <div class="score-split">
      <div class="stat">
        <div class="stat__value">${(result.mlProbability * 100).toFixed(1)}%</div>
        <div class="stat__label">ML model phishing probability</div>
      </div>
      <div class="stat">
        <div class="stat__value">${result.heuristicScore}/100</div>
        <div class="stat__label">Heuristic rule score</div>
      </div>
    </div>
  `
}

/** Render the triggered-heuristics reason list. */
export function renderReasons(result: ClassificationResult): string {
  if (result.reasons.length === 0) {
    return `
      <ul class="reasons" data-testid="reasons">
        <li><span class="dot dot--safe"></span><span>No heuristic rules were triggered by this URL.</span></li>
      </ul>
    `
  }
  const items = result.reasons
    .map(
      (hit) => `
      <li>
        <span class="dot dot--${hit.severity === 'high' ? 'danger' : hit.severity === 'medium' ? 'warn' : 'safe'}"></span>
        <span><strong>+${hit.points}</strong> &nbsp;${esc(hit.reason)}</span>
      </li>`,
    )
    .join('')
  return `<ul class="reasons" data-testid="reasons">${items}</ul>`
}

/** Render the per-feature breakdown table (all 20 features). */
export function renderFeatureTable(features: FeatureVector): string {
  const rows = FEATURE_NAMES.map((name) => {
    const value = features[name]
    let cell: string
    if (BOOLEAN_FEATURES.has(name)) {
      const isOne = value >= 1
      const risky = RISKY_WHEN_ONE.has(name) ? isOne : !isOne
      const text = isOne ? 'Yes' : 'No'
      cell = `<td class="feat-flag ${risky ? 'flag-yes' : 'flag-no'}">${text}</td>`
    } else {
      cell = `<td class="mono">${formatValue(name, value)}</td>`
    }
    return `<tr><td>${esc(FEATURE_LABELS[name])}</td>${cell}</tr>`
  }).join('')
  return `
    <div class="feature-table-wrap">
      <table class="data-table" data-testid="feature-table">
        <thead><tr><th>Feature</th><th>Value</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `
}

/** Render the signed ML-contribution bars (largest magnitude first). */
export function renderContributions(contributions: FeatureContribution[]): string {
  const maxMag = Math.max(
    0.001,
    ...contributions.map((c) => Math.abs(c.contribution)),
  )
  const rows = contributions
    .map((c) => {
      const pos = c.contribution >= 0
      const widthPct = (Math.abs(c.contribution) / maxMag) * 50
      const bar = pos
        ? `<div class="contrib__bar contrib__bar--pos" style="width:${widthPct}%"></div>`
        : `<div class="contrib__bar contrib__bar--neg" style="width:${widthPct}%"></div>`
      const sign = pos ? '+' : '-'
      return `
        <div class="contrib__row">
          <div class="contrib__name" title="${esc(c.name)}">${esc(FEATURE_LABELS[c.name as FeatureName] ?? c.name)}</div>
          <div class="contrib__track">${bar}</div>
          <div class="contrib__value contrib__value--${pos ? 'pos' : 'neg'}">${sign}${Math.abs(c.contribution).toFixed(2)}</div>
        </div>`
    })
    .join('')
  return `<div class="contrib" data-testid="contributions">${rows}</div>`
}
