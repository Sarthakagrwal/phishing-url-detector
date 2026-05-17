/**
 * Transparent heuristic rule engine — TypeScript port of `phishdetect/heuristics.py`.
 *
 * Same rules, same point values, same human-readable reasons and severities as
 * the Python engine, so the website's explanations match the CLI's exactly.
 * Each rule is a pure function of the 20-feature vector.
 */

import type { FeatureVector } from './features'

/** Maximum heuristic score; the raw point sum is clamped to this. */
export const MAX_SCORE = 100

/** Severity drives the UI dot colour. */
export type Severity = 'low' | 'medium' | 'high'

/** A single triggered heuristic rule. Mirrors Python RuleHit. */
export interface RuleHit {
  /** Stable short identifier. */
  readonly ruleId: string
  /** Risk points this rule contributes. */
  readonly points: number
  /** Human-readable explanation for a non-expert. */
  readonly reason: string
  /** `'low' | 'medium' | 'high'`. */
  readonly severity: Severity
}

/** A rule: feature vector -> RuleHit or null (did not fire). */
type Rule = (f: FeatureVector) => RuleHit | null

const ruleIpHost: Rule = (f) =>
  f.has_ip_host >= 1
    ? {
        ruleId: 'ip_host',
        points: 30,
        reason:
          'Uses a raw IP address instead of a domain name — legitimate sites ' +
          'almost always use a named domain.',
        severity: 'high',
      }
    : null

const ruleAtSymbol: Rule = (f) =>
  f.has_at_symbol >= 1
    ? {
        ruleId: 'at_symbol',
        points: 25,
        reason:
          "Contains an '@' symbol — a browser ignores everything before it, " +
          'so the real destination can be hidden.',
        severity: 'high',
      }
    : null

const rulePunycode: Rule = (f) =>
  f.has_punycode >= 1
    ? {
        ruleId: 'punycode',
        points: 22,
        reason:
          "Host uses Punycode ('xn--') — often used to spoof a trusted brand " +
          'with look-alike characters.',
        severity: 'high',
      }
    : null

const ruleHomograph: Rule = (f) =>
  f.has_homograph >= 1 && f.has_punycode < 1
    ? {
        ruleId: 'homograph',
        points: 20,
        reason:
          'Host contains non-ASCII characters that can visually imitate a ' +
          'trusted domain (homograph attack).',
        severity: 'high',
      }
    : null

const ruleSuspiciousTld: Rule = (f) =>
  f.suspicious_tld >= 1
    ? {
        ruleId: 'suspicious_tld',
        points: 18,
        reason:
          'Uses a top-level domain frequently associated with abuse ' +
          '(e.g. .zip, .xyz, .top, .tk).',
        severity: 'medium',
      }
    : null

const ruleShortener: Rule = (f) =>
  f.is_shortener >= 1
    ? {
        ruleId: 'shortener',
        points: 15,
        reason:
          'This is a URL-shortening service — the real destination is hidden ' +
          'and cannot be inspected before clicking.',
        severity: 'medium',
      }
    : null

const ruleNoHttps: Rule = (f) =>
  f.is_https < 1
    ? {
        ruleId: 'no_https',
        points: 10,
        reason:
          "Does not use HTTPS — traffic is not encrypted and the site's " +
          'identity is not verified.',
        severity: 'low',
      }
    : null

const ruleManySubdomains: Rule = (f) =>
  f.num_subdomains >= 3
    ? {
        ruleId: 'many_subdomains',
        points: 14,
        reason:
          'Has an unusually deep chain of subdomains — phishing pages bury a ' +
          'trusted brand name inside a long host to look legitimate.',
        severity: 'medium',
      }
    : null

const ruleSuspiciousKeywords: Rule = (f) => {
  const n = Math.trunc(f.num_suspicious_keywords)
  if (n >= 2) {
    return {
      ruleId: 'suspicious_keywords',
      points: 16,
      reason:
        `Contains ${n} security/banking trigger words (login, verify, ` +
        'account, bank, …) — common bait in phishing links.',
      severity: 'medium',
    }
  }
  if (n === 1) {
    return {
      ruleId: 'suspicious_keyword_single',
      points: 7,
      reason:
        'Contains a security/banking trigger word (login, verify, account, ' +
        '…) — mildly suspicious on its own.',
      severity: 'low',
    }
  }
  return null
}

const ruleDoubleSlashPath: Rule = (f) =>
  f.has_double_slash_in_path >= 1
    ? {
        ruleId: 'double_slash_path',
        points: 10,
        reason:
          "Path contains '//' — sometimes used to smuggle a second URL or " +
          'trigger an open redirect.',
        severity: 'low',
      }
    : null

const ruleManyHyphens: Rule = (f) =>
  f.num_hyphens >= 3
    ? {
        ruleId: 'many_hyphens',
        points: 12,
        reason:
          'Hostname is stuffed with hyphens — a pattern used to assemble ' +
          "fake brand names like 'secure-login-yourbank-com'.",
        severity: 'medium',
      }
    : null

const ruleLongUrl: Rule = (f) =>
  f.url_length >= 100
    ? {
        ruleId: 'long_url',
        points: 10,
        reason:
          'The URL is very long — excessive length is often used to push the ' +
          'deceptive part out of view in the address bar.',
        severity: 'low',
      }
    : null

const ruleDigitHeavyHost: Rule = (f) =>
  f.has_ip_host < 1 && f.digit_ratio_host >= 0.3
    ? {
        ruleId: 'digit_heavy_host',
        points: 12,
        reason:
          'The hostname is unusually full of digits — legitimate brand ' +
          'domains are rarely mostly numbers.',
        severity: 'medium',
      }
    : null

/** The ordered rule set; order only affects display order. Mirrors RULES. */
const RULES: readonly Rule[] = [
  ruleIpHost,
  ruleAtSymbol,
  rulePunycode,
  ruleHomograph,
  ruleSuspiciousTld,
  ruleShortener,
  ruleManySubdomains,
  ruleSuspiciousKeywords,
  ruleManyHyphens,
  ruleDigitHeavyHost,
  ruleDoubleSlashPath,
  ruleLongUrl,
  ruleNoHttps,
]

/** The outcome of running every heuristic rule. Mirrors Python HeuristicResult. */
export interface HeuristicResult {
  /** Integer risk score 0-100 (clamped sum of fired rules' points). */
  readonly score: number
  /** The rules that fired, in rule order. */
  readonly hits: RuleHit[]
}

/** Run every heuristic rule against `features` and return the result. */
export function evaluate(features: FeatureVector): HeuristicResult {
  const hits: RuleHit[] = []
  let total = 0
  for (const rule of RULES) {
    const hit = rule(features)
    if (hit !== null) {
      hits.push(hit)
      total += hit.points
    }
  }
  const score = Math.max(0, Math.min(MAX_SCORE, total))
  return { score, hits }
}
