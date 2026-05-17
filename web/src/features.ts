/**
 * URL feature extraction — TypeScript port of `phishdetect/features.py`.
 *
 * Extracts the SAME 20 string-only features the Python library does, so the
 * browser and the CLI feed the model identical vectors. No network of any kind.
 *
 * Parity notes
 * ------------
 * - Python's `len(str)`, `str.count(...)` and `ord(c)` operate on Unicode code
 *   *points*. JavaScript strings are UTF-16; naive `.length` and indexing count
 *   *code units*, which differ for astral characters. Every length/count here
 *   therefore iterates with `[...str]` / `for...of`, which yield code points,
 *   matching Python exactly.
 * - The feature order is fixed and identical to `FEATURE_NAMES` in Python.
 *
 * The shared `fixtures/parity_urls.json` fails if this drifts from `features.py`.
 */

import { KNOWN_TLDS, MULTI_LABEL_SUFFIXES } from './suffixList'
import { parseUrl, type ParsedURL } from './urlparse'

/** Fixed, ordered feature list — the model's coefficients align to this order. */
export const FEATURE_NAMES = [
  'url_length',
  'hostname_length',
  'path_length',
  'num_dots',
  'num_hyphens',
  'num_subdomains',
  'has_ip_host',
  'has_at_symbol',
  'num_query_params',
  'has_punycode',
  'has_homograph',
  'is_https',
  'num_digits_in_host',
  'digit_ratio_host',
  'suspicious_tld',
  'is_shortener',
  'num_suspicious_keywords',
  'has_double_slash_in_path',
  'num_special_chars',
  'tld_length',
] as const

/** A union of the 20 feature names. */
export type FeatureName = (typeof FEATURE_NAMES)[number]

/** The 20-feature record produced by {@link extractFeatures}. */
export type FeatureVector = Record<FeatureName, number>

/** TLDs disproportionately abused for phishing. Mirrors SUSPICIOUS_TLDS. */
export const SUSPICIOUS_TLDS: ReadonlySet<string> = new Set<string>([
  'zip',
  'mov',
  'xyz',
  'top',
  'tk',
  'gq',
  'ml',
  'cf',
  'ga',
  'country',
  'kim',
  'work',
  'click',
  'link',
  'gdn',
  'loan',
  'download',
  'review',
  'stream',
  'racing',
  'win',
  'bid',
  'party',
  'date',
  'men',
  'cricket',
  'science',
  'accountant',
  'trade',
])

/** Well-known URL-shortening services. Mirrors URL_SHORTENERS. */
export const URL_SHORTENERS: ReadonlySet<string> = new Set<string>([
  'bit.ly',
  'tinyurl.com',
  'goo.gl',
  't.co',
  'ow.ly',
  'is.gd',
  'buff.ly',
  'cutt.ly',
  'rebrand.ly',
  'bit.do',
  'shorturl.at',
  'rb.gy',
  'tiny.cc',
  't.ly',
  'soo.gd',
  's.id',
  'lnkd.in',
  'db.tt',
  'qr.ae',
  'adf.ly',
  'tr.im',
  'v.gd',
])

/** Phishing keywords counted in the lowercased URL. Mirrors SUSPICIOUS_KEYWORDS. */
export const SUSPICIOUS_KEYWORDS: readonly string[] = [
  'login',
  'secure',
  'account',
  'verify',
  'update',
  'bank',
  'confirm',
  'signin',
  'password',
  'webscr',
  'ebayisapi',
]

/** Count occurrences of `ch` in `text` (single-character code-point counting). */
function countChar(text: string, ch: string): number {
  let n = 0
  for (const c of text) {
    if (c === ch) {
      n += 1
    }
  }
  return n
}

/** Count code points in `text` (Python `len(str)` semantics). */
function codePointLength(text: string): number {
  let n = 0
  for (const _c of text) {
    n += 1
  }
  return n
}

/** Return true if `host` is a dotted-decimal IPv4 literal. */
function isIpv4(host: string): boolean {
  const parts = host.split('.')
  if (parts.length !== 4) {
    return false
  }
  for (const part of parts) {
    if (part === '' || part.length > 3) {
      return false
    }
    if (![...part].every((c) => c >= '0' && c <= '9')) {
      return false
    }
    if (Number.parseInt(part, 10) > 255) {
      return false
    }
  }
  return true
}

/** Return true if `host` is a bracketed IPv6 literal (`[...]`). */
function isIpv6(host: string): boolean {
  if (!(host.startsWith('[') && host.endsWith(']'))) {
    return false
  }
  const inner = host.slice(1, -1)
  if (inner === '') {
    return false
  }
  if (inner.includes('::')) {
    if ((inner.match(/::/g) ?? []).length > 1) {
      return false
    }
  }
  const allowed = '0123456789abcdefABCDEF:.'
  return [...inner].every((c) => allowed.includes(c)) && inner.includes(':')
}

/** Return true if the host is an IPv4 or IPv6 literal. */
function isIpHost(host: string): boolean {
  return isIpv4(host) || isIpv6(host)
}

/** Return true if every character of the host is ASCII (code point < 128). */
function hostIsAscii(host: string): boolean {
  for (const c of host) {
    if ((c.codePointAt(0) ?? 0) >= 128) {
      return false
    }
  }
  return true
}

/**
 * Split a host into `[subdomains, registrableDomain, etld]`.
 * Faithful port of `split_registrable_domain` in Python.
 */
export function splitRegistrableDomain(host: string): [string, string, string] {
  if (host === '' || isIpHost(host)) {
    return ['', host, '']
  }
  const labels = host.split('.')
  if (labels.length < 2) {
    return ['', host, '']
  }

  let etld = ''
  if (labels.length >= 3) {
    const two = labels.slice(-2).join('.')
    if (MULTI_LABEL_SUFFIXES.has(two)) {
      etld = two
    }
  }
  if (etld === '') {
    etld = labels[labels.length - 1]
    void KNOWN_TLDS // referenced for parity with the Python module
  }

  const etldLabels = countChar(etld, '.') + 1
  if (labels.length <= etldLabels) {
    return ['', host, etld]
  }

  const registrable = labels.slice(-(etldLabels + 1)).join('.')
  const subdomainLabels = labels.slice(0, labels.length - (etldLabels + 1))
  return [subdomainLabels.join('.'), registrable, etld]
}

/** Count parameters in a raw query string (split on `&` and legacy `;`). */
function countQueryParams(query: string): number {
  if (query === '') {
    return 0
  }
  let count = 0
  for (const chunk of query.replace(/;/g, '&').split('&')) {
    if (chunk !== '') {
      count += 1
    }
  }
  return count
}

/** Compute the 20-feature record from an already-{@link parseUrl}d result. */
export function extractFeaturesFromParsed(parsed: ParsedURL): FeatureVector {
  const urlStr = parsed.original
  const urlLower = urlStr.toLowerCase()
  const host = parsed.host

  const hostLabels = host ? host.split('.') : []
  const [subdomains, , etld] = splitRegistrableDomain(host)
  const numSubdomains = subdomains !== '' ? subdomains.split('.').length : 0

  let digitsInHost = 0
  for (const c of host) {
    if (c >= '0' && c <= '9') {
      digitsInHost += 1
    }
  }
  const hostnameLength = codePointLength(host)
  const digitRatio = hostnameLength > 0 ? digitsInHost / hostnameLength : 0

  const hasPunycode = hostLabels.some((lbl) => lbl.startsWith('xn--')) ? 1 : 0
  const hasHomograph = hostIsAscii(host) ? 0 : 1

  let numKeywords = 0
  for (const kw of SUSPICIOUS_KEYWORDS) {
    if (urlLower.includes(kw)) {
      numKeywords += 1
    }
  }

  let numSpecial = 0
  for (const c of urlStr) {
    if ('%=?&_'.includes(c)) {
      numSpecial += 1
    }
  }

  const features: FeatureVector = {
    url_length: codePointLength(urlStr),
    hostname_length: hostnameLength,
    path_length: codePointLength(parsed.path),
    num_dots: countChar(urlStr, '.'),
    num_hyphens: countChar(host, '-'),
    num_subdomains: numSubdomains,
    has_ip_host: isIpHost(host) ? 1 : 0,
    has_at_symbol: urlStr.includes('@') ? 1 : 0,
    num_query_params: countQueryParams(parsed.query),
    has_punycode: hasPunycode,
    has_homograph: hasHomograph,
    is_https: parsed.scheme === 'https' ? 1 : 0,
    num_digits_in_host: digitsInHost,
    digit_ratio_host: digitRatio,
    suspicious_tld: SUSPICIOUS_TLDS.has(etld) ? 1 : 0,
    is_shortener: URL_SHORTENERS.has(host) ? 1 : 0,
    num_suspicious_keywords: numKeywords,
    has_double_slash_in_path: parsed.path.includes('//') ? 1 : 0,
    num_special_chars: numSpecial,
    tld_length: codePointLength(etld),
  }
  return features
}

/** Extract the 20-feature record from a raw URL string. */
export function extractFeatures(url: string): FeatureVector {
  return extractFeaturesFromParsed(parseUrl(url))
}

/** Return the 20 feature values as an ordered `number[]` (the model's input). */
export function featureVector(url: string): number[] {
  const feats = extractFeatures(url)
  return FEATURE_NAMES.map((name) => feats[name])
}
