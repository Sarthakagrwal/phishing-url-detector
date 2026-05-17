/**
 * Vitest suite for the TypeScript heuristic rule engine.
 *
 * Checks that each rule fires exactly when expected, contributes the right
 * points, and that the overall score is the clamped sum. Also parity-checks the
 * triggered rule ids against the shared fixture.
 */

import { describe, expect, it } from 'vitest'
import { extractFeatures } from './features'
import { evaluate, MAX_SCORE } from './heuristics'
import { loadParityFixture } from './fixtures'

/** Evaluate the heuristics for a URL and return the result. */
function heuristicsFor(url: string) {
  return evaluate(extractFeatures(url))
}

/** Return the set of rule ids that fired for a URL. */
function firedIds(url: string): Set<string> {
  return new Set(heuristicsFor(url).hits.map((h) => h.ruleId))
}

describe('heuristic rules — fire conditions', () => {
  it('flags a raw-IP host', () => {
    const ids = firedIds('http://203.0.113.5/x')
    expect(ids.has('ip_host')).toBe(true)
    const hit = heuristicsFor('http://203.0.113.5/x').hits.find(
      (h) => h.ruleId === 'ip_host',
    )
    expect(hit?.points).toBe(30)
    expect(hit?.reason).toContain('raw IP address')
  })

  it('flags an @ symbol', () => {
    const ids = firedIds('http://paypal.com@evil.example/x')
    expect(ids.has('at_symbol')).toBe(true)
  })

  it('flags Punycode but not homograph for an xn-- host', () => {
    const ids = firedIds('http://xn--pple-43d.com/x')
    expect(ids.has('punycode')).toBe(true)
    expect(ids.has('homograph')).toBe(false)
  })

  it('flags a non-ASCII homograph host', () => {
    const ids = firedIds('https://göögle.com')
    expect(ids.has('homograph')).toBe(true)
    expect(ids.has('punycode')).toBe(false)
  })

  it('flags a suspicious TLD', () => {
    expect(firedIds('http://thing.xyz').has('suspicious_tld')).toBe(true)
    expect(firedIds('https://example.com').has('suspicious_tld')).toBe(false)
  })

  it('flags a known URL shortener', () => {
    expect(firedIds('https://bit.ly/abc').has('shortener')).toBe(true)
  })

  it('flags the absence of HTTPS', () => {
    expect(firedIds('http://example.com').has('no_https')).toBe(true)
    expect(firedIds('https://example.com').has('no_https')).toBe(false)
  })

  it('flags a deep subdomain chain', () => {
    expect(firedIds('https://a.b.c.example.com').has('many_subdomains')).toBe(
      true,
    )
    expect(firedIds('https://www.example.com').has('many_subdomains')).toBe(
      false,
    )
  })

  it('uses the single vs multiple keyword variants', () => {
    expect(
      firedIds('https://example.com/login').has('suspicious_keyword_single'),
    ).toBe(true)
    expect(
      firedIds('https://example.com/login/verify').has('suspicious_keywords'),
    ).toBe(true)
  })

  it('flags a hyphen-stuffed host', () => {
    expect(
      firedIds('http://a-b-c-d.example.com').has('many_hyphens'),
    ).toBe(true)
  })

  it('flags a digit-heavy host (but not a plain IP)', () => {
    expect(firedIds('http://1234567890ab.com').has('digit_heavy_host')).toBe(
      true,
    )
    // an IP host must NOT also trigger the digit-heavy rule
    expect(firedIds('http://192.168.1.1').has('digit_heavy_host')).toBe(false)
  })

  it('flags a double slash in the path and a very long URL', () => {
    expect(
      firedIds('http://example.com//x').has('double_slash_path'),
    ).toBe(true)
    const longUrl = 'http://example.com/' + 'a'.repeat(120)
    expect(firedIds(longUrl).has('long_url')).toBe(true)
  })
})

describe('heuristic scoring', () => {
  it('scores a clean HTTPS URL at zero', () => {
    expect(heuristicsFor('https://github.com').score).toBe(0)
  })

  it('sums points and clamps to the max', () => {
    const result = heuristicsFor(
      'http://secure-login-verify-account.paypal-update.gq@198.51.100.9/webscr',
    )
    expect(result.score).toBeGreaterThan(0)
    expect(result.score).toBeLessThanOrEqual(MAX_SCORE)
    // the sum of fired points, clamped, must equal the reported score
    const rawSum = result.hits.reduce((acc, h) => acc + h.points, 0)
    expect(result.score).toBe(Math.min(MAX_SCORE, rawSum))
  })
})

describe('parity with fixtures/parity_urls.json', () => {
  const fixture = loadParityFixture()

  it('fires the same heuristic rules and score as the Python side', () => {
    for (const entry of fixture.urls) {
      const result = heuristicsFor(entry.url)
      expect(result.score, `${entry.url} score`).toBe(entry.heuristic_score)
      expect(
        result.hits.map((h) => h.ruleId),
        `${entry.url} reasons`,
      ).toEqual(entry.reason_ids)
    }
  })
})
