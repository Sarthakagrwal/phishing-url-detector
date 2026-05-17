/**
 * Vitest suite for the TypeScript feature extractor.
 *
 * Two layers of checks:
 * 1. Crafted-URL unit tests for each of the 20 features.
 * 2. A parity check against the shared `fixtures/parity_urls.json` — every URL's
 *    feature vector must equal the value the Python extractor produced.
 */

import { describe, expect, it } from 'vitest'
import {
  extractFeatures,
  FEATURE_NAMES,
  featureVector,
  splitRegistrableDomain,
} from './features'
import { loadParityFixture } from './fixtures'

describe('extractFeatures — individual features', () => {
  it('measures url_length, hostname_length and path_length', () => {
    const f = extractFeatures('https://example.com/path/to/page')
    expect(f.url_length).toBe('https://example.com/path/to/page'.length)
    expect(f.hostname_length).toBe('example.com'.length)
    expect(f.path_length).toBe('/path/to/page'.length)
  })

  it('counts dots across the whole URL and hyphens only in the host', () => {
    const f = extractFeatures('https://a-b.sub.example.com/x-y-z?p=1')
    expect(f.num_dots).toBe(3)
    expect(f.num_hyphens).toBe(1) // only the 'a-b' in the host
  })

  it('counts subdomains beyond the registrable domain', () => {
    expect(extractFeatures('https://example.com').num_subdomains).toBe(0)
    expect(extractFeatures('https://www.example.com').num_subdomains).toBe(1)
    expect(extractFeatures('https://a.b.c.example.com').num_subdomains).toBe(3)
    // multi-label eTLD: example.co.uk is the registrable domain
    expect(extractFeatures('https://www.example.co.uk').num_subdomains).toBe(1)
  })

  it('detects IPv4 and IPv6 literal hosts', () => {
    expect(extractFeatures('http://192.168.1.1/x').has_ip_host).toBe(1)
    expect(extractFeatures('http://8.8.8.8').has_ip_host).toBe(1)
    expect(extractFeatures('https://[2001:db8::1]/x').has_ip_host).toBe(1)
    expect(extractFeatures('https://example.com').has_ip_host).toBe(0)
    expect(extractFeatures('http://999.1.1.1').has_ip_host).toBe(0) // out of range
  })

  it('detects the @ symbol anywhere in the URL', () => {
    expect(extractFeatures('http://safe.com@evil.com/x').has_at_symbol).toBe(1)
    expect(extractFeatures('https://example.com/x').has_at_symbol).toBe(0)
  })

  it('counts query parameters', () => {
    expect(extractFeatures('https://example.com').num_query_params).toBe(0)
    expect(extractFeatures('https://example.com?a=1').num_query_params).toBe(1)
    expect(
      extractFeatures('https://example.com?a=1&b=2&c=3').num_query_params,
    ).toBe(3)
  })

  it('detects Punycode labels and non-ASCII (homograph) hosts', () => {
    expect(extractFeatures('http://xn--pple-43d.com').has_punycode).toBe(1)
    expect(extractFeatures('http://xn--pple-43d.com').has_homograph).toBe(0)
    expect(extractFeatures('https://göögle.com').has_homograph).toBe(1)
    expect(extractFeatures('https://example.com').has_punycode).toBe(0)
    expect(extractFeatures('https://example.com').has_homograph).toBe(0)
  })

  it('detects HTTPS', () => {
    expect(extractFeatures('https://example.com').is_https).toBe(1)
    expect(extractFeatures('http://example.com').is_https).toBe(0)
    // no scheme -> http:// is prepended -> not https
    expect(extractFeatures('example.com').is_https).toBe(0)
  })

  it('counts digits in the host and the digit ratio', () => {
    const f = extractFeatures('http://abc123def456.com')
    expect(f.num_digits_in_host).toBe(6)
    expect(f.digit_ratio_host).toBeCloseTo(6 / 'abc123def456.com'.length, 10)
    expect(extractFeatures('https://example.com').digit_ratio_host).toBe(0)
  })

  it('flags suspicious TLDs and known shorteners', () => {
    expect(extractFeatures('http://free-prize.xyz').suspicious_tld).toBe(1)
    expect(extractFeatures('http://thing.zip/x').suspicious_tld).toBe(1)
    expect(extractFeatures('https://example.com').suspicious_tld).toBe(0)
    expect(extractFeatures('https://bit.ly/abc').is_shortener).toBe(1)
    expect(extractFeatures('https://tinyurl.com/abc').is_shortener).toBe(1)
    expect(extractFeatures('https://example.com').is_shortener).toBe(0)
  })

  it('counts phishing keywords in the URL', () => {
    expect(
      extractFeatures('http://x.com/login/verify/account').num_suspicious_keywords,
    ).toBe(3)
    expect(extractFeatures('https://example.com').num_suspicious_keywords).toBe(0)
  })

  it('detects a double slash in the path', () => {
    expect(
      extractFeatures('http://example.com//redirect').has_double_slash_in_path,
    ).toBe(1)
    expect(
      extractFeatures('http://example.com/redirect').has_double_slash_in_path,
    ).toBe(0)
  })

  it('counts special characters and measures the TLD length', () => {
    const f = extractFeatures('https://example.com/x?a=b&c_d=e')
    // special chars are % = ? & _
    expect(f.num_special_chars).toBe(5)
    expect(extractFeatures('https://example.com').tld_length).toBe(3)
    expect(extractFeatures('https://example.co.uk').tld_length).toBe(5)
  })
})

describe('splitRegistrableDomain', () => {
  it('splits common and multi-label suffixes', () => {
    expect(splitRegistrableDomain('www.example.com')).toEqual([
      'www',
      'example.com',
      'com',
    ])
    expect(splitRegistrableDomain('a.b.example.co.uk')).toEqual([
      'a.b',
      'example.co.uk',
      'co.uk',
    ])
    expect(splitRegistrableDomain('example.com')).toEqual([
      '',
      'example.com',
      'com',
    ])
  })

  it('returns an empty eTLD for IP hosts', () => {
    expect(splitRegistrableDomain('192.168.1.1')).toEqual([
      '',
      '192.168.1.1',
      '',
    ])
  })
})

describe('parity with fixtures/parity_urls.json', () => {
  const fixture = loadParityFixture()

  it('uses the same feature order as the Python side', () => {
    expect([...FEATURE_NAMES]).toEqual(fixture.feature_names)
  })

  it('reproduces every fixture feature vector exactly', () => {
    for (const entry of fixture.urls) {
      const got = extractFeatures(entry.url)
      for (const name of FEATURE_NAMES) {
        // Integers must match exactly; the lone ratio within a tiny epsilon.
        expect(got[name], `${entry.url} :: ${name}`).toBeCloseTo(
          entry.features[name],
          9,
        )
      }
    }
  })

  it('feature vectors have all 20 entries', () => {
    for (const entry of fixture.urls) {
      expect(featureVector(entry.url)).toHaveLength(20)
    }
  })
})
