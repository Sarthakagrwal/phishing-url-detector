/**
 * Vitest suite for the end-to-end TypeScript predictor.
 *
 * Verifies behaviour (probability range, banding, contributions) and parity:
 * every URL in `fixtures/parity_urls.json` must produce the same ML probability,
 * blended score and band as the Python `classify()`.
 */

import { describe, expect, it } from 'vitest'
import { loadParityFixture } from './fixtures'
import { classify, predictProba } from './predict'
import { featureVector } from './features'
import { MODEL } from './generated/model'

describe('predictProba', () => {
  it('always returns a probability in [0, 1]', () => {
    const urls = [
      'https://github.com',
      'http://192.168.0.1/login/verify/account',
      'http://a-b-c-secure-login.gq@1.2.3.4/webscr',
      '',
    ]
    for (const url of urls) {
      const p = predictProba(featureVector(url))
      expect(p).toBeGreaterThanOrEqual(0)
      expect(p).toBeLessThanOrEqual(1)
    }
  })
})

describe('classify — behaviour', () => {
  it('rates a well-known HTTPS site as Safe', () => {
    const result = classify('https://github.com')
    expect(result.band).toBe('Safe')
    expect(result.finalScore).toBeLessThan(35)
  })

  it('rates a blatant phishing URL as Dangerous', () => {
    const result = classify(
      'http://secure-login-verify-account.paypal-update.gq/webscr/signin',
    )
    expect(result.band).toBe('Dangerous')
    expect(result.finalScore).toBeGreaterThanOrEqual(65)
  })

  it('rates a raw-IP login page as Dangerous', () => {
    const result = classify('http://203.0.113.42/account/verify.php')
    expect(result.band).toBe('Dangerous')
  })

  it('returns the full structured result', () => {
    const result = classify('https://example.com/login')
    expect(result.url).toBe('https://example.com/login')
    expect(Object.keys(result.features)).toHaveLength(20)
    expect(result.contributions.length).toBeGreaterThan(0)
    expect(result.contributions.length).toBeLessThanOrEqual(6)
    // contributions are sorted by descending magnitude
    for (let i = 1; i < result.contributions.length; i += 1) {
      expect(Math.abs(result.contributions[i - 1].contribution)).toBeGreaterThanOrEqual(
        Math.abs(result.contributions[i].contribution),
      )
    }
  })

  it('maps scores to the right band thresholds', () => {
    // Band edges are < 35 Safe, 35-65 Suspicious, >= 65 Dangerous.
    for (const entry of loadParityFixture().urls) {
      const result = classify(entry.url)
      if (result.finalScore >= 65) {
        expect(result.band).toBe('Dangerous')
      } else if (result.finalScore >= 35) {
        expect(result.band).toBe('Suspicious')
      } else {
        expect(result.band).toBe('Safe')
      }
    }
  })
})

describe('model metadata', () => {
  it('exposes 20 aligned parameter arrays', () => {
    expect(MODEL.featureNames).toHaveLength(20)
    expect(MODEL.coef).toHaveLength(20)
    expect(MODEL.mean).toHaveLength(20)
    expect(MODEL.scale).toHaveLength(20)
  })

  it('reports plausible held-out metrics', () => {
    expect(MODEL.metrics.accuracy).toBeGreaterThan(0.7)
    expect(MODEL.metrics.roc_auc).toBeGreaterThan(0.8)
  })
})

describe('parity with fixtures/parity_urls.json', () => {
  const fixture = loadParityFixture()

  it('reproduces the ML probability for every fixture URL', () => {
    for (const entry of fixture.urls) {
      const result = classify(entry.url)
      expect(result.mlProbability, `${entry.url} ml_probability`).toBeCloseTo(
        entry.ml_probability,
        7,
      )
    }
  })

  it('reproduces the blended final score and band for every fixture URL', () => {
    for (const entry of fixture.urls) {
      const result = classify(entry.url)
      expect(result.finalScore, `${entry.url} final_score`).toBeCloseTo(
        entry.final_score,
        6,
      )
      expect(result.band, `${entry.url} band`).toBe(entry.band)
    }
  })

  it('reproduces the heuristic score for every fixture URL', () => {
    for (const entry of fixture.urls) {
      expect(classify(entry.url).heuristicScore, entry.url).toBe(
        entry.heuristic_score,
      )
    }
  })
})
