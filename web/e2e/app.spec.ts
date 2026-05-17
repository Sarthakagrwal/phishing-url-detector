/**
 * Playwright end-to-end tests for the PhishGuard site.
 *
 * Runs against the built + previewed production bundle (see playwright.config.ts)
 * so base-path and bundling issues surface. The final test intercepts every
 * request to prove the page is fully client-side — the central privacy claim.
 */

import { expect, test } from '@playwright/test'

test.describe('PhishGuard UI', () => {
  test('loads with the brand and the analyse form', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('.site-header__brand')).toContainText('PhishGuard')
    await expect(page.locator('#url-input')).toBeVisible()
    await expect(page.locator('#analyze-btn')).toBeVisible()
    // results are hidden until an analysis is run
    await expect(page.locator('#results')).toBeHidden()
  })

  test('a phishing-style URL produces a Dangerous verdict', async ({ page }) => {
    await page.goto('/')
    await page
      .locator('#url-input')
      .fill('http://secure-login-verify-account.paypal-update.gq/webscr/signin')
    await page.locator('#analyze-btn').click()

    await expect(page.locator('[data-testid="verdict"]')).toBeVisible()
    await expect(page.locator('[data-testid="verdict-band"]')).toHaveText(
      'Dangerous',
    )
    await expect(page.locator('.verdict')).toHaveClass(/verdict--danger/)
  })

  test('a well-known HTTPS site produces a Safe verdict', async ({ page }) => {
    await page.goto('/')
    await page.locator('#url-input').fill('https://github.com')
    await page.locator('#analyze-btn').click()

    await expect(page.locator('[data-testid="verdict-band"]')).toHaveText('Safe')
    await expect(page.locator('.verdict')).toHaveClass(/verdict--safe/)
  })

  test('the feature breakdown table populates with all 20 features', async ({
    page,
  }) => {
    await page.goto('/')
    await page.locator('#url-input').fill('http://192.168.1.1/login/verify')
    await page.locator('#analyze-btn').click()

    const rows = page.locator('[data-testid="feature-table"] tbody tr')
    await expect(rows).toHaveCount(20)
    // the raw-IP row must show the risky "Yes" flag
    await expect(
      page.locator('[data-testid="feature-table"]'),
    ).toContainText('Host is a raw IP')
  })

  test('the heuristic reasons and ML contributions render', async ({ page }) => {
    await page.goto('/')
    await page.locator('#url-input').fill('http://203.0.113.9/account/verify.php')
    await page.locator('#analyze-btn').click()

    // at least one heuristic reason is shown for a raw-IP phishing URL
    await expect(
      page.locator('[data-testid="reasons"] li').first(),
    ).toBeVisible()
    await expect(page.locator('[data-testid="reasons"]')).toContainText(
      'raw IP address',
    )
    // the ML contribution bars render
    await expect(
      page.locator('[data-testid="contributions"] .contrib__row').first(),
    ).toBeVisible()
  })

  test('clicking an example chip produces a verdict', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('#results')).toBeHidden()

    // click the first phishing-style example chip
    await page.locator('.chip--danger').first().click()

    await expect(page.locator('[data-testid="verdict"]')).toBeVisible()
    await expect(page.locator('[data-testid="verdict-band"]')).toHaveText(
      'Dangerous',
    )
    // the input was filled by the chip
    await expect(page.locator('#url-input')).not.toHaveValue('')
  })

  test('an empty input shows an inline error', async ({ page }) => {
    await page.goto('/')
    await page.locator('#analyze-btn').click()
    await expect(page.locator('#form-error')).toContainText('enter a URL')
    await expect(page.locator('#results')).toBeHidden()
  })

  test('makes ZERO network requests during analysis (fully client-side)', async ({
    page,
  }) => {
    // Record every request the page issues AFTER the initial document/bundle
    // has loaded. A truly client-side tool must make none.
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const requestsAfterLoad: string[] = []
    page.on('request', (request) => {
      requestsAfterLoad.push(request.url())
    })

    // Run several analyses, including an example chip.
    await page.locator('#url-input').fill('http://login-verify-account.gq/webscr')
    await page.locator('#analyze-btn').click()
    await expect(page.locator('[data-testid="verdict"]')).toBeVisible()

    await page.locator('.chip--safe').first().click()
    await expect(page.locator('[data-testid="verdict"]')).toBeVisible()

    await page.locator('#url-input').fill('https://example.com/some/deep/path')
    await page.locator('#analyze-btn').click()
    await expect(page.locator('[data-testid="verdict"]')).toBeVisible()

    // Give any stray async request a chance to appear, then assert none did.
    await page.waitForTimeout(500)
    expect(
      requestsAfterLoad,
      `unexpected network requests: ${requestsAfterLoad.join(', ')}`,
    ).toHaveLength(0)
  })
})
