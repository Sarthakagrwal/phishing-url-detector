/**
 * Capture a documentation screenshot of the PhishGuard page.
 *
 * Not a test — a one-off helper run after the site is built. It loads the
 * previewed production build, analyses a phishing-style URL so the verdict and
 * breakdown are visible, and saves the result to docs/screenshot.png.
 *
 * Usage (with the preview server already running on :4173):
 *   node e2e/screenshot.mjs
 */

import { chromium } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const BASE = 'http://localhost:4173/phishing-url-detector/'
const here = dirname(fileURLToPath(import.meta.url))
const outPath = resolve(here, '..', '..', 'docs', 'screenshot.png')

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })

await page.goto(BASE, { waitUntil: 'networkidle' })

// Analyse a phishing-style URL so the screenshot shows a populated verdict.
await page
  .locator('#url-input')
  .fill('http://secure-login-verify-account.paypal-update.gq/webscr/signin')
await page.locator('#analyze-btn').click()
await page.locator('[data-testid="verdict"]').waitFor({ state: 'visible' })
await page.waitForTimeout(600) // let the meter animation settle

// Return to the top so the sticky site header is captured in place — a
// full-page screenshot taken while scrolled strands sticky elements mid-page.
await page.evaluate(() => window.scrollTo(0, 0))
await page.waitForTimeout(300)

await page.screenshot({ path: outPath, fullPage: true })
await browser.close()

console.log(`screenshot saved to ${outPath}`)
