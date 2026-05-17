/**
 * PhishGuard — application entry point.
 *
 * Builds the page, wires the Analyse button + example chips, and renders the
 * classification result. Everything runs client-side: classification calls
 * `predict.ts`, which never makes a network request — the URL stays on-device.
 */

import './styles/theme.css'
import './styles/app.css'

import { MODEL } from './generated/model'
import { classify } from './predict'
import {
  renderContributions,
  renderFeatureTable,
  renderReasons,
  renderVerdict,
} from './ui'

const GITHUB_URL = 'https://github.com/Sarthakagrwal/phishing-url-detector'

/** One-click example URLs: a mix of safe and phishing-style links. */
const EXAMPLES: ReadonlyArray<{ url: string; kind: 'safe' | 'danger'; label: string }> =
  [
    { url: 'https://github.com/anthropics', kind: 'safe', label: 'github.com' },
    {
      url: 'https://en.wikipedia.org/wiki/Phishing',
      kind: 'safe',
      label: 'wikipedia.org',
    },
    {
      url: 'http://secure-login-update-account.verify-paypal.gq/webscr',
      kind: 'danger',
      label: 'fake PayPal login',
    },
    {
      url: 'http://192.168.10.4/account/verify.php',
      kind: 'danger',
      label: 'raw-IP login page',
    },
    {
      url: 'http://www.apple.com@198.51.100.23/signin',
      kind: 'danger',
      label: "'@' redirect trick",
    },
    {
      url: 'http://xn--pple-43d.com/id/login',
      kind: 'danger',
      label: 'punycode look-alike',
    },
  ]

/** Build the full static page markup. */
function pageMarkup(): string {
  const m = MODEL.metrics
  const examplesHtml = EXAMPLES.map(
    (e) =>
      `<button class="chip chip--${e.kind}" data-url="${e.url}" type="button">${e.label}</button>`,
  ).join('')

  return `
    <header class="site-header">
      <div class="wrap site-header__inner">
        <div class="site-header__brand">
          <span class="logo">P</span>
          <span>PhishGuard</span>
        </div>
        <nav class="site-header__nav">
          <a href="#how">How it works</a>
          <a href="${GITHUB_URL}" target="_blank" rel="noopener">GitHub</a>
        </nav>
      </div>
    </header>

    <main class="wrap">
      <section class="hero">
        <span class="eyebrow">URL phishing analysis</span>
        <h1>Is this link safe to click?</h1>
        <p class="lede">
          Paste any URL below. PhishGuard scores its phishing risk by combining
          transparent heuristic rules with a logistic-regression model trained
          on ${'~'}235k real URLs &mdash; running entirely in your browser.
        </p>
      </section>

      <section class="card card--pad-lg">
        <div class="field">
          <label for="url-input">URL to analyse</label>
          <div class="analyze-form">
            <input
              id="url-input"
              type="text"
              class="input--mono"
              placeholder="https://example.com/login"
              autocomplete="off"
              autocapitalize="off"
              spellcheck="false"
            />
            <button id="analyze-btn" class="btn btn--primary" type="button">
              Analyse URL
            </button>
          </div>
          <div class="form-error" id="form-error" role="alert"></div>
        </div>

        <div class="examples mt-3">
          <span class="examples__label">Try an example:</span>
          ${examplesHtml}
        </div>

        <div class="privacy-note mt-4">
          <span aria-hidden="true">&#128274;</span>
          <span>
            <strong>Private by design.</strong> The URL is analysed entirely on
            your device. Nothing is sent to any server &mdash; there are zero
            network requests after the page loads.
          </span>
        </div>
      </section>

      <section id="results" class="mt-5" hidden>
        <div class="card card--pad-lg">
          <div class="card__title">Verdict</div>
          <div class="verdict-block" id="verdict-area"></div>
        </div>

        <div class="grid grid-2">
          <div class="card">
            <div class="card__title">Why &mdash; heuristic signals</div>
            <div id="reasons-area"></div>
          </div>
          <div class="card">
            <div class="card__title">Top model contributions</div>
            <p class="muted" style="font-size:0.82rem;margin-bottom:var(--sp-3)">
              How each feature pushed the model. Red bars push toward phishing,
              green toward safe.
            </p>
            <div id="contrib-area"></div>
          </div>
        </div>

        <div class="card">
          <div class="card__title">Full feature breakdown (20 string-only features)</div>
          <div id="feature-area"></div>
        </div>
      </section>

      <section id="how" class="card card--pad-lg mt-5">
        <h2>How it works</h2>
        <p class="muted mt-3" style="margin-bottom:var(--sp-4)">
          PhishGuard never visits the URL or resolves DNS. Every signal is
          derived from the URL <em>string</em> alone, which is what lets the
          identical analysis run offline in your browser.
        </p>
        <ol class="how-list">
          <li>
            <span>
              <strong>Parse.</strong> A deterministic parser splits the URL into
              scheme, host, path and query &mdash; matching the Python
              implementation exactly so results are identical everywhere.
            </span>
          </li>
          <li>
            <span>
              <strong>Extract 20 features.</strong> Length, dots, hyphens,
              subdomain depth, raw-IP host, <code>@</code> tricks, Punycode,
              suspicious TLDs, shorteners, phishing keywords and more.
            </span>
          </li>
          <li>
            <span>
              <strong>Run the heuristics.</strong> A transparent rule engine
              flags known phishing tells and explains each one in plain English.
            </span>
          </li>
          <li>
            <span>
              <strong>Run the model.</strong> A logistic-regression classifier
              &mdash; the same model as the Python CLI, evaluated here as a
              closed-form sigmoid &mdash; estimates a phishing probability.
            </span>
          </li>
          <li>
            <span>
              <strong>Blend.</strong> The final score is 60% model + 40%
              heuristics, mapped to a Safe / Suspicious / Dangerous band.
            </span>
          </li>
        </ol>
      </section>
    </main>

    <footer class="site-footer">
      <div class="wrap">
        <p>
          Built by Sarthak Aggarwal as part of a cybersecurity learning portfolio.
          Model: logistic regression &mdash; held-out accuracy
          ${(m.accuracy * 100).toFixed(1)}%, precision
          ${(m.precision * 100).toFixed(1)}%, recall
          ${(m.recall * 100).toFixed(1)}%, ROC-AUC ${m.roc_auc.toFixed(3)}.
        </p>
        <p class="mt-3">
          Trained on the
          <a
            href="https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset"
            target="_blank"
            rel="noopener"
            >PhiUSIIL Phishing URL Dataset</a
          >
          (UCI Machine Learning Repository, CC BY 4.0). PhishGuard is an
          educational tool and not a guarantee of safety.
        </p>
      </div>
    </footer>
  `
}

/**
 * Run a classification for `rawUrl` and paint the results section.
 * Returns true on success, false if the input was empty/invalid.
 */
function runAnalysis(rawUrl: string): boolean {
  const errorEl = document.getElementById('form-error') as HTMLElement
  const resultsEl = document.getElementById('results') as HTMLElement

  const trimmed = rawUrl.trim()
  if (trimmed === '') {
    errorEl.textContent = 'Please enter a URL to analyse.'
    resultsEl.hidden = true
    return false
  }
  if (/\s/.test(trimmed)) {
    errorEl.textContent = 'A URL cannot contain spaces — please check the input.'
    resultsEl.hidden = true
    return false
  }
  errorEl.textContent = ''

  const result = classify(trimmed)

  ;(document.getElementById('verdict-area') as HTMLElement).innerHTML =
    renderVerdict(result)
  ;(document.getElementById('reasons-area') as HTMLElement).innerHTML =
    renderReasons(result)
  ;(document.getElementById('contrib-area') as HTMLElement).innerHTML =
    renderContributions(result.contributions)
  ;(document.getElementById('feature-area') as HTMLElement).innerHTML =
    renderFeatureTable(result.features)

  resultsEl.hidden = false
  resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  return true
}

/** Mount the app: render markup and attach event handlers. */
function mount(): void {
  const app = document.getElementById('app')
  if (app === null) {
    throw new Error('PhishGuard: #app mount point not found.')
  }
  app.innerHTML = pageMarkup()

  const input = document.getElementById('url-input') as HTMLInputElement
  const button = document.getElementById('analyze-btn') as HTMLButtonElement

  button.addEventListener('click', () => {
    runAnalysis(input.value)
  })

  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      runAnalysis(input.value)
    }
  })

  // Example chips: fill the input and analyse immediately.
  for (const chip of document.querySelectorAll<HTMLButtonElement>('.chip')) {
    chip.addEventListener('click', () => {
      const url = chip.dataset.url ?? ''
      input.value = url
      runAnalysis(url)
    })
  }
}

mount()
