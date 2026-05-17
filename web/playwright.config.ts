import { defineConfig, devices } from '@playwright/test'

// Playwright runs against the *built and previewed* site (not `vite dev`) so
// that base-path bugs and production-bundle issues surface in CI.
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:4173/phishing-url-detector/',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'npm run build && npm run preview',
    url: 'http://localhost:4173/phishing-url-detector/',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
