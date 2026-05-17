/// <reference types="node" />
import { defineConfig } from 'vitest/config'

// `base` MUST be the repo name for GitHub Project Pages, or the deployed site
// loads with broken asset paths and renders blank.
//
// `defineConfig` is imported from `vitest/config` (a superset of Vite's own)
// so the `test` block is type-checked. The build output is unaffected.
export default defineConfig({
  base: '/phishing-url-detector/',
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.test.ts'],
  },
})
