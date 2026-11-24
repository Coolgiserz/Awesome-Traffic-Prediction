// @ts-check
import { defineConfig } from 'astro/config';

/**
 * Deployment knobs are environment-driven so the same config works for:
 *   - local dev / preview      -> SITE_BASE default '/'
 *   - GitHub Project Pages     -> SITE_BASE=/Awesome-Traffic-Prediction/
 *   - custom domain / user page-> SITE_BASE=/
 *
 * All internal links go through src/lib/url.ts (withBase) so nothing hardcodes
 * the repo prefix. See docs/evolution/04-pages-deploy.md.
 */
const SITE_URL = process.env.SITE_URL ?? 'https://coolgiserz.github.io';
const SITE_BASE = process.env.SITE_BASE ?? '/Awesome-Traffic-Prediction/';

export default defineConfig({
  site: SITE_URL,
  base: SITE_BASE,
  trailingSlash: 'always',
  // Static output only: GitHub Pages serves files, there is no Node runtime.
  output: 'static',
  build: {
    // Emit /papers/index.html instead of /papers.html so GitHub Pages resolves
    // the directory URL without a redirect hop.
    format: 'directory',
    inlineStylesheets: 'auto',
  },
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh'],
    routing: {
      // '/' serves English, '/zh/...' serves Chinese. 'prefix-other-locales'
      // keeps English URLs clean while giving Chinese a shareable prefix.
      prefixDefaultLocale: false,
      redirectToDefaultLocale: false,
    },
  },
  markdown: {
    shikiConfig: { theme: 'github-light' },
  },
  devToolbar: { enabled: false },
  vite: {
    build: {
      // Keep the client bundle small: ECharts is dynamically imported only on
      // pages that actually draw a chart (see src/components/TrendChart.astro).
      chunkSizeWarningLimit: 900,
    },
  },
});