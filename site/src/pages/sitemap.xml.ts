/**
 * src/pages/sitemap.xml.ts — hand-rolled sitemap (no @astrojs/sitemap dep).
 *
 * @astrojs/sitemap is not used because it does not understand the two-locale
 * route split (`/` vs `/zh/`) nor the Project Pages base path, and it would
 * need an absolute-URL fix-up pass anyway.
 *
 * Since Phase 2 the sitemap also lists every paper detail page
 * (`/papers/<id>/` and `/zh/papers/<id>/`), each with xhtml:link alternates so
 * Google indexes the pair as one bilingual document instead of two duplicates.
 */
import type { APIRoute } from 'astro';
import { routes, languages } from '../lib/i18n';
import { withBase } from '../lib/url';
import { papers } from '../lib/papers';

const SITE = (import.meta.env.SITE ?? 'https://coolgiserz.github.io');

export const GET: APIRoute = ({ site }) => {
  const base = site?.href ?? `${SITE}/`;
  const abs = (p: string) => new URL(p, base).href;

  type Entry = { loc: string; alt?: { en: string; zh: string }; priority: string };
  const entries: Entry[] = [];

  for (const route of Object.values(routes)) {
    const en = abs(withBase(route));
    const zh = abs(withBase(`/zh${route}`));
    entries.push({ loc: en, alt: { en, zh }, priority: route === '/' ? '1.0' : '0.7' });
    entries.push({ loc: zh, alt: { en, zh }, priority: route === '/' ? '1.0' : '0.7' });
  }
  entries.push({ loc: abs(withBase('/search/')), priority: '0.5' });

  // Detail pages: 104 papers x 2 locales. Priority is higher than section pages
  // because a paper page is the citable unit and the entry point from search.
  for (const p of papers) {
    const en = abs(withBase(`/papers/${p.id}/`));
    const zh = abs(withBase(`/zh/papers/${p.id}/`));
    entries.push({ loc: en, alt: { en, zh }, priority: '0.6' });
    entries.push({ loc: zh, alt: { en, zh }, priority: '0.6' });
  }

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml">
${entries
  .map(
    (e) => `  <url>
    <loc>${e.loc}</loc>${
      e.alt
        ? `
    <xhtml:link rel="alternate" hreflang="en" href="${e.alt.en}"/>
    <xhtml:link rel="alternate" hreflang="zh-CN" href="${e.alt.zh}"/>
    <xhtml:link rel="alternate" hreflang="x-default" href="${e.alt.en}"/>`
        : ''
    }
    <changefreq>weekly</changefreq>
    <priority>${e.priority}</priority>
  </url>`,
  )
  .join('\n')}
</urlset>
`;
  return new Response(body, {
    headers: { 'Content-Type': 'application/xml; charset=utf-8' },
  });
};