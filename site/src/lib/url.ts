/**
 * url.ts — the ONLY place that knows about the GitHub Pages base path.
 *
 * Astro's `base` is injected at build time; hardcoding '/foo/' anywhere else
 * breaks either local dev or Project Pages. Always wrap internal hrefs.
 */
const RAW_BASE = import.meta.env.BASE_URL || '/';

/** Normalized base without trailing slash, '' when served from domain root. */
export const BASE = RAW_BASE.endsWith('/') ? RAW_BASE.slice(0, -1) : RAW_BASE;

/** Prefix an internal absolute path with the deployment base. */
export function withBase(p = '/') {
  const clean = p.startsWith('/') ? p : `/${p}`;
  return `${BASE}${clean}`;
}

/** Locale-aware route builder. en is unprefixed, zh lives under /zh/. */
export function localePath(lang, p = '/') {
  const clean = p.startsWith('/') ? p : `/${p}`;
  return lang === 'zh' ? withBase(`/zh${clean === '/' ? '/' : clean}`) : withBase(clean);
}

/** Mirror the current pathname into the other locale (used by the lang switch). */
export function altLocalePath(pathname, targetLang) {
  const base = BASE || '';
  let rel = pathname.startsWith(base) ? pathname.slice(base.length) : pathname;
  if (!rel.startsWith('/')) rel = `/${rel}`;
  const isZh = rel === '/zh' || rel.startsWith('/zh/');
  let bare = isZh ? rel.slice(3) || '/' : rel;
  if (!bare.startsWith('/')) bare = `/${bare}`;
  if (targetLang === 'zh') return `${base}/zh${bare === '/' ? '/' : bare}`;
  return `${base}${bare}`;
}