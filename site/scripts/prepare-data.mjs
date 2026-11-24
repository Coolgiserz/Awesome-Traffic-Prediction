#!/usr/bin/env node
/**
 * prepare-data.mjs — build-time data bridge (runs in `prebuild`).
 *
 * Three data sources, deliberately ordered:
 *   1. PREFERRED  data/papers/*.json      (structured, produced by the data track)
 *   2. FALLBACK   README.md / README.zh.md (markdown scrape, always available)
 *
 * Structured records carry far more than the README scrape can recover
 * (doi, authors, venue_type, layers, direction_tags, verification), so every
 * field is passed through to the generated payload instead of being flattened
 * to the lowest common shape.
 *
 * Nothing in this script writes outside site/. It reads the repo root read-only
 * and emits:
 *   - src/data/papers.generated.json  papers + source provenance
 *   - public/_data.json               chart payload (hotspots/wordcloud/venues)
 *
 * Exit codes: 0 ok (possibly degraded), 1 unrecoverable (no data at all).
 */
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SITE_DIR = path.resolve(HERE, '..');
const REPO_DIR = path.resolve(SITE_DIR, '..');
const FORCE = process.argv.includes('--force');

const log = (...a) => console.log('[prepare-data]', ...a);

/* ------------------------------------------------------------------ utils */

async function readJsonSafe(p) {
  try {
    return JSON.parse(await readFile(p, 'utf8'));
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    log(`WARN cannot parse ${path.relative(REPO_DIR, p)}: ${err.message}`);
    return null;
  }
}

function slugify(input) {
  return String(input)
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80);
}

/** Stable content hash used to skip regeneration when nothing changed. */
function fingerprint(obj) {
  const s = JSON.stringify(obj);
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16);
}

/* ------------------------------------------------ source 1: data/papers/*.json */

/**
 * Structured record -> site payload.
 *
 * Every field the site renders must survive this function. Fields absent from
 * the structured schema fall back to the same heuristics the README scrape uses,
 * so a partially migrated year still renders sensibly.
 */
function normalizePaper(item, file) {
  const title = item.title ?? item.name;
  if (!title) return null;
  const year = Number(item.year ?? 0) || null;
  const tags = item.method_tags ?? item.methodTags ?? item.tags ?? [];
  const codeLinks = Array.isArray(item.code_links) ? item.code_links : [];
  const altUrls = item.alt_urls && typeof item.alt_urls === 'object' ? item.alt_urls : {};

  return {
    id: item.id ?? slugify(`${title}-${year ?? 'x'}`),
    lang: item.lang ?? null,
    year,
    altYears: Array.isArray(item.alt_years) ? item.alt_years : [],
    venue: item.venue ?? 'Unknown',
    venueId: item.venue_id ?? null,
    venueType: item.venue_type ?? 'unknown',
    // Short display name: explicit acronym > curated short name > title head.
    shortName: item.short_name ?? title.split(':')[0].trim().slice(0, 42),
    title,
    titleZh: item.title_zh ?? null,
    paperUrl: item.paper_url ?? item.paperUrl ?? null,
    altUrls,
    codeUrl: item.code_url ?? item.codeUrl ?? null,
    codeLinks,
    dataUrl: item.data_url ?? item.dataUrl ?? null,
    // Legacy records used `review` / `sota` as extra links, not always `review`.
    reviewUrl: item.review_url ?? item.reviewUrl ?? altUrls.review ?? null,
    projectUrl: item.project_url ?? null,
    doi: item.doi ?? null,
    arxivId: item.arxiv_id ?? null,
    openalexId: item.openalex_id ?? null,
    authors: Array.isArray(item.authors) ? item.authors : [],
    datasets: Array.isArray(item.datasets) ? item.datasets : [],
    layers: Array.isArray(item.layers) ? item.layers : [],
    directionTags: Array.isArray(item.direction_tags) ? item.direction_tags : [],
    badges: [],
    section: item.section ?? 'deep_learning',
    track: item.track ?? item.section ?? 'deep_learning',
    task: item.task_en ?? item.task ?? detectTask(title),
    taskZh: item.task_zh ?? null,
    methodTags: Array.isArray(tags) ? tags : [],
    note: item.note_en ?? item.note ?? null,
    noteZh: item.note_zh ?? null,
    featured: Boolean(item.featured ?? item.in_digest),
    status: item.status ?? 'active',
    verification: item.verification ?? null,
    provenance: item.provenance ?? null,
    source: `data/papers/${file}`,
  };
}

async function loadStructuredPapers() {
  const dir = path.join(REPO_DIR, 'data', 'papers');
  if (!existsSync(dir)) return null;
  const files = (await readdir(dir)).filter((f) => f.endsWith('.json')).sort();
  if (files.length === 0) return null;

  const papers = [];
  for (const f of files) {
    const raw = await readJsonSafe(path.join(dir, f));
    if (!raw) continue;
    const arr = Array.isArray(raw) ? raw : raw.papers ?? raw.items ?? [];
    for (const item of arr) {
      const p = normalizePaper(item, f);
      if (p) papers.push(p);
    }
  }
  if (papers.length === 0) return null;

  // Integrity guard: data/papers is built incrementally by the data track, so a
  // single partially-migrated year file must not silently shrink the site.
  // Compare coverage with the README entry count before trusting it
  // (SITE_MIN_STRUCTURED_PAPERS override).
  const readmeCount = (await scrapeReadme('README.md', 'en')).length;
  const minExpected = Number(process.env.SITE_MIN_STRUCTURED_PAPERS ?? 0) ||
    Math.max(1, Math.floor(readmeCount * 0.75));

  if (papers.length < minExpected) {
    log(
      `REJECT data/papers: ${papers.length} papers < expected >=${minExpected} ` +
        `(README has ${readmeCount}); falling back to README scrape`,
    );
    return null;
  }

  const noUrl = papers.filter((p) => !p.paperUrl).length;
  if (noUrl > papers.length * 0.1) {
    log(`WARN data/papers: ${noUrl}/${papers.length} records missing paper_url`);
  }

  log(`source=data/papers (${files.length} file(s), ${papers.length} papers)`);
  return { papers, provenance: { source: 'data/papers', files, readmeCount } };
}

/* ------------------------------------------ source 2: scrape README*.md */

const LINK_RE = /\[\[([a-zA-Z0-9_-]+)\]\]\((https?:\/\/[^)\s]+)\)/g;

/**
 * README entry grammar (see AGENTS.md):
 *   [Venue Year] Title [[paper]](url) [[code]](url) [[data]](url) ...
 * Some entries put the title itself in markdown link form instead of a
 * leading [Venue Year] tag; both shapes are supported.
 */
function parseLine(line) {
  const trimmed = line.trim();
  const m = /^\d+\.\s+(.*)$/.exec(trimmed);
  if (!m) return null;
  let body = m[1].trim();
  if (!body || body.startsWith('……') || body.startsWith('...')) return null;

  const links = {};
  LINK_RE.lastIndex = 0;
  let lm;
  while ((lm = LINK_RE.exec(body)) !== null) {
    const key = lm[1].toLowerCase();
    // First hit wins for each key; keep explicit -official/-pytorch variants.
    if (!links[key]) links[key] = lm[2];
  }
  body = body.replace(LINK_RE, '').trim();

  // Leading "[Venue Year]" tag.
  let venue = null;
  let year = null;
  const tag = /^\[([^\]]{2,60}?)\s+(\d{4})\]\s*/.exec(body);
  if (tag) {
    venue = tag[1].trim();
    year = Number(tag[2]);
    body = body.slice(tag[0].length).trim();
  }

  // Title: either a markdown link or the remaining prose.
  let title = body;
  let paperUrl = links.paper ?? null;
  const titleLink = /^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/.exec(body);
  if (titleLink) {
    title = titleLink[1].trim();
    paperUrl = paperUrl ?? titleLink[2];
  }
  title = title
    .replace(/\[\[[^\]]*\]\]\([^)]*\)/g, '')
    .replace(/\s+/g, ' ')
    .replace(/[.\s]+$/, '')
    .trim();
  if (!title || title.length < 8) return null;

  return { venue, year, title, links, paperUrl };
}

function detectTags(title, venue) {
  const t = `${title} ${venue ?? ''}`.toLowerCase();
  const tags = new Set();
  const rules = [
    ['STGNN', /\b(gcn|graph (convolution|neural|attention|wavenet|transformer)|stgnn|spatio-?temporal graph|gnn)\b/],
    ['Transformer', /\btransformer|attention\b/],
    ['Foundation Model', /\b(foundation model|pre-?train(ing)?|universal model|prompt)\b/],
    ['Diffusion', /\bdiffusion\b/],
    ['Gan', /\bgan\b|generative adversarial/],
    ['Federated Learning', /\bfederated\b/],
    ['Few-Shot', /\b(few-?shot|meta-?learning|transfer)\b/],
    ['Probabilistic', /\b(probabilistic|uncertainty|bayesian)\b/],
    ['Imputation', /\b(imputation|missing value)\b/],
    ['Trajectory', /\btrajector/],
  ];
  for (const [tag, re] of rules) if (re.test(t)) tags.add(tag);
  return [...tags];
}

function detectTask(title) {
  const t = title.toLowerCase();
  if (/trajector/.test(t)) return 'trajectory-prediction';
  if (/travel time|eta\b/.test(t)) return 'travel-time-estimation';
  if (/(bike|taxi|demand|order|od\b)/.test(t)) return 'demand-prediction';
  if (/nowcast|precipitation/.test(t)) return 'nowcasting';
  if (/flow/.test(t)) return 'traffic-flow';
  if (/speed/.test(t)) return 'traffic-speed';
  if (/survey|review|overview/.test(t)) return 'survey';
  return 'traffic-forecasting';
}

async function scrapeReadme(file, lang) {
  const p = path.join(REPO_DIR, file);
  if (!existsSync(p)) return [];
  const text = await readFile(p, 'utf8');
  const lines = text.split(/\r?\n/);

  const papers = [];
  let section = 'misc';
  let sub = null;
  let currentYear = null;
  let inAuto = false;

  for (const line of lines) {
    if (line.trim() === '<!-- START AUTO RECENT PAPERS -->') {
      inAuto = true;
      continue;
    }
    if (line.trim() === '<!-- END AUTO RECENT PAPERS -->') {
      inAuto = false;
      continue;
    }
    const h2 = /^##\s+(.*)$/.exec(line);
    if (h2) {
      section = h2[1].trim();
      sub = null;
      currentYear = null;
      continue;
    }
    const h3 = /^###\s+(.*)$/.exec(line);
    if (h3) {
      sub = h3[1].trim();
      currentYear = null;
      continue;
    }
    const h4 = /^####\s+(\d{4})\s*$/.exec(line);
    if (h4) {
      currentYear = Number(h4[1]);
      continue;
    }
    // Only the paper-bearing sections are scraped; datasets/toolkits have no
    // venue-year tag and would pollute the paper table.
    if (!/0x00|papers/i.test(section)) continue;
    if (inAuto) continue;

    const parsed = parseLine(line);
    if (!parsed) continue;

    const year = parsed.year ?? currentYear ?? null;
    const id = slugify(`${parsed.title}-${year ?? 'x'}`);
    papers.push({
      id,
      lang,
      year,
      altYears: [],
      venue: parsed.venue ?? 'Unknown',
      venueId: null,
      venueType: 'unknown',
      shortName: parsed.title.split(':')[0].trim().slice(0, 42),
      title: parsed.title,
      titleZh: null,
      paperUrl: parsed.paperUrl,
      altUrls: {},
      codeUrl: parsed.links.code ?? parsed.links['code-official'] ?? parsed.links['code-pytorch'] ?? null,
      codeLinks: [],
      dataUrl: parsed.links.data ?? null,
      reviewUrl: parsed.links.review ?? parsed.links.sota ?? null,
      projectUrl: null,
      doi: null,
      arxivId: null,
      openalexId: null,
      authors: [],
      datasets: [],
      layers: [],
      directionTags: [],
      badges: Object.keys(parsed.links).filter((k) => !k.startsWith('paper')),
      section,
      track: sub ?? section,
      task: detectTask(parsed.title),
      taskZh: null,
      methodTags: detectTags(parsed.title, parsed.venue),
      note: null,
      noteZh: null,
      featured: false,
      status: 'active',
      verification: null,
      provenance: null,
      source: file,
    });
  }
  return papers;
}

/* ----------------------------------------------------- chart data bundle */

async function buildChartData(papers) {
  const yearly = await readJsonSafe(
    path.join(REPO_DIR, 'updates/hotspots/yearly-hotspots.json'),
  );
  const wcDir = path.join(REPO_DIR, 'updates/wordcloud');
  let wordcloud = null;
  if (existsSync(wcDir)) {
    const files = (await readdir(wcDir))
      .filter((f) => f.endsWith('.json'))
      .sort()
      .reverse();
    if (files.length) wordcloud = await readJsonSafe(path.join(wcDir, files[0]));
    wordcloud = wordcloud ? { ...wordcloud, file: files[0] } : null;
  }

  const byYear = new Map();
  const byVenue = new Map();
  const byTask = new Map();
  const byLayer = new Map();
  const byDirection = new Map();
  for (const p of papers) {
    if (p.year) byYear.set(p.year, (byYear.get(p.year) ?? 0) + 1);
    if (p.venue) byVenue.set(p.venue, (byVenue.get(p.venue) ?? 0) + 1);
    if (p.task) byTask.set(p.task, (byTask.get(p.task) ?? 0) + 1);
    for (const l of p.layers ?? []) byLayer.set(l, (byLayer.get(l) ?? 0) + 1);
    for (const d of p.directionTags ?? []) byDirection.set(d, (byDirection.get(d) ?? 0) + 1);
  }
  const top = (m, n = 15) =>
    [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, n);

  return {
    generatedUtc: new Date().toISOString(),
    papersByYear: [...byYear.entries()].sort((a, b) => a[0] - b[0]),
    topVenues: top(byVenue),
    taskCounts: [...byTask.entries()].sort((a, b) => b[1] - a[1]),
    layerCounts: top(byLayer),
    directionCounts: top(byDirection, 20),
    hotspots: yearly
      ? { generatedUtc: yearly.generated_utc, topk: yearly.topk, years: yearly.years }
      : null,
    wordcloud: wordcloud
      ? {
          generatedUtc: wordcloud.generated_utc,
          file: wordcloud.file,
          topWords: wordcloud.top_words ?? [],
        }
      : null,
  };
}

/* ---------------------------------------------------------------- main */

async function main() {
  const structured = await loadStructuredPapers();

  let papers;
  let provenance;
  if (structured) {
    papers = structured.papers;
    provenance = structured.provenance;
  } else {
    log('data/papers/*.json not found -> falling back to README scrape');
    const en = await scrapeReadme('README.md', 'en');
    const zh = await scrapeReadme('README.zh.md', 'zh');
    // README.md is the canonical entry list; zh only fills in extra entries and
    // keeps the bilingual labels for the same line position.
    papers = en.length ? en : zh;
    provenance = {
      source: 'README.md',
      files: ['README.md', 'README.zh.md'],
      en: en.length,
      zh: zh.length,
      degraded: true,
    };
  }

  if (papers.length === 0) {
    console.error('[prepare-data] FATAL no papers from any source');
    process.exit(1);
  }

  const shaped = {
    generatedUtc: new Date().toISOString(),
    provenance,
    count: papers.length,
    papers,
  };

  const outJson = path.join(SITE_DIR, 'src/data/papers.generated.json');
  await mkdir(path.dirname(outJson), { recursive: true });

  const prev = await readJsonSafe(outJson);
  const same = prev && fingerprint(prev.papers) === fingerprint(shaped.papers);
  // Provenance is compared too: a source switch must not be hidden by an
  // identical paper list (the footer and the degraded banner depend on it).
  const sameProvenance =
    prev && fingerprint(prev.provenance ?? {}) === fingerprint(shaped.provenance);
  if (same && sameProvenance && !FORCE) {
    log(`unchanged (${papers.length} papers) -> skip write`);
  } else {
    await writeFile(outJson, JSON.stringify(shaped, null, 2) + '\n', 'utf8');
    log(`wrote ${path.relative(REPO_DIR, outJson)} (${papers.length} papers)`);
  }

  const charts = await buildChartData(papers);
  const outCharts = path.join(SITE_DIR, 'public/_data.json');
  await mkdir(path.dirname(outCharts), { recursive: true });
  await writeFile(outCharts, JSON.stringify(charts), 'utf8');
  log(`wrote ${path.relative(REPO_DIR, outCharts)}`);

  // Quick sanity report so CI logs show what the site will render.
  const years = papers.map((p) => p.year).filter(Boolean);
  log(
    `years ${Math.min(...years)}..${Math.max(...years)} | with code ${papers.filter((p) => p.codeUrl).length}` +
      ` | with data ${papers.filter((p) => p.dataUrl).length}` +
      ` | with doi ${papers.filter((p) => p.doi).length}`,
  );
}

main().catch((err) => {
  console.error('[prepare-data] failed:', err);
  process.exit(1);
});