/**
 * papers.ts — single accessor for the generated paper dataset.
 *
 * The JSON is produced by scripts/prepare-data.mjs (prebuild step) from either
 * data/papers/*.json or a README scrape. Pages must not import the repo files
 * directly: Vite would either fail to resolve them (outside project root) or
 * bundle a second copy per page.
 */
import generated from '../data/papers.generated.json';

export interface CodeLink {
  label: string;
  url: string;
}

export interface Verification {
  checked_at?: string | null;
  link_ok?: boolean | null;
  metadata_source?: string | null;
  confidence?: string | null;
  note?: string | null;
}

export interface Provenance {
  source?: string | null;
  readme_line?: number | null;
  legacy_id?: string | null;
  migrated_at?: string | null;
}

export interface Impact {
  tier: 'landmark' | 'notable' | 'standard';
  role?: string | null;
  role_zh?: string | null;
  track?: string | null;
}

export interface Paper {
  id: string;
  lang: string | null;
  year: number | null;
  altYears: number[];
  venue: string;
  venueId: string | null;
  venueType: string;
  title: string;
  titleZh: string | null;
  shortName: string;
  paperUrl: string | null;
  altUrls: Record<string, string>;
  codeUrl: string | null;
  codeLinks: CodeLink[];
  dataUrl: string | null;
  reviewUrl: string | null;
  projectUrl: string | null;
  doi: string | null;
  arxivId: string | null;
  openalexId: string | null;
  authors: string[];
  datasets: string[];
  layers: string[];
  directionTags: string[];
  badges: string[];
  section: string;
  track: string;
  task: string;
  taskZh: string | null;
  methodTags: string[];
  note: string | null;
  noteZh: string | null;
  featured: boolean;
  status: string;
  impact: Impact | null;
  /** Venue tier from data/venues.json: top / mid / low / preprint. */
  venueTier: string | null;
  /** Citation count from the committed snapshot, or null when not snapshotted. */
  citations: number | null;
  verification: Verification | null;
  provenance: Provenance | null;
  source: string;
}

export interface Dataset {
  generatedUtc: string;
  /** ISO date the citation snapshot was taken, so the UI can show its age. */
  citationsRetrievedOn: string | null;
  provenance: {
    source: string;
    files: string[];
    degraded?: boolean;
    en?: number;
    zh?: number;
  };
  count: number;
  papers: Paper[];
}

const data = generated as unknown as Dataset;

export const provenance = data.provenance;
export const generatedUtc = data.generatedUtc;
export const citationsRetrievedOn = data.citationsRetrievedOn;

/** Deduplicate by normalized title: the same paper can appear under both
 *  year-shifted headings or in both README languages after a merge. */
function dedupe(list: Paper[]): Paper[] {
  const seen = new Map<string, Paper>();
  for (const p of list) {
    const key = p.title.toLowerCase().replace(/[^a-z0-9]+/g, '');
    const prev = seen.get(key);
    if (!prev) {
      seen.set(key, p);
      continue;
    }
    // Keep the richer record (more badges/links, has year).
    const score = (x: Paper) =>
      (x.codeUrl ? 2 : 0) + (x.dataUrl ? 1 : 0) + (x.year ? 1 : 0) + (x.paperUrl ? 1 : 0);
    if (score(p) > score(prev)) seen.set(key, p);
  }
  return [...seen.values()];
}

export const papers: Paper[] = dedupe(data.papers).sort(
  (a, b) => (b.year ?? 0) - (a.year ?? 0) || a.title.localeCompare(b.title),
);

export const years = [...new Set(papers.map((p) => p.year).filter(Boolean))].sort(
  (a, b) => (b as number) - (a as number),
) as number[];

export const venues = [...new Set(papers.map((p) => p.venue))].sort();

export const tasks = [...new Set(papers.map((p) => p.task))].sort();

export const methodTags = [...new Set(papers.flatMap((p) => p.methodTags))].sort();

export const directionTags = [...new Set(papers.flatMap((p) => p.directionTags))].sort();

export const stats = {
  total: papers.length,
  withCode: papers.filter((p) => p.codeUrl).length,
  withData: papers.filter((p) => p.dataUrl).length,
  inTopVenue: papers.filter((p) => p.venueTier === 'top').length,
  landmarks: papers.filter((p) => p.impact?.tier === 'landmark').length,
  withDoi: papers.filter((p) => p.doi).length,
  withAuthors: papers.filter((p) => p.authors.length).length,
  minYear: Math.min(...years),
  maxYear: Math.max(...years),
};

export function byYear(): Array<[number, number]> {
  const m = new Map<number, number>();
  for (const p of papers) if (p.year) m.set(p.year, (m.get(p.year) ?? 0) + 1);
  return [...m.entries()].sort((a, b) => a[0] - b[0]);
}

export function byVenue(limit = 15): Array<[string, number]> {
  const m = new Map<string, number>();
  for (const p of papers) m.set(p.venue, (m.get(p.venue) ?? 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, limit);
}

export function byTask(): Array<[string, number]> {
  const m = new Map<string, number>();
  for (const p of papers) m.set(p.task, (m.get(p.task) ?? 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1]);
}

/** Papers ordered oldest -> newest for the research-lineage timeline. */
export function lineage(): Array<{ year: number; papers: Paper[] }> {
  const m = new Map<number, Paper[]>();
  for (const p of papers) {
    if (!p.year) continue;
    if (!m.has(p.year)) m.set(p.year, []);
    m.get(p.year)!.push(p);
  }
  return [...m.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([year, list]) => ({ year, papers: list }));
}

/**
 * Index every paper by id for the detail route.
 *
 * Ids come from data/papers and are guaranteed unique (validated by
 * `migrate_readme_to_json.py --check`); a duplicate here would silently shadow
 * a page, so the build-time generator is the one place that has to enforce it.
 */
export const paperById: Map<string, Paper> = new Map(papers.map((p) => [p.id, p]));

/** Papers sharing a direction tag, nearest years first, excluding self. */
export function related(p: Paper, limit = 6): Paper[] {
  const tags = new Set(p.directionTags);
  if (tags.size === 0) {
    // Fall back to the same primary method tag so the block is never empty for
    // pre-2020 baselines, which predate the direction vocabulary.
    const [first] = p.methodTags;
    if (!first) return [];
    return papers
      .filter((x) => x.id !== p.id && x.methodTags.includes(first))
      .slice(0, limit);
  }
  return papers
    .filter((x) => x.id !== p.id && x.directionTags.some((t) => tags.has(t)))
    .sort(
      (a, b) =>
        b.directionTags.filter((t) => tags.has(t)).length -
          a.directionTags.filter((t) => tags.has(t)).length ||
        Math.abs((a.year ?? 0) - (p.year ?? 0)) - Math.abs((b.year ?? 0) - (p.year ?? 0)),
    )
    .slice(0, limit);
}

/** Landmark papers grouped by research track, oldest first inside a track. */
export const TRACKS = [
  'early_deep',
  'graph_basics',
  'adaptive_graph',
  'transformer',
  'foundation_model',
] as const;
export type Track = (typeof TRACKS)[number];

export interface TrackGroup {
  track: Track;
  papers: Paper[];
}

export function landmarksByTrack(): TrackGroup[] {
  const groups: TrackGroup[] = [];
  for (const track of TRACKS) {
    const list = papers
      .filter((p) => p.impact?.tier === 'landmark' && p.impact.track === track)
      .sort((a, b) => (a.year ?? 0) - (b.year ?? 0));
    if (list.length) groups.push({ track, papers: list });
  }
  return groups;
}

export const landmarkPapers = papers.filter((p) => p.impact?.tier === 'landmark');

export function impactCounts(): Record<string, number> {
  const out: Record<string, number> = { landmark: 0, notable: 0, standard: 0 };
  for (const p of papers) {
    const t = p.impact?.tier;
    if (t) out[t] = (out[t] ?? 0) + 1;
  }
  return out;
}

export function venueTierCounts(): Record<string, number> {
  const out: Record<string, number> = {};
  for (const p of papers) {
    if (p.venueTier) out[p.venueTier] = (out[p.venueTier] ?? 0) + 1;
  }
  return out;
}


/** Papers that carry a citation count, most cited first. */
export function byCitations(limit = 10): Paper[] {
  return papers
    .filter((p) => typeof p.citations === 'number')
    .sort((a, b) => (b.citations ?? 0) - (a.citations ?? 0))
    .slice(0, limit);
}
