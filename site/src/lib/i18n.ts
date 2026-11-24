/**
 * i18n.ts — minimal dictionary-based i18n (no runtime, no client JS).
 *
 * Astro content collections are overkill for ~60 UI strings; a typed record
 * keeps the bilingual contract checkable at build time and adds nothing to the
 * client bundle. Chinese strings mirror README.zh.md section naming so the
 * site and the repo read the same vocabulary.
 */
export const languages = ['en', 'zh'] as const;
export type Lang = (typeof languages)[number];

export const htmlLang: Record<Lang, string> = { en: 'en', zh: 'zh-CN' };
export const ogLocale: Record<Lang, string> = { en: 'en_US', zh: 'zh_CN' };

const en = {
  siteName: 'Awesome Traffic Prediction',
  siteTagline: 'Papers, datasets, toolkits and benchmarks for traffic prediction',
  nav: {
    home: 'Overview',
    papers: 'Papers',
    datasets: 'Datasets',
    toolkits: 'Toolkits',
    benchmark: 'Benchmark',
    trends: 'Trends',
    lineage: 'Lineage',
  },
  langSwitch: '中文',
  hero: {
    total: 'papers indexed',
    years: 'years covered',
    code: 'with code',
    data: 'with data',
  },
  papers: {
    title: 'Paper library',
    lead: 'Every paper in the collection, tagged by year, venue, task and method.',
    search: 'Search title, venue or tag…',
    filters: 'Filters',
    year: 'Year',
    venue: 'Venue',
    task: 'Task',
    method: 'Method tags',
    reset: 'Reset',
    sort: 'Sort',
    sortNewest: 'Newest first',
    sortOldest: 'Oldest first',
    sortVenue: 'By venue',
    shown: 'shown',
    empty: 'No paper matches the current filters.',
    code: 'code',
    data: 'data',
    pdf: 'pdf',
    review: 'review',
    copyLink: 'Copy filter link',
    copied: 'Link copied',
    details: 'details',
    layer: 'Lineage layer',
    direction: 'Direction',
    openDetails: 'Open paper page',
    totalLabel: 'total',
  },
  detail: {
    overview: 'Overview',
    authors: 'Authors',
    venue: 'Venue',
    year: 'Year',
    doi: 'DOI',
    arxiv: 'arXiv',
    datasets: 'Datasets',
    methodTags: 'Method tags',
    directionTags: 'Direction tags',
    layers: 'Lineage layer',
    links: 'Links',
    notes: 'Notes',
    paper: 'Paper',
    code: 'Code',
    data: 'Data',
    review: 'Review',
    project: 'Project',
    originalLink: 'Original source link',
    verified: 'Metadata verified',
    unverified: 'Metadata not verified yet',
    backToLibrary: 'Back to paper library',
    related: 'Same direction',
    noNotes: 'No curator note for this paper yet.',
  },
  datasets: { title: 'Datasets', lead: 'Benchmark datasets with coverage and evaluation notes.' },
  toolkits: { title: 'Toolkits', lead: 'Frameworks and libraries you can build on.' },
  benchmark: {
    title: 'Benchmark & Evaluation',
    lead: 'Reporting conventions that make cross-paper comparison honest.',
    matrix: 'Benchmark matrix',
  },
  trends: {
    title: 'Trend overview',
    lead: 'How the field moved: output per year, where it is published, and which terms dominate.',
    byYear: 'Papers per year',
    byVenue: 'Top venues',
    byTask: 'Task distribution',
    hotspots: 'Yearly hotspot terms',
    wordcloud: 'Monthly title terms',
    noJs: 'Charts need JavaScript to draw. The same numbers are listed below.',
  },
  lineage: { title: 'Research lineage', lead: 'The collection in chronological order, year by year.' },
  search: { title: 'Search', placeholder: 'Search the site…', open: 'Search', close: 'Close' },
  footer: {
    repo: 'Source repository',
    edit: 'Contribution guide',
  },
  fallbackTable: 'View as table',
  provenance: {
    structured: '',
    readme: '',
    degraded: '',
  },
};

const zh: typeof en = {
  siteName: '交通流预测资源汇总',
  siteTagline: '交通流预测方向的论文、数据集、工具库与基准',
  nav: {
    home: '概览',
    papers: '论文库',
    datasets: '数据集',
    toolkits: '工具',
    benchmark: '基准评测',
    trends: '趋势',
    lineage: '研究脉络',
  },
  langSwitch: 'English',
  hero: { total: '篇论文', years: '个年份', code: '有代码', data: '有数据' },
  papers: {
    title: '论文库',
    lead: '收录的全部论文，按年份、会议/期刊、任务与方法标注。',
    search: '搜索标题、期刊或标签…',
    filters: '筛选',
    year: '年份',
    venue: '会议/期刊',
    task: '任务',
    method: '方法标签',
    reset: '重置',
    sort: '排序',
    sortNewest: '最新优先',
    sortOldest: '最早优先',
    sortVenue: '按期刊',
    shown: '条结果',
    empty: '没有符合当前筛选条件的论文。',
    code: '代码',
    data: '数据',
    pdf: '原文',
    review: '综述',
    copyLink: '复制筛选链接',
    copied: '链接已复制',
    details: '详情',
    layer: '脉络层级',
    direction: '研究方向',
    openDetails: '打开论文页',
    totalLabel: '总条目',
  },
  detail: {
    overview: '基本信息',
    authors: '作者',
    venue: '期刊/会议',
    year: '年份',
    doi: 'DOI',
    arxiv: 'arXiv',
    datasets: '数据集',
    methodTags: '方法标签',
    directionTags: '研究方向',
    layers: '脉络层级',
    links: '链接',
    notes: '备注',
    paper: '原文',
    code: '代码',
    data: '数据',
    review: '综述',
    project: '项目主页',
    originalLink: '原始来源链接',
    verified: '元数据已核验',
    unverified: '元数据尚未核验',
    backToLibrary: '返回论文库',
    related: '同方向论文',
    noNotes: '暂时还没有该论文的策展备注。',
  },
  datasets: { title: '数据集', lead: '基准数据集的覆盖范围与评测注意事项。' },
  toolkits: { title: '工具库', lead: '可直接复用的框架与库。' },
  benchmark: {
    title: '基准与评测',
    lead: '让跨论文对比可信的汇报约定。',
    matrix: '基准矩阵',
  },
  trends: {
    title: '趋势总览',
    lead: '领域是怎么走的：逐年产出、发表去向与高频术语。',
    byYear: '逐年论文数',
    byVenue: '高频期刊/会议',
    byTask: '任务分布',
    hotspots: '年度热点词',
    wordcloud: '月度标题词频',
    noJs: '图表需要 JavaScript 绘制，同样的数据在下方以表格列出。',
  },
  lineage: { title: '研究脉络', lead: '按年份排列的完整论文列表。' },
  search: { title: '搜索', placeholder: '全站搜索…', open: '搜索', close: '关闭' },
  footer: {
    repo: '源仓库',
    edit: '贡献指南',
  },
  fallbackTable: '以表格查看',
  provenance: {
    structured: '',
    readme: '',
    degraded: '',
  },
};

/** Task vocabulary -> Chinese.
 *
 *  Composite labels ("Large-scale traffic flow prediction") are translated by
 *  matching the longest known suffix and translating any leading qualifiers, so
 *  a new paper usually needs no new entry: only the head noun has to be known.
 */
const taskZh: Record<string, string> = {
  'traffic forecasting': '交通预测',
  'traffic prediction': '交通预测',
  'traffic flow prediction': '交通流量预测',
  'traffic flow forecasting': '交通流量预测',
  'traffic speed prediction': '交通速度预测',
  survey: '综述',
  'missing value imputation': '缺失值填补',
  'trajectory prediction': '轨迹预测',
  'demand prediction': '需求预测',
  nowcasting: '临近预报',
  'accident / incident prediction': '事故/事件预测',
  'travel time estimation': '行程时间估计',
  'urban spatio-temporal prediction': '城市时空预测',
  'probabilistic traffic forecasting': '概率交通预测',
};

const taskQualifierZh: Record<string, string> = {
  urban: '城市',
  'large-scale': '大规模',
  'short-term': '短期',
  'long-horizon': '长时程',
  'real-time': '实时',
  'multi-step': '多步',
};

export function localizeTask(task: string | null, lang: Lang): string {
  if (!task) return '';
  if (lang !== 'zh') return task;
  const hit = taskZh[task.toLowerCase()];
  if (hit) return hit;

  const words = task.split(' ');
  for (let i = 1; i < words.length; i++) {
    const head = taskZh[words.slice(i).join(' ').toLowerCase()];
    if (head) {
      const prefix = words
        .slice(0, i)
        .map((w) => taskQualifierZh[w.toLowerCase()] ?? '')
        .join('');
      return prefix + head;
    }
  }
  return task;
}

/** Lineage layer labels; the raw value stays in the data for filtering. */
const layerZh: Record<string, string> = {
  sota_core: '近年主流',
  classic_baseline: '经典基线',
  survey: '综述',
  application: '应用',
};

export function localizeLayer(layer: string, lang: Lang): string {
  return lang === 'zh' ? (layerZh[layer] ?? layer) : layer.replace(/_/g, ' ');
}

export const ui = { en, zh } as const;

export function t(lang: Lang) {
  return ui[lang];
}

/** Route map: zh is prefixed, en is at root (see astro.config.mjs i18n). */
export const routes = {
  home: '/',
  papers: '/papers/',
  datasets: '/datasets/',
  toolkits: '/toolkits/',
  benchmark: '/benchmark/',
  trends: '/trends/',
  lineage: '/lineage/',
} as const;

export function pathFor(lang: Lang, route: keyof typeof routes): string {
  const p = routes[route];
  return lang === 'zh' ? `/zh${p}` : p;
}