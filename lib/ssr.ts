/**
 * Semantic Similarity Rating (SSR) — paper-aligned scoring pipeline.
 *
 * Reference: arXiv:2510.08338 §3.3 + Appendix C.1
 * "LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation
 *  of Likert Ratings" — PyMC Labs / Colgate-Palmolive, 2025.
 *
 * Pipeline:
 *   1. Strip stopwords and compute a TF vector for the response and each anchor.
 *   2. For each Likert point k ∈ {1…5}, average cosine similarity across 6 anchors.
 *   3. Softmax-normalise the 5 raw scores → probability mass function (pmf).
 *   4. Expected value: EV = Σ k·pmf[k].  Round to integer for the point estimate.
 *
 * Softmax vs min-subtract:
 *   The paper's production SSR uses embedding cosine similarity where raw scores
 *   span a wider range. With TF cosine similarity the 5 raw scores are tightly
 *   clustered (e.g. 0.31, 0.28, 0.30, 0.22, 0.19). Min-subtract amplifies tiny
 *   numerical differences into extreme pmf values. Softmax (temperature τ=10)
 *   produces a smoother, more stable distribution while still reflecting the
 *   relative ordering faithfully.
 *
 * Stopword filtering:
 *   Tokens like 'i','this','would','not','the' appear in every anchor sentence
 *   and every persona response. Without filtering, cosine similarity converges
 *   toward ~1.0 for all five Likert points, collapsing the pmf to uniform.
 *   Removing high-frequency function words restores the lexical signal.
 *
 * Accuracy note:
 *   Swap cosineSimTF for an OpenAI text-embedding-3-small call to reach the
 *   paper's KS similarity ≈ 0.88. TF cosine without embeddings is the closest
 *   approximation that requires no external API key.
 */

export interface ScorePmf {
  p1: number;
  p2: number;
  p3: number;
  p4: number;
  p5: number;
}

export interface SsrResult {
  score: number;
  evScore: number;
  scorePmf: ScorePmf;
  method: 'ssr';
  reasoning: string;
}

// ── Stopwords ─────────────────────────────────────────────────────────────────
// English function words that appear uniformly across anchors and responses.
// Removing them prevents cosine similarity from collapsing toward 1.0.

const STOPWORDS = new Set([
  'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
  'yourself','he','him','his','himself','she','her','hers','herself','it',
  'its','itself','they','them','their','theirs','themselves','what','which',
  'who','whom','this','that','these','those','am','is','are','was','were',
  'be','been','being','have','has','had','having','do','does','did','doing',
  'a','an','the','and','but','if','or','because','as','until','while','of',
  'at','by','for','with','about','against','between','into','through','during',
  'before','after','above','below','to','from','up','down','in','out','on',
  'off','over','under','again','further','then','once','here','there','when',
  'where','why','how','all','both','each','few','more','most','other','some',
  'such','no','nor','not','only','own','same','so','than','too','very','s',
  't','can','will','just','don','should','now','d','ll','m','o','re','ve',
  'y','ain','aren','couldn','didn','doesn','hadn','hasn','haven','isn',
  'ma','mightn','mustn','needn','shan','shouldn','wasn','weren','won','wouldn',
]);

// ── Anchor statements ─────────────────────────────────────────────────────────
// 6 first-person purchase-intent statements per Likert point.
// Mirror the template in paper Appendix C.1.

const ANCHORS: Record<1 | 2 | 3 | 4 | 5, readonly string[]> = {
  1: [
    'I would definitely not buy this product.',
    'This does not appeal to me at all.',
    'I would actively avoid using something like this.',
    'I am strongly uninterested in this idea.',
    'This feels like a poor fit for my needs and I would reject it.',
    'I have no desire to try this under any circumstances.',
  ],
  2: [
    'I am mostly uninterested, though I can see one small upside.',
    'This is not very compelling and I probably would not buy it.',
    'My overall reaction is slightly negative, with a minor caveat.',
    'I would be unlikely to try this unless someone pushed me to.',
    'This does not fit me well even if it has one minor benefit.',
    'I lean toward not buying this but I am not entirely opposed.',
  ],
  3: [
    'I have mixed feelings and could go either way on this.',
    'I am unsure whether I would buy this — I need more information.',
    'My reaction is neutral; I see both pros and cons here.',
    'This could be useful but I have not made up my mind.',
    'I am on the fence about whether this product is right for me.',
    'I feel indifferent and would consider it if circumstances changed.',
  ],
  4: [
    'I am interested and would seriously consider buying this.',
    'This sounds useful to me and I have a positive reaction overall.',
    'I can see myself using this and would likely try it.',
    'This seems like a good fit and I feel fairly motivated to act.',
    'I would explore this further and probably make a purchase.',
    'My reaction is positive and I think this addresses a real need for me.',
  ],
  5: [
    'I am very interested and would buy this right away.',
    'This strongly appeals to me and I am excited about it.',
    'I would actively seek this out as soon as it is available.',
    'This feels exactly like something I want and I am highly enthusiastic.',
    'I would recommend this to others and purchase it immediately.',
    'My reaction is extremely positive — this is a must-have for me.',
  ],
};

// ── TF cosine similarity ──────────────────────────────────────────────────────

function tokenise(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 1 && !STOPWORDS.has(t));
}

function tfVector(text: string): Map<string, number> {
  const map = new Map<string, number>();
  for (const token of tokenise(text)) {
    map.set(token, (map.get(token) ?? 0) + 1);
  }
  return map;
}

function cosineSimTF(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0, normA = 0, normB = 0;
  for (const v of a.values()) normA += v * v;
  for (const v of b.values()) normB += v * v;
  if (!normA || !normB) return 0;
  for (const [token, va] of a.entries()) {
    const vb = b.get(token);
    if (vb !== undefined) dot += va * vb;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ── Softmax PMF ───────────────────────────────────────────────────────────────
// Temperature τ=10 amplifies differences in the TF cosine score range (~0.1–0.5)
// to a meaningful pmf without driving it to near-one-hot.

const SOFTMAX_TEMP = 10;

function softmaxToPmf(raw: Record<1 | 2 | 3 | 4 | 5, number>): ScorePmf {
  const vals = [raw[1], raw[2], raw[3], raw[4], raw[5]];
  const maxVal = Math.max(...vals);
  const exps = vals.map(v => Math.exp((v - maxVal) * SOFTMAX_TEMP));
  const total = exps.reduce((s, v) => s + v, 0);
  const [p1, p2, p3, p4, p5] = exps.map(v => v / total);
  return { p1, p2, p3, p4, p5 };
}

// ── Public API ────────────────────────────────────────────────────────────────

export function scoreWithSSR(response: string): SsrResult {
  const rv = tfVector(response);

  const rawScores = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 } as Record<1 | 2 | 3 | 4 | 5, number>;
  for (const k of [1, 2, 3, 4, 5] as const) {
    const sims = ANCHORS[k].map(anchor => cosineSimTF(rv, tfVector(anchor)));
    rawScores[k] = sims.reduce((s, v) => s + v, 0) / sims.length;
  }

  const pmf = softmaxToPmf(rawScores);
  const ev = pmf.p1 * 1 + pmf.p2 * 2 + pmf.p3 * 3 + pmf.p4 * 4 + pmf.p5 * 5;
  const score = Math.min(5, Math.max(1, Math.round(ev))) as 1 | 2 | 3 | 4 | 5;

  const peak = (['p1','p2','p3','p4','p5'] as const).reduce(
    (best, k) => (pmf[k] > pmf[best] ? k : best),
    'p3' as keyof ScorePmf
  );
  const peakLabel = { p1:'1', p2:'2', p3:'3', p4:'4', p5:'5' }[peak];

  return {
    score,
    evScore: Math.round(ev * 100) / 100,
    scorePmf: pmf,
    method: 'ssr',
    reasoning: `SSR anchor matching: peak mass at ${peakLabel}/5, EV = ${ev.toFixed(2)} → rounded to ${score}.`,
  };
}
