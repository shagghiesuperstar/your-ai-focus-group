/**
 * Semantic Similarity Rating (SSR) — paper-aligned scoring pipeline.
 *
 * Reference: arXiv:2510.08338 §3.3 + Appendix C.1
 * "LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation
 *  of Likert Ratings" — PyMC Labs / Colgate-Palmolive, 2025.
 *
 * Method:
 *   1. For each Likert point k ∈ {1…5}, maintain a set of 6 anchor sentences.
 *   2. Compute cosine similarity between the persona response and every anchor
 *      using a lightweight TF vector (no external embedding API required).
 *   3. Average the 6 per-anchor similarities to get rawScore[k].
 *   4. Min-subtract and re-normalise to produce a probability mass function
 *      (pmf) that sums to 1.
 *   5. Derive a point-estimate via expected value: EV = Σ k * pmf[k-1].
 *   6. Round EV to the nearest integer for the discrete Likert output.
 *
 * Note: The paper's production SSR uses text-embedding-3-small cosine
 * similarity, which yields higher distributional fidelity (KS sim ≈ 0.88 vs
 * ~0.72 for FLR). This implementation replicates the structural logic using
 * TF cosine similarity, which is serviceable without an OpenAI key and
 * preserves the pmf output that makes SSR superior to direct LLM rating.
 * Swap `cosineSimTF` for an embedding call to reach paper-level accuracy.
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

// ── Anchor statements ─────────────────────────────────────────────────────────
// 6 reference sentences per Likert point, mirroring the paper's Appendix C.1
// template: declarative first-person purchase-intent statements graded from
// strongly negative (1) to strongly positive (5).

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
    .filter(Boolean);
}

function tfVector(text: string): Map<string, number> {
  const map = new Map<string, number>();
  for (const token of tokenise(text)) {
    map.set(token, (map.get(token) ?? 0) + 1);
  }
  return map;
}

function cosineSimTF(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (const v of a.values()) normA += v * v;
  for (const v of b.values()) normB += v * v;
  if (!normA || !normB) return 0;
  for (const [token, va] of a.entries()) {
    const vb = b.get(token);
    if (vb !== undefined) dot += va * vb;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ── PMF normalisation ─────────────────────────────────────────────────────────

function normaliseToPmf(raw: Record<1 | 2 | 3 | 4 | 5, number>): ScorePmf {
  const min = Math.min(raw[1], raw[2], raw[3], raw[4], raw[5]);
  const eps = 1e-9;
  const shifted = {
    1: raw[1] - min + eps,
    2: raw[2] - min + eps,
    3: raw[3] - min + eps,
    4: raw[4] - min + eps,
    5: raw[5] - min + eps,
  };
  const total = shifted[1] + shifted[2] + shifted[3] + shifted[4] + shifted[5];
  return {
    p1: shifted[1] / total,
    p2: shifted[2] / total,
    p3: shifted[3] / total,
    p4: shifted[4] / total,
    p5: shifted[5] / total,
  };
}

// ── Public API ────────────────────────────────────────────────────────────────

export function scoreWithSSR(response: string): SsrResult {
  const rv = tfVector(response);

  const rawScores = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 } as Record<1 | 2 | 3 | 4 | 5, number>;
  for (const k of [1, 2, 3, 4, 5] as const) {
    const sims = ANCHORS[k].map(anchor => cosineSimTF(rv, tfVector(anchor)));
    rawScores[k] = sims.reduce((s, v) => s + v, 0) / sims.length;
  }

  const pmf = normaliseToPmf(rawScores);
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
