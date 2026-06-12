/**
 * Semantic Similarity Rating (SSR) — arXiv:2510.08338
 *
 * The paper's primary contribution. Instead of asking an LLM to output
 * an integer, we:
 *   1. Embed the persona's free-text response.
 *   2. Embed each of 6 reference statement sets (one set per Likert point,
 *      Appendix C.1 of the paper).
 *   3. Compute cosine similarity between the response and every anchor.
 *   4. Build a pmf over {1,2,3,4,5} by averaging similarities per point
 *      and softmax-normalising.
 *   5. Derive a scalar score as E[X] over the pmf.
 *
 * Requires OPENAI_API_KEY (text-embedding-3-small, 1536-dim).
 */

export type ScorePmf = [number, number, number, number, number];

// ---------------------------------------------------------------------------
// Anchor statements — Appendix C.1, arXiv:2510.08338
// Six reference statements per Likert point, covering the full purchase-intent
// spectrum from strong rejection to strong enthusiasm.
// ---------------------------------------------------------------------------
export const ANCHOR_STATEMENTS: Record<1 | 2 | 3 | 4 | 5, string[]> = {
  1: [
    'I would definitely not buy this product.',
    'I have no interest in purchasing this at all.',
    'This product is not for me and I would not consider buying it.',
    'I would never purchase something like this.',
    'I am completely uninterested in buying this.',
    'There is no chance I would buy this product.',
  ],
  2: [
    'I probably would not buy this product.',
    'I am unlikely to purchase this.',
    'I do not think I would buy this.',
    'I would most likely not buy this product.',
    'Buying this product is something I would probably not do.',
    'I lean toward not purchasing this.',
  ],
  3: [
    'I might or might not buy this product.',
    'I am unsure whether I would purchase this.',
    'I could see myself buying this or not buying it.',
    'I am neutral about purchasing this product.',
    'Whether I would buy this is uncertain to me.',
    'I have mixed feelings about purchasing this.',
  ],
  4: [
    'I probably would buy this product.',
    'I think I would purchase this.',
    'I am likely to buy this product.',
    'Buying this product is something I would probably do.',
    'I lean toward purchasing this.',
    'I would most likely buy this.',
  ],
  5: [
    'I would definitely buy this product.',
    'I am very likely to purchase this.',
    'I would certainly buy this product.',
    'There is a strong chance I would purchase this.',
    'I am highly motivated to buy this product.',
    'I would eagerly purchase this product.',
  ],
};

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// ---------------------------------------------------------------------------
// Core SSR computation
// ---------------------------------------------------------------------------

/**
 * Given embeddings for the response and all anchors, compute the pmf.
 *
 * @param responseEmbedding  1536-dim vector for the persona response
 * @param anchorEmbeddings   Map from Likert point (1-5) to array of
 *                           per-anchor embeddings (6 per point)
 * @returns ScorePmf — probability mass over points 1-5 (sums to 1.0)
 */
export function computeSSRPmf(
  responseEmbedding: number[],
  anchorEmbeddings: Record<1 | 2 | 3 | 4 | 5, number[][]>
): ScorePmf {
  const meanSims: number[] = ([1, 2, 3, 4, 5] as const).map(point => {
    const sims = anchorEmbeddings[point].map(anchor =>
      cosineSimilarity(responseEmbedding, anchor)
    );
    return sims.reduce((a, b) => a + b, 0) / sims.length;
  });

  const pmf = softmax(meanSims);
  return pmf as unknown as ScorePmf;
}

/**
 * Convert a pmf to an expected-value scalar score (1–5, one decimal).
 */
export function pmfToScore(pmf: ScorePmf): number {
  const ev = pmf.reduce((sum, p, i) => sum + p * (i + 1), 0);
  return Math.round(ev * 10) / 10;
}

/**
 * Derive a rounded integer score (1-5) from a pmf via argmax.
 * Use this when a single integer is required by downstream UI.
 */
export function pmfToIntScore(pmf: ScorePmf): number {
  const maxIdx = pmf.reduce((best, p, i) => (p > pmf[best] ? i : best), 0);
  return maxIdx + 1;
}
