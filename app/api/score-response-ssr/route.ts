/**
 * SSR scoring endpoint — arXiv:2510.08338
 *
 * POST body: { personaId: string, personaName: string, response: string }
 *
 * Returns:
 *   { success: true, data: { personaId, score, scorePmf, method, reasoning } }
 *
 * If OPENAI_API_KEY is not set, falls back to the fixed FLR route
 * (score-response) and sets method: 'flr' in the response so callers
 * can detect the degraded path.
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  ANCHOR_STATEMENTS,
  cosineSimilarity,
  computeSSRPmf,
  pmfToScore,
  pmfToIntScore,
  type ScorePmf,
} from '@/lib/ssr';
import { callGeminiLowTemp, parseJSON } from '@/lib/gemini';
import { SYSTEM_LIKERT_SCORING, buildScoringPrompt } from '@/lib/prompts';

const OPENAI_EMBED_URL = 'https://api.openai.com/v1/embeddings';
const EMBED_MODEL = 'text-embedding-3-small';

async function embed(text: string, apiKey: string): Promise<number[]> {
  const res = await fetch(OPENAI_EMBED_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({ model: EMBED_MODEL, input: text }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenAI embed failed (${res.status}): ${err}`);
  }
  const json = (await res.json()) as { data: Array<{ embedding: number[] }> };
  return json.data[0].embedding;
}

async function embedBatch(
  texts: string[],
  apiKey: string
): Promise<number[][]> {
  const res = await fetch(OPENAI_EMBED_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({ model: EMBED_MODEL, input: texts }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenAI embed batch failed (${res.status}): ${err}`);
  }
  const json = (await res.json()) as { data: Array<{ index: number; embedding: number[] }> };
  return json.data
    .sort((a, b) => a.index - b.index)
    .map(d => d.embedding);
}

interface FLRResult {
  score: number;
  reasoning: string;
}

async function fallbackFLR(
  personaName: string,
  response: string
): Promise<{ score: number; reasoning: string }> {
  const raw = await callGeminiLowTemp(
    buildScoringPrompt(personaName, response),
    SYSTEM_LIKERT_SCORING
  );
  const result = parseJSON<FLRResult>(raw);
  return {
    score: Math.min(5, Math.max(1, Math.round(result.score))),
    reasoning: result.reasoning,
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { personaId, personaName, response } = body as {
      personaId: string;
      personaName: string;
      response: string;
    };

    if (!personaName || !response) {
      return NextResponse.json(
        { success: false, error: 'Missing personaName or response.' },
        { status: 400 }
      );
    }

    const openaiKey = process.env.OPENAI_API_KEY;

    if (!openaiKey) {
      console.warn(
        '[SSR] OPENAI_API_KEY not set — degrading to FLR fallback.'
      );
      const flr = await fallbackFLR(personaName, response);
      return NextResponse.json({
        success: true,
        data: {
          personaId,
          score: flr.score,
          scorePmf: null,
          method: 'flr',
          reasoning: flr.reasoning,
        },
      });
    }

    // Collect all anchor texts in a flat list for a single batched embed call
    const points = [1, 2, 3, 4, 5] as const;
    const allAnchorTexts: string[] = [];
    const anchorCounts: number[] = [];
    for (const pt of points) {
      const stmts = ANCHOR_STATEMENTS[pt];
      allAnchorTexts.push(...stmts);
      anchorCounts.push(stmts.length);
    }

    // Embed response + all anchors in two API calls
    const [responseEmbedding, ...anchorEmbeddingsFlat] = await Promise.all([
      embed(response, openaiKey),
      embedBatch(allAnchorTexts, openaiKey),
    ]).then(([re, ae]) => [re, ae]);

    // Reshape flat anchor embeddings back into per-point arrays
    const anchorEmbeddings = {} as Record<1 | 2 | 3 | 4 | 5, number[][]>;
    let offset = 0;
    const flatAnchors = anchorEmbeddingsFlat as unknown as number[][];
    for (let i = 0; i < points.length; i++) {
      const pt = points[i];
      anchorEmbeddings[pt] = flatAnchors.slice(offset, offset + anchorCounts[i]);
      offset += anchorCounts[i];
    }

    const pmf = computeSSRPmf(responseEmbedding as number[], anchorEmbeddings);
    const evScore = pmfToScore(pmf);
    const intScore = pmfToIntScore(pmf);

    return NextResponse.json({
      success: true,
      data: {
        personaId,
        score: intScore,
        evScore,
        scorePmf: pmf,
        method: 'ssr',
        reasoning: `SSR via text-embedding-3-small. EV=${evScore.toFixed(2)}, pmf=[${pmf.map(p => p.toFixed(3)).join(', ')}]`,
      },
    });
  } catch (error) {
    console.error('score-response-ssr error:', error);
    return NextResponse.json(
      { success: false, error: 'Scoring failed. Please try again.' },
      { status: 500 }
    );
  }
}
