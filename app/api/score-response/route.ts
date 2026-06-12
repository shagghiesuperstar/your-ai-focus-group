import { NextRequest, NextResponse } from 'next/server';
import { callGemini, parseJSON } from '@/lib/gemini';
import { SYSTEM_LIKERT_SCORING, buildScoringPrompt } from '@/lib/prompts';
import { scoreWithSSR } from '@/lib/ssr';

interface FlrResult {
  score: number;
  reasoning: string;
}

/**
 * POST /api/score-response
 *
 * Scoring strategy (SSR-first, FLR fallback):
 *   - Default: SSR (Semantic Similarity Rating) — paper §3.3.
 *     Pure server-side TF cosine similarity against anchor sentences.
 *     No additional API key required.
 *   - FLR fallback: set env var FOCUS_GROUP_SCORING_METHOD=flr to use
 *     the original LLM-as-rater path at temperature 0.3 (paper §4.1).
 *
 * Response shape:
 *   { success, data: { personaId, score, evScore?, scorePmf?, method, reasoning } }
 */
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

    // ── SSR path (default) ───────────────────────────────────────────────────
    if (process.env.FOCUS_GROUP_SCORING_METHOD !== 'flr') {
      const result = scoreWithSSR(response);
      return NextResponse.json({
        success: true,
        data: { personaId, ...result },
      });
    }

    // ── FLR path (opt-in via env var) ────────────────────────────────────────
    // Temperature 0.3 per paper §4.1 recommendation for rating tasks.
    let raw: string;
    try {
      raw = await callGemini(
        buildScoringPrompt(personaName, response),
        SYSTEM_LIKERT_SCORING,
        4096,
        0.3
      );
    } catch (err) {
      console.error('Gemini FLR scoring call failed:', err);
      return NextResponse.json(
        { success: false, error: 'Scoring failed.' },
        { status: 500 }
      );
    }

    let result: FlrResult;
    try {
      result = parseJSON<FlrResult>(raw);
    } catch {
      try {
        const retryRaw = await callGemini(
          buildScoringPrompt(personaName, response) + '\n\nIMPORTANT: Respond ONLY with valid JSON.',
          SYSTEM_LIKERT_SCORING,
          4096,
          0.3
        );
        result = parseJSON<FlrResult>(retryRaw);
      } catch (err) {
        console.error('FLR score JSON parse failed:', err);
        return NextResponse.json(
          { success: false, error: 'Scoring failed.' },
          { status: 500 }
        );
      }
    }

    const score = Math.min(5, Math.max(1, Math.round(result.score))) as 1 | 2 | 3 | 4 | 5;
    return NextResponse.json({
      success: true,
      data: { personaId, score, reasoning: result.reasoning, method: 'flr' },
    });
  } catch (error) {
    console.error('score-response error:', error);
    return NextResponse.json(
      { success: false, error: 'Something went wrong. Please try again.' },
      { status: 500 }
    );
  }
}
