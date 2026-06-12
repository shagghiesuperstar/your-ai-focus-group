import { NextRequest, NextResponse } from 'next/server';
import { callGeminiLowTemp, parseJSON } from '@/lib/gemini';
import { SYSTEM_LIKERT_SCORING, buildScoringPrompt } from '@/lib/prompts';

interface FlrResult {
  score: number;
  reasoning: string;
}

/**
 * POST /api/score-response
 *
 * Uses callGeminiLowTemp (T=0.3) per paper §4.1 recommendation.
 * Lower temperature reduces inter-run variance in rating tasks without
 * sacrificing the directional accuracy gained by the FLR two-step.
 *
 * Note: The SSR branch (feat/ssr-scoring-pipeline) replaces this LLM-rater
 * with anchor-based cosine similarity scoring, which achieves higher
 * distributional fidelity (paper KS sim 0.88 vs 0.72 for FLR).
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

    let raw: string;
    try {
      raw = await callGeminiLowTemp(buildScoringPrompt(personaName, response), SYSTEM_LIKERT_SCORING);
    } catch (err) {
      console.error('Gemini scoring call failed:', err);
      return NextResponse.json({ success: false, error: 'Scoring failed.' }, { status: 500 });
    }

    let result: FlrResult;
    try {
      result = parseJSON<FlrResult>(raw);
    } catch {
      try {
        const retryRaw = await callGeminiLowTemp(
          buildScoringPrompt(personaName, response) + '\n\nIMPORTANT: Respond ONLY with valid JSON.',
          SYSTEM_LIKERT_SCORING
        );
        result = parseJSON<FlrResult>(retryRaw);
      } catch (err) {
        console.error('Score JSON parse failed:', err);
        return NextResponse.json({ success: false, error: 'Scoring failed.' }, { status: 500 });
      }
    }

    const score = Math.min(5, Math.max(1, Math.round(result.score)));
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
