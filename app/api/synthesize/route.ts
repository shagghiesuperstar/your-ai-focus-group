import { NextRequest, NextResponse } from 'next/server';
import { callGemini, parseJSON } from '@/lib/gemini';
import { SYSTEM_SYNTHESIS, buildSynthesisPrompt } from '@/lib/prompts';
import { SynthesisResult } from '@/lib/types';

interface SynthesisInput {
  name: string;
  age: number;
  occupation: string;
  response: string;
  score: number;
}

/**
 * POST /api/synthesize
 *
 * averageScore is computed server-side from the actual integer scores
 * before the LLM call. The LLM receives the verified average and is
 * instructed to use it rather than recalculate it.
 *
 * Why: LLMs are unreliable at arithmetic over JSON arrays (off-by-one
 * rounding, hallucinated sums). Delegating average calculation to the
 * LLM introduces a silent accuracy bug in the results card.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { results } = body as { results: SynthesisInput[] };

    if (!Array.isArray(results) || results.length === 0) {
      return NextResponse.json({ success: false, error: 'No results to synthesize.' }, { status: 400 });
    }

    // Compute the average server-side — do not delegate to the LLM.
    const validScores = results
      .map(r => r.score)
      .filter(s => typeof s === 'number' && s >= 1 && s <= 5);
    const averageScore = validScores.length > 0
      ? Math.round((validScores.reduce((s, v) => s + v, 0) / validScores.length) * 10) / 10
      : 3.0;

    let raw: string;
    try {
      raw = await callGemini(buildSynthesisPrompt(results, averageScore), SYSTEM_SYNTHESIS, 2048);
    } catch (err) {
      console.error('Gemini synthesis call failed:', err);
      return NextResponse.json({ success: false, error: 'Synthesis failed. Please try again.' }, { status: 500 });
    }

    let synthesis: Omit<SynthesisResult, 'averageScore'>;
    try {
      synthesis = parseJSON<Omit<SynthesisResult, 'averageScore'>>(raw);
    } catch {
      try {
        const retryRaw = await callGemini(
          buildSynthesisPrompt(results, averageScore) + '\n\nIMPORTANT: Respond ONLY with valid JSON.',
          SYSTEM_SYNTHESIS,
          2048
        );
        synthesis = parseJSON<Omit<SynthesisResult, 'averageScore'>>(retryRaw);
      } catch (err) {
        console.error('Synthesis JSON parse failed:', err);
        return NextResponse.json({ success: false, error: 'Synthesis failed. Please try again.' }, { status: 500 });
      }
    }

    // Always return the server-computed average, regardless of what the LLM wrote.
    return NextResponse.json({
      success: true,
      data: { ...synthesis, averageScore },
    });
  } catch (error) {
    console.error('synthesize error:', error);
    return NextResponse.json(
      { success: false, error: 'Something went wrong. Please try again.' },
      { status: 500 }
    );
  }
}
