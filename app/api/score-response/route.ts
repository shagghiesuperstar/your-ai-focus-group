import { NextRequest, NextResponse } from 'next/server';
import { callGemini, parseJSON } from '@/lib/gemini';
import { SYSTEM_LIKERT_SCORING, buildScoringPrompt } from '@/lib/prompts';
import { sanitiseFreeText } from '@/lib/validate';

const PERSONA_NAME_MAX = 80;
const RESPONSE_MAX = 1200;

interface ScoreResult {
  score: number;
  reasoning: string;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const personaId = typeof body?.personaId === 'string' ? body.personaId : '';
    const personaName = sanitiseFreeText(
      typeof body?.personaName === 'string' ? body.personaName : '',
      PERSONA_NAME_MAX
    );
    const response = sanitiseFreeText(
      typeof body?.response === 'string' ? body.response : '',
      RESPONSE_MAX
    );

    if (!personaName || !response) {
      return NextResponse.json(
        { success: false, error: 'Missing personaName or response.' },
        { status: 400 }
      );
    }

    let raw: string;
    try {
      raw = await callGemini(buildScoringPrompt(personaName, response), SYSTEM_LIKERT_SCORING);
    } catch (err) {
      console.error('Gemini scoring call failed:', err);
      return NextResponse.json({ success: false, error: 'Scoring failed.' }, { status: 500 });
    }

    let result: ScoreResult;
    try {
      result = parseJSON<ScoreResult>(raw);
    } catch {
      try {
        const retryRaw = await callGemini(
          buildScoringPrompt(personaName, response) + '\n\nIMPORTANT: Respond ONLY with valid JSON.',
          SYSTEM_LIKERT_SCORING
        );
        result = parseJSON<ScoreResult>(retryRaw);
      } catch (err) {
        console.error('Score JSON parse failed:', err);
        return NextResponse.json({ success: false, error: 'Scoring failed.' }, { status: 500 });
      }
    }

    const score = Math.min(5, Math.max(1, Math.round(result.score)));
    return NextResponse.json({
      success: true,
      data: { personaId, score, reasoning: result.reasoning },
    });
  } catch (error) {
    console.error('score-response error:', error);
    return NextResponse.json(
      { success: false, error: 'Something went wrong. Please try again.' },
      { status: 500 }
    );
  }
}
