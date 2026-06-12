import { NextRequest, NextResponse } from 'next/server';
import { callGemini } from '@/lib/gemini';
import { SYSTEM_INTERVIEW, buildInterviewPrompt } from '@/lib/prompts';
import { Persona } from '@/lib/types';
import { sanitiseFreeText, containsPromptInjection } from '@/lib/validate';

const CONCEPT_MAX_CHARS = 2000;
const QUESTION_MAX_CHARS = 500;
const RESPONSE_MAX_CHARS = 1200;
const MAX_PREVIOUS_ROUNDS = 5;

interface PreviousRound {
  question: string;
  response: string;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const persona = body?.persona as Persona;

    const conceptDescription = sanitiseFreeText(
      typeof body?.conceptDescription === 'string' ? body.conceptDescription : '',
      CONCEPT_MAX_CHARS
    );
    const userQuestion = sanitiseFreeText(
      typeof body?.userQuestion === 'string' ? body.userQuestion : '',
      QUESTION_MAX_CHARS
    );

    // Cap and sanitise previous rounds to prevent history-padding attacks.
    const previousRounds: PreviousRound[] | undefined = Array.isArray(body?.previousRounds)
      ? (body.previousRounds as unknown[])
          .slice(0, MAX_PREVIOUS_ROUNDS)
          .map(round => {
            const r = round as Partial<PreviousRound>;
            return {
              question: sanitiseFreeText(typeof r?.question === 'string' ? r.question : '', QUESTION_MAX_CHARS),
              response: sanitiseFreeText(typeof r?.response === 'string' ? r.response : '', RESPONSE_MAX_CHARS),
            };
          })
      : undefined;

    if (!persona?.id || !conceptDescription || !userQuestion) {
      return NextResponse.json(
        { success: false, error: 'Missing persona, concept, or question.' },
        { status: 400 }
      );
    }

    if (containsPromptInjection(conceptDescription) || containsPromptInjection(userQuestion)) {
      return NextResponse.json(
        {
          success: false,
          error: 'Please ask a straightforward product research question. Instruction-override language is not allowed.',
        },
        { status: 400 }
      );
    }

    const response = await callGemini(
      buildInterviewPrompt(persona, conceptDescription, userQuestion, previousRounds),
      SYSTEM_INTERVIEW
    );

    return NextResponse.json({
      success: true,
      data: { personaId: persona.id, response: response.trim() },
    });
  } catch (error) {
    console.error('run-interview error:', error);
    return NextResponse.json(
      { success: false, error: 'Something went wrong during the interview. Please try again.' },
      { status: 500 }
    );
  }
}
