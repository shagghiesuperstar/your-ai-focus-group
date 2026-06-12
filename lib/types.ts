export interface ExtractedContext {
  productCategory: string;
  emotionalProblems: string[];
  useCases: string[];
  targetContext: string;
}

export interface Persona {
  id: string;
  name: string;
  age: number;
  occupation: string;
  personality: string;
  emotionalTendencies: string;
  goals: string;
  fears: string;
}

export interface InterviewResponse {
  personaId: string;
  response: string;
}

/**
 * scorePmf — optional probability mass function over Likert points 1-5.
 * Present when the SSR pipeline ran (OPENAI_API_KEY configured).
 * Index 0 = point 1, index 4 = point 5.
 * Sums to 1.0.
 */
export interface ScoredResponse extends InterviewResponse {
  score: number;
  evScore?: number;
  scorePmf?: [number, number, number, number, number];
  method?: 'ssr' | 'flr';
  reasoning: string;
}

export interface QuestionRound {
  question: string;
  responses: Map<string, string>;
  scoredResponses: ScoredResponse[];
  isScoring: boolean;
}

export interface AttributedInsight {
  text: string;
  personas?: string[];
}

export interface SynthesisResult {
  averageScore: number;
  overallSentiment: string;
  whatWorked: Array<string | AttributedInsight>;
  whatDidnt: Array<string | AttributedInsight>;
  surprises: string | null;
  recommendation: string;
}

export type AppStep =
  | 'input'
  | 'extracting'
  | 'selecting-personas'
  | 'session'
  | 'results';
