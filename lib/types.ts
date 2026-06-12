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
 * ScorePmf — probability mass function over Likert points 1–5.
 * Present on ScoredResponse when the SSR pipeline ran.
 * p1..p5 sum to 1.0.
 * The paper (arXiv:2510.08338) shows distributional output from SSR
 * achieves KS similarity ≈ 0.88 vs ≈ 0.72 for FLR.
 */
export interface ScorePmf {
  p1: number;
  p2: number;
  p3: number;
  p4: number;
  p5: number;
}

export interface ScoredResponse extends InterviewResponse {
  score: number;        // rounded integer 1–5 (point estimate)
  evScore?: number;     // expected value from pmf before rounding (SSR only)
  scorePmf?: ScorePmf; // full distribution (SSR only)
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
