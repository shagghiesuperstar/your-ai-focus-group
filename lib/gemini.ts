import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);

const MODEL = 'gemini-2.5-flash';

async function runGemini(
  prompt: string,
  systemPrompt: string,
  temperature: number,
  maxTokens = 4096
): Promise<string> {
  const model = genAI.getGenerativeModel({
    model: MODEL,
    generationConfig: {
      temperature,
      maxOutputTokens: maxTokens,
    },
  });

  const result = await model.generateContent({
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    systemInstruction: { role: 'system', parts: [{ text: systemPrompt }] },
  });

  return result.response.text();
}

/** T=0.7 — use for generation tasks (personas, interviews, synthesis). */
export async function callGemini(
  prompt: string,
  systemPrompt: string,
  maxTokens = 4096
): Promise<string> {
  return runGemini(prompt, systemPrompt, 0.7, maxTokens);
}

/**
 * T=0.3 — use for rating/scoring tasks.
 * Paper §4.1: lower temperature reduces variance in Likert rating without
 * sacrificing the distributional accuracy gains from the FLR two-step.
 */
export async function callGeminiLowTemp(
  prompt: string,
  systemPrompt: string,
  maxTokens = 4096
): Promise<string> {
  return runGemini(prompt, systemPrompt, 0.3, maxTokens);
}

export function parseJSON<T = unknown>(text: string): T {
  const cleaned = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
  return JSON.parse(cleaned) as T;
}
