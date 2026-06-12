/**
 * Input validation utilities for user-supplied free text.
 *
 * All user input that flows into LLM prompts must pass through these helpers
 * before use. The goals are:
 *   1. Length enforcement — prevent prompt bloat / runaway token spend.
 *   2. Control-character stripping — remove C0/DEL chars that can confuse
 *      tokenisers or create invisible prompt segments.
 *   3. Injection detection — reject common prompt-override patterns before
 *      the text reaches an LLM system prompt or user turn.
 *
 * This is defence-in-depth, not a complete sandbox. A determined attacker
 * with API access can craft inputs that evade keyword detection. The primary
 * value is protecting the hosted (Vercel) deployment from casual abuse and
 * accidental override via user error.
 */

/** C0 control characters and DEL, excluding \t, \n, \r which are safe. */
const CONTROL_CHARS = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g;

/**
 * Common prompt-injection / jailbreak patterns.
 * Patterns are intentionally broad to catch variations.
 * False-positive rate is acceptable — legitimate concept descriptions
 * will not contain these phrases.
 */
const INJECTION_PATTERNS: RegExp[] = [
  /ignore (all |the |previous |above )*instructions?/i,
  /disregard (all |the |previous |above )*instructions?/i,
  /forget (all |the |previous |above )*instructions?/i,
  /reveal (your )?(system|hidden|original) prompt/i,
  /print (your )?(system|hidden|original) prompt/i,
  /show (me )?(your )?(system|hidden|original) prompt/i,
  /you are now/i,
  /act as (if )?you (are|were)/i,
  /developer (mode|message|override)/i,
  /jailbreak/i,
  /\bDAN\b/,
  /tool[_\s-]?call/i,
  /function[_\s-]?call/i,
  /<\/?system>/i,
  /<\/?prompt>/i,
  /\[INST\]/i,
  /\[\/?SYS\]/i,
];

/**
 * Sanitise a user-supplied free-text field.
 *
 * Steps:
 *   1. Strip C0/DEL control characters.
 *   2. Collapse all runs of whitespace to a single space.
 *   3. Trim leading/trailing whitespace.
 *   4. Hard-truncate to `maxLength` characters.
 *
 * @param value     Raw string from request body.
 * @param maxLength Maximum allowed character count after sanitisation.
 * @returns         Cleaned string (may be empty — caller must validate length).
 */
export function sanitiseFreeText(value: string, maxLength: number): string {
  return value
    .replace(CONTROL_CHARS, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, maxLength);
}

/**
 * Returns true if the text matches any known prompt-injection pattern.
 * Call this after sanitiseFreeText so whitespace normalisation has run.
 *
 * @param value Sanitised user input.
 */
export function containsPromptInjection(value: string): boolean {
  return INJECTION_PATTERNS.some(pattern => pattern.test(value));
}
