/**
 * Sentence-aware chunker with overlap and word-offset tracking.
 *
 * Why these defaults: bge-small-en-v1.5 truncates at 512 tokens. Each chunk is
 * later prefixed with a ~40-60 token contextual header before embedding, so the
 * body must stay well under 512. ~270 words ≈ ~360 tokens leaves comfortable
 * headroom. ~40-word overlap preserves context across chunk boundaries so a fact
 * split mid-thought is still retrievable from both sides.
 *
 * wordOffset (word index of the chunk's first word within the full transcript)
 * is what later lets us estimate a video timestamp: seconds ≈ wordOffset / WPM * 60.
 */
export interface Chunk {
  text: string;
  wordOffset: number;
  wordCount: number;
}

export interface ChunkOptions {
  targetWords?: number;
  overlapWords?: number;
}

/** Split prose into sentences, collapsing whitespace first. */
export function splitSentences(text: string): string[] {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return [];
  const matches = clean.match(/[^.!?]+[.!?]+(?:["')\]]+)?|\S[^.!?]*$/g);
  return (matches ?? [clean]).map((s) => s.trim()).filter(Boolean);
}

/**
 * Hard-wrap an over-long unit on word boundaries. Needed because some
 * transcripts are auto-caption style (lowercase, almost no punctuation), so a
 * single "sentence" can run for thousands of words. Prefers comma boundaries.
 */
function hardWrap(sentence: string, maxWords: number): string[] {
  const words = sentence.split(/\s+/);
  if (words.length <= maxWords) return [sentence];
  const out: string[] = [];
  for (let i = 0; i < words.length; i += maxWords) {
    out.push(words.slice(i, i + maxWords).join(" "));
  }
  return out;
}

export function chunkText(text: string, opts: ChunkOptions = {}): Chunk[] {
  const targetWords = opts.targetWords ?? 270;
  const overlapWords = opts.overlapWords ?? 40;

  // Split into sentences, then hard-wrap any sentence longer than the target so
  // no single unit can blow past the embedding token limit.
  const sentences = splitSentences(text).flatMap((s) => hardWrap(s, targetWords));
  if (sentences.length === 0) return [];

  const sWords = sentences.map((s) => s.split(/\s+/).length);
  // prefix[i] = total words before sentence i (the chunk's wordOffset).
  const prefix = [0];
  for (const w of sWords) prefix.push(prefix[prefix.length - 1] + w);

  const chunks: Chunk[] = [];
  let start = 0;

  while (start < sentences.length) {
    let end = start;
    let words = 0;
    // Pack sentences but stop BEFORE exceeding the target, so a chunk never runs
    // past ~targetWords (hard-wrap already caps any single unit at targetWords).
    while (end < sentences.length) {
      if (words > 0 && words + sWords[end] > targetWords) break;
      words += sWords[end];
      end++;
    }

    chunks.push({
      text: sentences.slice(start, end).join(" "),
      wordOffset: prefix[start],
      wordCount: words,
    });

    if (end >= sentences.length) break;

    // Walk back from `end` to create ~overlapWords of trailing overlap.
    let back = end;
    let ov = 0;
    while (back > start + 1 && ov < overlapWords) {
      back--;
      ov += sWords[back];
    }
    start = back > start ? back : start + 1; // guarantee forward progress
  }

  return chunks;
}
