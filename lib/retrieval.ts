/**
 * In-memory retrieval core (semantic / keyword / hybrid-RRF).
 *
 * This is the JS counterpart to the Supabase match_hybrid() RPC. It powers the
 * offline demo (scripts/search-local) and the eval harness (scripts/eval), so we
 * can measure and validate retrieval without a database. The production query
 * path uses the SQL function, but both implement the same RRF logic.
 */
import { embedQuery } from "./embeddings.ts";

export type Row = {
  id: string;
  title: string;
  date: string;
  url: string;
  estSeconds: number;
  text: string;
  embedding: number[];
};

export const DEFAULTS = { pool: 40, rrfK: 60, top: 6 };

export type Ranked = { id: string; score: number };

function dot(a: number[], b: number[]): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += a[i] * b[i];
  return d;
}

const STOP = new Set([
  "the", "a", "an", "of", "for", "to", "and", "is", "are", "in", "on", "how",
  "what", "does", "do", "i", "my", "with", "about", "his", "he", "you", "your",
]);
export function terms(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2 && !STOP.has(t));
}

/** Semantic ranking by cosine (unit vectors → dot product). */
export function semanticRanked(qVec: number[], rows: Row[], pool = DEFAULTS.pool): Ranked[] {
  return rows
    .map((r) => ({ id: r.id, score: dot(qVec, r.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, pool);
}

/** Keyword ranking, BM25-lite: sum of log(1+tf) over query terms in title+text. */
export function keywordRanked(query: string, rows: Row[], pool = DEFAULTS.pool): Ranked[] {
  const qTerms = terms(query);
  return rows
    .map((r) => {
      const tf = new Map<string, number>();
      for (const t of terms(r.title + " " + r.text)) tf.set(t, (tf.get(t) ?? 0) + 1);
      let score = 0;
      for (const t of qTerms) score += Math.log1p(tf.get(t) ?? 0);
      return { id: r.id, score };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, pool);
}

/** Reciprocal Rank Fusion of two ranked lists. */
export function rrfFuse(a: Ranked[], b: Ranked[], k = DEFAULTS.rrfK): Ranked[] {
  const rankA = new Map(a.map((x, i) => [x.id, i + 1]));
  const rankB = new Map(b.map((x, i) => [x.id, i + 1]));
  const ids = new Set([...rankA.keys(), ...rankB.keys()]);
  return [...ids]
    .map((id) => {
      const ra = rankA.get(id);
      const rb = rankB.get(id);
      return { id, score: (ra ? 1 / (k + ra) : 0) + (rb ? 1 / (k + rb) : 0) };
    })
    .sort((a, b) => b.score - a.score);
}

export type Hit = { row: Row; score: number; semRank: number | null; kwRank: number | null };

/** Full hybrid search: returns top results with per-retriever ranks attached. */
export async function hybridSearch(query: string, rows: Row[], top = DEFAULTS.top): Promise<Hit[]> {
  const qVec = await embedQuery(query);
  const sem = semanticRanked(qVec, rows);
  const kw = keywordRanked(query, rows);
  const fused = rrfFuse(sem, kw).slice(0, top);

  const semRank = new Map(sem.map((x, i) => [x.id, i + 1]));
  const kwRank = new Map(kw.map((x, i) => [x.id, i + 1]));
  const byId = new Map(rows.map((r) => [r.id, r]));
  return fused.map((f) => ({
    row: byId.get(f.id)!,
    score: f.score,
    semRank: semRank.get(f.id) ?? null,
    kwRank: kwRank.get(f.id) ?? null,
  }));
}
