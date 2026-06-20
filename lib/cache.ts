/**
 * Semantic answer cache, backed by Upstash Vector.
 *
 * Exact-match caching misses paraphrases ("cold plunge benefits" vs "benefits of
 * cold exposure"). Instead we index each answered query by its embedding and, on
 * a new query, look up the nearest cached query — if it's close enough, reuse the
 * cached answer. This cuts latency and LLM cost for semantically-repeated asks.
 *
 * Reuses the query embedding already computed for retrieval (no extra model call).
 * Gracefully no-ops when Upstash Vector isn't configured.
 */
import { Index } from "@upstash/vector";

const SIMILARITY_THRESHOLD = 0.97; // cosine; high so only near-duplicate queries hit

type CachedAnswer = { answer: string; sources: unknown[]; query: string };

let index: Index | null | undefined;

function getIndex(): Index | null {
  if (index !== undefined) return index;
  const url = process.env.UPSTASH_VECTOR_REST_URL;
  const token = process.env.UPSTASH_VECTOR_REST_TOKEN;
  index = url && token ? new Index({ url, token }) : null;
  return index;
}

export async function getCachedAnswer(embedding: number[]): Promise<CachedAnswer | null> {
  const idx = getIndex();
  if (!idx) return null;
  try {
    const [hit] = await idx.query({ vector: embedding, topK: 1, includeMetadata: true });
    if (hit && hit.score >= SIMILARITY_THRESHOLD && hit.metadata) {
      return hit.metadata as unknown as CachedAnswer;
    }
  } catch {
    /* cache is best-effort */
  }
  return null;
}

export async function setCachedAnswer(embedding: number[], value: CachedAnswer): Promise<void> {
  const idx = getIndex();
  if (!idx) return;
  try {
    // id keyed by query so re-asking the same thing overwrites rather than dupes.
    const id = `q:${value.query.toLowerCase().replace(/\s+/g, " ").trim().slice(0, 200)}`;
    await idx.upsert({ id, vector: embedding, metadata: value as unknown as Record<string, unknown> });
  } catch {
    /* best-effort */
  }
}
