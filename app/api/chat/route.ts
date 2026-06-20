/**
 * Retrieval + generation endpoint (serverless-friendly).
 *
 *   rate-limit → embed query (HF API) → semantic-cache lookup
 *     → hybrid retrieve (RRF) → grounded, cited answer → store in cache
 *
 * NOTE: the cross-encoder reranker (lib/rerank.ts) is NOT run here — bundling its
 * ONNX runtime exceeds Vercel's 250MB function limit. Reranking is exercised in
 * the eval harness (where its gains are measured) and runs locally. The live
 * function uses hybrid RRF retrieval and embeds the query via the HF Inference
 * API (same bge-small model as ingest, so vectors share one space).
 *
 * Sources ride in the `x-sources` response header (base64 JSON) so the client can
 * render source cards immediately while the answer streams in the body.
 */
import { createGroq } from "@ai-sdk/groq";
import { streamText } from "ai";
import { embedQueryRemote } from "@/lib/inference";
import { publicClient } from "@/lib/supabase";
import { checkRateLimit } from "@/lib/ratelimit";
import { getCachedAnswer, setCachedAnswer } from "@/lib/cache";

export const runtime = "nodejs";
export const maxDuration = 60;

const TOP_K = 6;

function fmtTime(s: number | null): string {
  if (s == null) return "";
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function textStream(text: string, headers: Record<string, string> = {}): Response {
  const stream = new ReadableStream({
    start(c) {
      c.enqueue(new TextEncoder().encode(text));
      c.close();
    },
  });
  return new Response(stream, { headers: { "content-type": "text/plain; charset=utf-8", ...headers } });
}

type Source = {
  id: string;
  title: string;
  date: string;
  url: string;
  est_seconds: number | null;
  content: string;
  semantic_rank: number | null;
  keyword_rank: number | null;
};

export async function POST(req: Request) {
  const { query } = await req.json();
  if (!query || typeof query !== "string") return new Response("Missing query", { status: 400 });
  if (query.length > 500) return new Response("Query too long (max 500 chars)", { status: 413 });

  // 0. Rate limit (per IP).
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "anon";
  const rl = await checkRateLimit(ip);
  if (!rl.ok) return new Response("Rate limit exceeded — try again in a minute.", { status: 429 });

  // 1. Embed query via HF (reused for cache lookup + retrieval).
  let queryEmbedding: number[];
  try {
    queryEmbedding = await embedQueryRemote(query);
  } catch (e: any) {
    return new Response(`Embedding error: ${e?.message ?? e}`, { status: 502 });
  }

  // 2. Semantic cache: reuse a previous answer to a near-identical question.
  const cached = await getCachedAnswer(queryEmbedding);
  if (cached) {
    return textStream(cached.answer, {
      "x-sources": Buffer.from(JSON.stringify(cached.sources)).toString("base64"),
      "x-cache": "hit",
    });
  }

  // 3. Hybrid retrieve (semantic + keyword, RRF) via Supabase RPC.
  const supabase = publicClient();
  const { data: matches, error } = await supabase.rpc("match_hybrid", {
    query_embedding: queryEmbedding,
    query_text: query,
    match_count: TOP_K,
  });
  if (error) return new Response(`Retrieval error: ${error.message}`, { status: 500 });

  const topSources = (matches ?? []) as Source[];
  if (topSources.length === 0) {
    return textStream(
      "I couldn't find anything about that in the Huberman Lab episodes I've indexed. Try a topic he's covered — sleep, dopamine, focus, fitness, etc."
    );
  }

  // 4. Grounded generation.
  const context = topSources
    .map((s, i) => `[${i + 1}] Episode: "${s.title}" (${s.date}${s.est_seconds != null ? `, ~${fmtTime(s.est_seconds)}` : ""})\n${s.content}`)
    .join("\n\n");

  const system =
    "You are a knowledgeable assistant answering questions using ONLY the provided excerpts " +
    "from the Huberman Lab podcast. Rules:\n" +
    "- Answer strictly from the excerpts. If they don't contain the answer, say so plainly.\n" +
    "- Cite sources inline with bracketed numbers like [1], [2] that map to the excerpts.\n" +
    "- Be specific: name protocols, durations, and mechanisms when the excerpts give them.\n" +
    "- Be concise and clear. Do not invent studies, numbers, or recommendations.";

  const groq = createGroq({ apiKey: process.env.GROQ_API_KEY });
  const model = process.env.GROQ_MODEL ?? "llama-3.3-70b-versatile";
  const result = streamText({
    model: groq(model),
    system,
    prompt: `Question: ${query}\n\nExcerpts:\n${context}\n\nAnswer (with [n] citations):`,
    temperature: 0.3,
    maxRetries: 4,
  });

  const headerSources = topSources.map((s, i) => ({
    n: i + 1,
    title: s.title,
    date: s.date,
    url: s.url,
    timestamp: fmtTime(s.est_seconds),
    snippet: s.content.slice(0, 220).trim() + (s.content.length > 220 ? "…" : ""),
    semantic_rank: s.semantic_rank,
    keyword_rank: s.keyword_rank,
  }));

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      let full = "";
      try {
        for await (const delta of result.textStream) {
          full += delta;
          controller.enqueue(encoder.encode(delta));
        }
      } catch (e: any) {
        // Generation failed (e.g. LLM rate limit) — emit a message, never empty.
        const msg =
          full.length === 0
            ? "The answer service is briefly rate-limited. Please try again in a moment."
            : "\n\n[Answer was cut off — the LLM service is rate-limited. Please retry.]";
        controller.enqueue(encoder.encode(msg));
        controller.close();
        return;
      }
      // Only cache complete, non-empty answers.
      if (full.trim()) await setCachedAnswer(queryEmbedding, { answer: full, sources: headerSources, query });
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "x-sources": Buffer.from(JSON.stringify(headerSources)).toString("base64"),
      "x-cache": "miss",
    },
  });
}
