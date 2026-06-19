/**
 * Retrieval + generation endpoint.
 *
 * 1. Embed the user's query with the SAME bge-small model used at ingest.
 * 2. Hybrid retrieve via the Supabase match_hybrid RPC (semantic + keyword, RRF).
 * 3. Stream a grounded, citation-anchored answer from Groq.
 *
 * The retrieved sources are returned in the `x-sources` response header (base64
 * JSON, trimmed snippets) so the client can render source cards immediately while
 * the answer text streams in the body.
 */
import { createGroq } from "@ai-sdk/groq";
import { streamText } from "ai";
import { embedQuery } from "@/lib/embeddings";
import { publicClient } from "@/lib/supabase";

export const runtime = "nodejs";
export const maxDuration = 60;

const MATCH_COUNT = 6;

function fmtTime(s: number | null): string {
  if (s == null) return "";
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${String(sec).padStart(2, "0")}`;
}

export async function POST(req: Request) {
  const { query } = await req.json();
  if (!query || typeof query !== "string") {
    return new Response("Missing query", { status: 400 });
  }

  // 1. Embed query + 2. hybrid retrieve
  const queryEmbedding = await embedQuery(query);
  const supabase = publicClient();
  const { data: matches, error } = await supabase.rpc("match_hybrid", {
    query_embedding: queryEmbedding,
    query_text: query,
    match_count: MATCH_COUNT,
  });

  if (error) {
    return new Response(`Retrieval error: ${error.message}`, { status: 500 });
  }

  const sources = (matches ?? []) as Array<{
    id: string;
    title: string;
    date: string;
    url: string;
    est_seconds: number | null;
    content: string;
    semantic_rank: number | null;
    keyword_rank: number | null;
    score: number;
  }>;

  // Guardrail: nothing retrieved → don't hallucinate.
  if (sources.length === 0) {
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(
          new TextEncoder().encode(
            "I couldn't find anything about that in the Huberman Lab episodes I've indexed. Try rephrasing, or ask about a topic he's covered (sleep, dopamine, focus, fitness, etc.)."
          )
        );
        c.close();
      },
    });
    return new Response(stream, { headers: { "content-type": "text/plain; charset=utf-8" } });
  }

  // Build the grounded context block.
  const context = sources
    .map(
      (s, i) =>
        `[${i + 1}] Episode: "${s.title}" (${s.date}${
          s.est_seconds != null ? `, ~${fmtTime(s.est_seconds)}` : ""
        })\n${s.content}`
    )
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
  });

  // Trim source content for the header payload (cards show a snippet only).
  const headerSources = sources.map((s, i) => ({
    n: i + 1,
    title: s.title,
    date: s.date,
    url: s.url,
    timestamp: fmtTime(s.est_seconds),
    snippet: s.content.slice(0, 220).trim() + (s.content.length > 220 ? "…" : ""),
    semantic_rank: s.semantic_rank,
    keyword_rank: s.keyword_rank,
  }));
  const sourcesB64 = Buffer.from(JSON.stringify(headerSources)).toString("base64");

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      for await (const delta of result.textStream) {
        controller.enqueue(encoder.encode(delta));
      }
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "x-sources": sourcesB64,
    },
  });
}
