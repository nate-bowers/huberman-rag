/**
 * Stage 3 — Contextual Retrieval (Anthropic's technique, episode-scoped variant).
 *
 * Each chunk, in isolation, often loses what episode/topic it belongs to ("he
 * recommends 13 minutes" — of what?). We prepend a short context header so the
 * embedding (and keyword index) captures that situating information.
 *
 * To stay within free LLM limits we generate ONE overview per episode (342 calls)
 * rather than one per chunk (32k calls), then reuse it across that episode's
 * chunks. Summaries are cached to data/summaries.json so re-runs are incremental.
 *
 * Set GROQ_API_KEY to generate real overviews. Without it, the script falls back
 * to a deterministic title/date header so the pipeline still runs end to end.
 *
 * Input:  data/chunks.json
 * Output: data/chunks_ctx.json  (adds `embedText`)  +  data/summaries.json (cache)
 */
import "dotenv/config";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { createGroq } from "@ai-sdk/groq";
import { generateText } from "ai";

type Chunk = {
  id: string;
  docId: string;
  date: string;
  title: string;
  text: string;
  [k: string]: unknown;
};

// Groq free tier is capped at ~6000 tokens/min for llama-3.1-8b-instant. We run
// serially and pace requests against a sliding 60s token budget so we never trip
// the TPM wall (and let the SDK honor retry-after if we ever do).
const SUMMARY_MODEL = process.env.GROQ_SUMMARY_MODEL ?? "llama-3.1-8b-instant";
const TPM_BUDGET = 5400; // headroom under the 6000 cap
const EST_TOKENS_PER_CALL = 1000;

const tokenLog: { t: number; tokens: number }[] = [];
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitForTokenBudget(est: number) {
  for (;;) {
    const now = Date.now();
    while (tokenLog.length && now - tokenLog[0].t > 60_000) tokenLog.shift();
    const used = tokenLog.reduce((s, e) => s + e.tokens, 0);
    if (used + est <= TPM_BUDGET) return;
    const waitMs = 60_000 - (now - tokenLog[0].t) + 250;
    await sleep(Math.max(waitMs, 500));
  }
}

const chunks: Chunk[] = JSON.parse(readFileSync("data/chunks.json", "utf8"));

// Group chunks by episode.
const byDoc = new Map<string, Chunk[]>();
for (const c of chunks) {
  if (!byDoc.has(c.docId)) byDoc.set(c.docId, []);
  byDoc.get(c.docId)!.push(c);
}

const summaries: Record<string, string> = existsSync("data/summaries.json")
  ? JSON.parse(readFileSync("data/summaries.json", "utf8"))
  : {};

const apiKey = process.env.GROQ_API_KEY;
const useLLM = Boolean(apiKey);
const groq = useLLM ? createGroq({ apiKey }) : null;

async function summarizeEpisode(title: string, fullText: string): Promise<string> {
  // Sample across the episode (start + middle) so the overview reflects the whole
  // thing, not just the sponsor-heavy intro.
  // Keep the sample small to respect free-tier token/min caps (~850 tokens/call).
  const words = fullText.split(/\s+/);
  const head = words.slice(0, 350).join(" ");
  const mid = words.slice(Math.floor(words.length / 2), Math.floor(words.length / 2) + 200).join(" ");
  const sample = `${head}\n...\n${mid}`;

  await waitForTokenBudget(EST_TOKENS_PER_CALL);
  const { text, usage } = await generateText({
    model: groq!(SUMMARY_MODEL),
    temperature: 0.2,
    maxRetries: 6, // SDK honors Groq's retry-after on 429
    prompt:
      `You are indexing a Huberman Lab podcast episode titled "${title}".\n` +
      `Write ONE sentence (max 35 words) describing the episode's main topic and the ` +
      `key subjects, protocols, or people it covers. No preamble, just the sentence.\n\n` +
      `Transcript sample:\n"""${sample}"""`,
  });
  tokenLog.push({ t: Date.now(), tokens: usage?.totalTokens ?? EST_TOKENS_PER_CALL });
  return text.trim().replace(/\s+/g, " ");
}

async function ensureSummaries() {
  if (!useLLM) {
    console.log("⚠ GROQ_API_KEY not set — using deterministic fallback headers (no LLM overviews).");
    return;
  }
  const docs = [...byDoc.entries()].filter(([id]) => !summaries[id]);
  const total = docs.length;
  console.log(`generating overviews for ${total} episodes (${Object.keys(summaries).length} cached)…`);

  const t0 = Date.now();
  let done = 0;
  for (const [docId, group] of docs) {
    const { title } = group[0];
    const fullText = group.map((c) => c.text).join(" ");
    try {
      summaries[docId] = await summarizeEpisode(title, fullText);
    } catch (err: any) {
      console.warn(`  ! failed ${docId}: ${err?.message ?? err}`);
      summaries[docId] = ""; // fallback header for this one
    }
    if (++done % 10 === 0 || done === total) {
      writeFileSync("data/summaries.json", JSON.stringify(summaries, null, 2));
      const rate = done / ((Date.now() - t0) / 1000);
      const eta = Math.round((total - done) / rate / 60);
      console.log(`  ${done}/${total}  (${(rate * 60).toFixed(1)}/min, eta ${eta}m)`);
    }
  }
  writeFileSync("data/summaries.json", JSON.stringify(summaries, null, 2));
}

function header(c: Chunk): string {
  const overview = summaries[c.docId];
  const base = `Episode: "${c.title}" (Huberman Lab, ${c.date}).`;
  return overview ? `${base} Overview: ${overview}` : base;
}

await ensureSummaries();

const out = chunks.map((c) => ({
  ...c,
  embedText: `${header(c)}\n\n${c.text}`,
}));

writeFileSync("data/chunks_ctx.json", JSON.stringify(out, null, 2));
console.log(`wrote data/chunks_ctx.json (${out.length} chunks, mode: ${useLLM ? "LLM overviews" : "fallback"})`);
console.log("sample embedText header:\n  " + header(chunks[0]));
