/**
 * Stage 4 — embed.
 * Embeds each contextualized chunk locally with bge-small-en-v1.5 (384-dim) via
 * the shared lib/embeddings module — the SAME code the live query path uses, so
 * passage and query vectors share one space.
 *
 * Writes JSONL (one record per line) so a 30k+ row corpus streams to disk and the
 * run is resumable: re-running skips chunks already embedded.
 *
 * Input:  data/chunks_ctx.json
 * Output: data/embedded.jsonl
 */
import { readFileSync, existsSync, appendFileSync, readFileSync as rf } from "node:fs";
import { embedPassages } from "../lib/embeddings.ts";

const IN = "data/chunks_ctx.json";
const OUT = "data/embedded.jsonl";
const BATCH = 64;

type Chunk = {
  id: string;
  docId: string;
  date: string;
  title: string;
  videoId: string | null;
  url: string;
  chunkIndex: number;
  wordOffset: number;
  estSeconds: number;
  text: string;
  embedText: string;
};

const chunks: Chunk[] = JSON.parse(readFileSync(IN, "utf8"));

// Resume support: collect ids already present in the output.
const done = new Set<string>();
if (existsSync(OUT)) {
  for (const line of rf(OUT, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try {
      done.add(JSON.parse(line).id);
    } catch {
      /* ignore partial last line */
    }
  }
}

const todo = chunks.filter((c) => !done.has(c.id));
console.log(`to embed: ${todo.length.toLocaleString()} (${done.size.toLocaleString()} already done)`);

const t0 = Date.now();
for (let i = 0; i < todo.length; i += BATCH) {
  const batch = todo.slice(i, i + BATCH);
  const vectors = await embedPassages(batch.map((c) => c.embedText));

  const lines = batch.map((c, j) => {
    const { embedText, ...keep } = c; // don't persist the header-prefixed text
    return JSON.stringify({ ...keep, embedding: vectors[j] });
  });
  appendFileSync(OUT, lines.join("\n") + "\n");

  const n = i + batch.length;
  if (n % (BATCH * 10) === 0 || n === todo.length) {
    const rate = n / ((Date.now() - t0) / 1000);
    const eta = Math.round((todo.length - n) / rate);
    console.log(`  ${n.toLocaleString()}/${todo.length.toLocaleString()}  (${rate.toFixed(1)}/s, eta ${eta}s)`);
  }
}
console.log(`done -> ${OUT}`);
