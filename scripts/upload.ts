/**
 * Stage 5 — upload.
 * Batch-upserts embedded chunks into Supabase. Idempotent (upsert on id), so it
 * can be re-run safely. Requires NEXT_PUBLIC_SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY.
 *
 * Input: data/embedded.jsonl
 */
import "dotenv/config";
import { readFileSync } from "node:fs";
import { serviceClient } from "../lib/supabase.ts";

// Small batches: each insert also updates the HNSW index, and large batches can
// exceed Supabase's per-statement timeout as the table grows.
const BATCH = 150;
const supabase = serviceClient();
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const allRows = readFileSync("data/embedded.jsonl", "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l))
  .map((r) => ({
    id: r.id,
    doc_id: r.docId,
    chunk_index: r.chunkIndex,
    date: r.date || null,
    title: r.title,
    video_id: r.videoId,
    url: r.url,
    word_offset: r.wordOffset,
    est_seconds: r.estSeconds,
    content: r.text,
    embedding: r.embedding,
  }));

// Resume: pull ids already in the table (paginated) and skip them.
const existing = new Set<string>();
for (let from = 0; ; from += 1000) {
  const { data, error } = await supabase.from("chunks").select("id").range(from, from + 999);
  if (error) {
    console.error("failed reading existing ids:", error.message);
    process.exit(1);
  }
  if (!data || data.length === 0) break;
  for (const r of data) existing.add(r.id);
  if (data.length < 1000) break;
}

const rows = allRows.filter((r) => !existing.has(r.id));
console.log(`total ${allRows.length.toLocaleString()}, already in table ${existing.size.toLocaleString()}, to upload ${rows.length.toLocaleString()} (batch ${BATCH})…`);

for (let i = 0; i < rows.length; i += BATCH) {
  const batch = rows.slice(i, i + BATCH);
  let ok = false;
  for (let attempt = 0; attempt < 4 && !ok; attempt++) {
    const { error } = await supabase.from("chunks").upsert(batch, { onConflict: "id" });
    if (!error) {
      ok = true;
    } else {
      console.warn(`  batch @${i} attempt ${attempt + 1} failed: ${error.message}`);
      await sleep(1500 * (attempt + 1));
    }
  }
  if (!ok) {
    console.error(`batch @${i} failed after retries — re-run to resume.`);
    process.exit(1);
  }
  if ((i / BATCH) % 10 === 0 || i + BATCH >= rows.length) {
    console.log(`  ${Math.min(i + BATCH, rows.length).toLocaleString()}/${rows.length.toLocaleString()}`);
  }
}

const { count } = await supabase.from("chunks").select("*", { count: "exact", head: true });
console.log(`done. rows in table: ${count}`);
