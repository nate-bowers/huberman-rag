/**
 * Offline hybrid search — quick CLI demo of the retrieval core without a DB.
 *   pnpm exec tsx scripts/search-local.ts "your question here"
 */
import { readFileSync } from "node:fs";
import { hybridSearch, type Row } from "../lib/retrieval.ts";

export function loadRows(path = "data/embedded.jsonl"): Row[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));
}

const q = process.argv.slice(2).join(" ");
if (q) {
  const rows = loadRows();
  console.log(`loaded ${rows.length.toLocaleString()} chunks\nquery: "${q}"\n`);
  const results = await hybridSearch(q, rows);
  for (const r of results) {
    const ts = `${Math.floor(r.row.estSeconds / 60)}:${String(r.row.estSeconds % 60).padStart(2, "0")}`;
    console.log(
      `score ${r.score.toFixed(4)}  sem#${r.semRank ?? "-"} kw#${r.kwRank ?? "-"}  ▶${ts}\n` +
        `  ${r.row.title} (${r.row.date})\n  ${r.row.text.slice(0, 160).replace(/\s+/g, " ")}…\n`
    );
  }
}
