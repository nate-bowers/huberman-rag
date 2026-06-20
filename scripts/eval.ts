/**
 * Eval harness — rigorous version.
 *
 * (A) Retrieval ablation: semantic-only vs keyword-only vs hybrid (RRF).
 *     Relevance is decided by an LLM judge (binary YES/NO) over the POOLED
 *     candidates of all three methods — so it is unbiased between lexical and
 *     semantic retrieval (a lexical "contains the phrase" proxy would unfairly
 *     favor keyword search). Judgments are cached to data/eval-judgments.json.
 *     Metrics: precision@1, hit-rate@k, MRR, nDCG@k.
 *
 * (B) Generation faithfulness: answer from the hybrid context, then LLM-judge
 *     whether the answer is supported by that context (1-5 → normalized 0-1).
 *
 * Judge/gen calls are paced under Groq's free-tier ~6000 TPM cap.
 *
 *   pnpm exec tsx scripts/eval.ts
 */
import "dotenv/config";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { createGroq } from "@ai-sdk/groq";
import { generateText } from "ai";
import { semanticRanked, keywordRanked, rrfFuse, type Row } from "../lib/retrieval.ts";
import { embedQuery } from "../lib/embeddings.ts";
import { rerank } from "../lib/rerank.ts";

const K = 6;
const RERANK_POOL = 30; // hybrid candidates handed to the cross-encoder
const FAITH_SAMPLE = 25; // faithfulness is slow; sample rather than all 100
const golden: { q: string; level?: string }[] = JSON.parse(readFileSync("eval/golden.json", "utf8"));
const rows: Row[] = readFileSync("data/embedded.jsonl", "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));
const byId = new Map(rows.map((r) => [r.id, r]));

const groq = createGroq({ apiKey: process.env.GROQ_API_KEY });
const GEN_MODEL = process.env.GROQ_MODEL ?? "llama-3.3-70b-versatile";
const JUDGE_MODEL = "llama-3.1-8b-instant";

// ── free-tier token pacing (sliding 60s window) ──────────────────────────────
const TPM_BUDGET = 5200;
const tokenLog: { t: number; tokens: number }[] = [];
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
async function pace(est: number) {
  for (;;) {
    const now = Date.now();
    while (tokenLog.length && now - tokenLog[0].t > 60_000) tokenLog.shift();
    const used = tokenLog.reduce((s, e) => s + e.tokens, 0);
    if (used + est <= TPM_BUDGET) return;
    await sleep(Math.max(60_000 - (now - tokenLog[0].t) + 250, 500));
  }
}
function words(s: string, n: number) {
  return s.split(/\s+/).slice(0, n).join(" ");
}

// ── relevance judge (cached) ─────────────────────────────────────────────────
const judgments: Record<string, 0 | 1> = existsSync("data/eval-judgments.json")
  ? JSON.parse(readFileSync("data/eval-judgments.json", "utf8"))
  : {};

async function judgeRelevance(q: string, row: Row): Promise<0 | 1> {
  const key = `${q}::${row.id}`; // keyed by question text, robust to reordering
  if (key in judgments) return judgments[key];
  await pace(350);
  try {
    const { text, usage } = await generateText({
      model: groq(JUDGE_MODEL),
      temperature: 0,
      maxRetries: 6,
      prompt:
        `Is the excerpt RELEVANT to answering the question (does it contain information that helps answer it)? ` +
        `Reply with only YES or NO.\n\nQuestion: ${q}\n\nExcerpt: ${words(row.text, 120)}`,
    });
    tokenLog.push({ t: Date.now(), tokens: usage?.totalTokens ?? 350 });
    const rel: 0 | 1 = /yes/i.test(text) ? 1 : 0;
    judgments[key] = rel;
    return rel;
  } catch (e: any) {
    // A single failed judge call must not crash the whole run. Treat as not
    // relevant for this pass and don't cache, so a re-run can retry it.
    console.warn(`  ! judge failed (treating as 0): ${e?.message ?? e}`);
    return 0;
  }
}

// ── metrics ──────────────────────────────────────────────────────────────────
function dcg(rels: number[]): number {
  return rels.reduce((s, r, i) => s + r / Math.log2(i + 2), 0);
}
function metricsFor(rels: number[], relevantInPool: number) {
  const top = rels.slice(0, K);
  const firstRel = top.findIndex((r) => r === 1);
  const idcg = dcg(Array(Math.min(relevantInPool, K)).fill(1));
  return {
    p1: top[0] ?? 0,
    hit: top.some((r) => r === 1) ? 1 : 0,
    rr: firstRel >= 0 ? 1 / (firstRel + 1) : 0,
    ndcg: idcg > 0 ? dcg(top) / idcg : 0,
  };
}

// ── faithfulness ─────────────────────────────────────────────────────────────
async function faithfulness(q: string, contextRows: Row[]): Promise<number | null> {
  const genCtx = contextRows.map((r, i) => `[${i + 1}] ${r.text}`).join("\n\n");
  await pace(2600);
  const { text: answer, usage: u1 } = await generateText({
    model: groq(GEN_MODEL),
    temperature: 0.3,
    maxRetries: 6,
    prompt: `Answer using ONLY these excerpts, cite [n].\n\nQuestion: ${q}\n\n${genCtx}\n\nAnswer:`,
  });
  tokenLog.push({ t: Date.now(), tokens: u1?.totalTokens ?? 2600 });
  // Judge with a TRUNCATED context to stay under the 6000-token single-request cap.
  const judgeCtx = contextRows.slice(0, 4).map((r, i) => `[${i + 1}] ${words(r.text, 150)}`).join("\n\n");
  await pace(1400);
  const { text: verdict, usage: u2 } = await generateText({
    model: groq(JUDGE_MODEL),
    temperature: 0,
    maxRetries: 6,
    prompt:
      `Rate how well the ANSWER is supported by the CONTEXT, 1-5 (5 = every claim supported, ` +
      `1 = mostly unsupported). Reply with ONLY the integer.\n\nCONTEXT:\n${judgeCtx}\n\nANSWER:\n${words(answer, 220)}`,
  });
  tokenLog.push({ t: Date.now(), tokens: u2?.totalTokens ?? 1400 });
  const n = parseInt(verdict.match(/[1-5]/)?.[0] ?? "", 10);
  return Number.isFinite(n) ? (n - 1) / 4 : null;
}

// ── run ──────────────────────────────────────────────────────────────────────
const METHODS = ["semantic", "keyword", "hybrid", "hybrid_rerank"] as const;
type Method = (typeof METHODS)[number];
const blank = () => ({ p1: 0, hit: 0, rr: 0, ndcg: 0 });
const agg: Record<Method, ReturnType<typeof blank>> = {
  semantic: blank(), keyword: blank(), hybrid: blank(), hybrid_rerank: blank(),
};
// per-difficulty hit-rate for the best method (hybrid_rerank)
const byLevel: Record<string, { hit: number; n: number }> = {};
const faithScores: number[] = [];

console.log(`Evaluating ${golden.length} questions over ${rows.length.toLocaleString()} chunks (LLM-judged)…\n`);

for (let qi = 0; qi < golden.length; qi++) {
  const q = golden[qi].q;
  const level = golden[qi].level ?? "n/a";
  const qVec = await embedQuery(q);

  const sem = semanticRanked(qVec, rows);
  const kw = keywordRanked(q, rows);
  const hybridFull = rrfFuse(sem, kw);

  const semIds = sem.slice(0, K).map((x) => x.id);
  const kwIds = kw.slice(0, K).map((x) => x.id);
  const hyIds = hybridFull.slice(0, K).map((x) => x.id);

  // hybrid → cross-encoder rerank of the top RERANK_POOL candidates → top K
  const poolForRerank = hybridFull.slice(0, RERANK_POOL).map((x) => byId.get(x.id)!);
  const reranked = await rerank(q, poolForRerank.map((r) => ({ id: r.id, text: r.text })));
  const rrIds = reranked.slice(0, K).map((x) => x.id);

  const ids: Record<Method, string[]> = { semantic: semIds, keyword: kwIds, hybrid: hyIds, hybrid_rerank: rrIds };

  // Judge the pooled candidate set of all methods once (cached).
  const poolToJudge = [...new Set(METHODS.flatMap((mm) => ids[mm]))];
  const rel = new Map<string, number>();
  for (const id of poolToJudge) rel.set(id, await judgeRelevance(q, byId.get(id)!));
  writeFileSync("data/eval-judgments.json", JSON.stringify(judgments));
  const relevantInPool = [...rel.values()].filter((r) => r === 1).length;

  for (const mm of METHODS) {
    const m = metricsFor(ids[mm].map((id) => rel.get(id)!), relevantInPool);
    agg[mm].p1 += m.p1; agg[mm].hit += m.hit; agg[mm].rr += m.rr; agg[mm].ndcg += m.ndcg;
  }
  byLevel[level] ??= { hit: 0, n: 0 };
  byLevel[level].n++;
  byLevel[level].hit += metricsFor(rrIds.map((id) => rel.get(id)!), relevantInPool).hit;

  // Faithfulness on a sample (slow), using the reranked context.
  if (qi < FAITH_SAMPLE) {
    try {
      const f = await faithfulness(q, rrIds.map((id) => byId.get(id)!));
      if (f != null) faithScores.push(f);
    } catch (e: any) {
      console.warn(`  ! faithfulness skipped: ${e?.message ?? e}`);
    }
  }
  console.log(`  [${qi + 1}/${golden.length}] ${level.padEnd(6)} ${relevantInPool} rel · "${q.slice(0, 44)}…"`);
}

const n = golden.length;
const pct = (x: number) => ((x / n) * 100).toFixed(1) + "%";
const f3 = (x: number) => (x / n).toFixed(3);

console.log(`\nRetrieval ablation (k=${K}, LLM-judged relevance, n=${n}):`);
console.log(`  method          P@1     hit@${K}    MRR     nDCG@${K}`);
for (const mm of METHODS) {
  console.log(`  ${mm.padEnd(14)}  ${pct(agg[mm].p1).padStart(6)}  ${pct(agg[mm].hit).padStart(6)}  ${f3(agg[mm].rr)}  ${f3(agg[mm].ndcg)}`);
}
console.log(`\nhybrid_rerank hit@${K} by difficulty:`);
for (const lvl of ["easy", "medium", "hard"]) {
  if (byLevel[lvl]) console.log(`  ${lvl.padEnd(8)} ${((byLevel[lvl].hit / byLevel[lvl].n) * 100).toFixed(1)}%  (n=${byLevel[lvl].n})`);
}

const avgFaith = faithScores.length ? faithScores.reduce((a, b) => a + b, 0) / faithScores.length : null;
console.log(avgFaith != null ? `\nFaithfulness (LLM-judge, n=${faithScores.length}): ${avgFaith.toFixed(3)} (0-1)` : `\n(faithfulness unavailable)`);

writeFileSync(
  "data/eval-results.json",
  JSON.stringify(
    {
      k: K, chunks: rows.length, questions: n,
      ablation: Object.fromEntries(METHODS.map((mm) => [mm, { p1: agg[mm].p1 / n, hit: agg[mm].hit / n, mrr: agg[mm].rr / n, ndcg: agg[mm].ndcg / n }])),
      byLevel: Object.fromEntries(Object.entries(byLevel).map(([k, v]) => [k, { hit: v.hit / v.n, n: v.n }])),
      faithfulness: avgFaith,
    },
    null, 2
  )
);
console.log(`\nwrote data/eval-results.json`);
