/**
 * Cross-encoder reranker (stage 2 of retrieval).
 *
 * Bi-encoder retrieval (embeddings) scores query and doc independently — fast,
 * but coarse. A cross-encoder reads the (query, doc) pair jointly and scores
 * relevance directly, which is far more accurate but too slow to run over the
 * whole corpus. So we retrieve a wide candidate pool with hybrid search, then
 * rerank just those candidates here and keep the top few.
 *
 * Model: bge-reranker-base via Transformers.js (WASM on Vercel, native locally).
 */
import {
  AutoTokenizer,
  AutoModelForSequenceClassification,
  env,
  type PreTrainedTokenizer,
  type PreTrainedModel,
} from "@huggingface/transformers";

const RERANK_MODEL = "Xenova/bge-reranker-base";
if (process.env.VERCEL) env.cacheDir = "/tmp/transformers-cache";

let tokenizerPromise: Promise<PreTrainedTokenizer> | null = null;
let modelPromise: Promise<PreTrainedModel> | null = null;

function load() {
  if (!tokenizerPromise) {
    tokenizerPromise = AutoTokenizer.from_pretrained(RERANK_MODEL);
    modelPromise = AutoModelForSequenceClassification.from_pretrained(
      RERANK_MODEL
    ) as Promise<PreTrainedModel>;
  }
  return Promise.all([tokenizerPromise, modelPromise!]);
}

export type RerankItem = { id: string; text: string };
export type RerankResult = { id: string; score: number };

/**
 * Rerank `items` against `query`, returning all items sorted by cross-encoder
 * relevance (sigmoid of the logit, 0-1). Caller slices to top-N.
 */
export async function rerank(query: string, items: RerankItem[]): Promise<RerankResult[]> {
  if (items.length === 0) return [];
  const [tokenizer, model] = await load();

  const inputs = tokenizer(
    items.map(() => query),
    { text_pair: items.map((it) => it.text), padding: true, truncation: true }
  );
  const output: any = await model(inputs);
  // bge-reranker emits a single logit per pair; sigmoid → relevance score.
  const scores: number[] = output.logits.sigmoid().tolist().map((row: number[]) => row[0]);

  return items
    .map((it, i) => ({ id: it.id, score: scores[i] }))
    .sort((a, b) => b.score - a.score);
}
