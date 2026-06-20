/**
 * Cross-encoder reranking via the Hugging Face Inference API (bge-reranker-base).
 *
 * Serverless-friendly counterpart to lib/rerank.ts (which needs the local ONNX
 * runtime that can't fit in a Vercel function). Sends all candidate (query,
 * passage) pairs in ONE request and gets a relevance score (sigmoid, 0-1) per
 * pair, so the whole pool is reranked with a single network call.
 */
const HF_RERANK_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-reranker-base";

export type RerankItem = { id: string; text: string };
export type RerankResult = { id: string; score: number };

export async function rerankRemote(query: string, items: RerankItem[]): Promise<RerankResult[]> {
  const token = process.env.HF_TOKEN;
  if (!token || items.length === 0) throw new Error("rerank unavailable");

  const res = await fetch(HF_RERANK_URL, {
    method: "POST",
    headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({
      inputs: items.map((it) => ({ text: query, text_pair: it.text.slice(0, 1200) })),
      options: { wait_for_model: true },
    }),
  });
  if (!res.ok) throw new Error(`HF rerank ${res.status}: ${(await res.text()).slice(0, 150)}`);

  // Response: [[{label, score}, ...]] in input order (occasionally un-nested).
  const data = await res.json();
  const rows: any[] = Array.isArray(data[0]) ? data[0] : data;
  const scores = rows.map((r) => (typeof r?.score === "number" ? r.score : Array.isArray(r) ? r[0]?.score ?? 0 : 0));

  return items
    .map((it, i) => ({ id: it.id, score: scores[i] ?? 0 }))
    .sort((a, b) => b.score - a.score);
}
