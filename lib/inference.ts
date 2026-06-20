/**
 * Remote query embedding via Hugging Face Inference API.
 *
 * The serverless function can't bundle the ~370MB ONNX runtime that Transformers.js
 * needs, so on the live query path we call HF's hosted copy of the SAME model
 * (bge-small-en-v1.5) over HTTP. Same weights as the local ingest embedder →
 * vectors stay in the same space. This module is fetch-only (no heavy deps), so
 * the function stays tiny.
 *
 * The corpus is still embedded locally with Transformers.js (lib/embeddings.ts);
 * only the per-request query embedding goes through this API.
 */
const HF_MODEL = "BAAI/bge-small-en-v1.5";
const HF_URL = `https://router.huggingface.co/hf-inference/models/${HF_MODEL}/pipeline/feature-extraction`;
// bge query-side instruction (passages get none) — must match lib/embeddings.ts.
const QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: ";

export const EMBED_DIM = 384;

function l2normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / norm);
}

export async function embedQueryRemote(query: string): Promise<number[]> {
  const token = process.env.HF_TOKEN;
  if (!token) throw new Error("Missing HF_TOKEN");

  const res = await fetch(HF_URL, {
    method: "POST",
    headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({ inputs: QUERY_INSTRUCTION + query, options: { wait_for_model: true } }),
  });
  if (!res.ok) throw new Error(`HF embed ${res.status}: ${(await res.text()).slice(0, 200)}`);

  const data = await res.json();
  let vec: number[];
  if (Array.isArray(data) && typeof data[0] === "number") {
    vec = data as number[]; // pooled sentence embedding
  } else if (Array.isArray(data) && Array.isArray(data[0]) && typeof data[0][0] === "number") {
    // token embeddings → mean-pool
    const toks = data as number[][];
    vec = toks[0].map((_, j) => toks.reduce((s, t) => s + t[j], 0) / toks.length);
  } else {
    throw new Error("Unexpected HF embedding response shape");
  }
  if (vec.length !== EMBED_DIM) throw new Error(`Expected ${EMBED_DIM}-d, got ${vec.length}`);
  return l2normalize(vec);
}
