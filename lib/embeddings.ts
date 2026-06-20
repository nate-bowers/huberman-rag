/**
 * Shared embedding module — used by BOTH the ingestion pipeline (scripts/*) and
 * the live query path (app/api/chat). Using the exact same model + pooling in
 * both places is what guarantees the query vector lands in the same space as the
 * indexed passage vectors. If these ever diverge, retrieval silently degrades.
 *
 * Model: BAAI/bge-small-en-v1.5 (ONNX build via Xenova), 384 dimensions.
 * bge models are asymmetric: queries get a short instruction prefix, passages
 * do not. We expose embedQuery / embedPassages to keep that distinction explicit.
 */
import { pipeline, env, type FeatureExtractionPipeline } from "@huggingface/transformers";

export const EMBED_MODEL = "Xenova/bge-small-en-v1.5";
export const EMBED_DIM = 384;

// On Vercel, model files are downloaded at runtime; only /tmp is writable, so
// point the cache there. The native onnx backend is used (pruned to linux-x64
// in next.config.ts).
if (process.env.VERCEL) {
  env.cacheDir = "/tmp/transformers-cache";
}

// bge-small-en-v1.5 recommended retrieval instruction for the QUERY side only.
const QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: ";

let extractorPromise: Promise<FeatureExtractionPipeline> | null = null;

function getExtractor(): Promise<FeatureExtractionPipeline> {
  // Singleton: load the model once per process (cold start in serverless,
  // once per run in scripts). Cast through unknown: the pipeline() overloads
  // produce a union TS can't represent (TS2590) when the task is a variable.
  if (!extractorPromise) {
    extractorPromise = pipeline(
      "feature-extraction",
      EMBED_MODEL
    ) as unknown as Promise<FeatureExtractionPipeline>;
  }
  return extractorPromise;
}

async function embed(texts: string[]): Promise<number[][]> {
  const extractor = await getExtractor();
  const output = await extractor(texts, { pooling: "mean", normalize: true });
  return output.tolist() as number[][];
}

/** Embed corpus passages (indexing side). No instruction prefix. */
export async function embedPassages(texts: string[]): Promise<number[][]> {
  return embed(texts);
}

/** Embed a single search query (retrieval side). Adds the bge query instruction. */
export async function embedQuery(query: string): Promise<number[]> {
  const [vec] = await embed([QUERY_INSTRUCTION + query]);
  return vec;
}
