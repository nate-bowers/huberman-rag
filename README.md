<div align="center">

# Huberman GPT

**Ask anything across ~800 hours of the Huberman Lab podcast and get answers grounded in the transcripts, with cited sources that deep-link to the moment in the video.**

[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Supabase](https://img.shields.io/badge/Supabase-pgvector-3ecf8e?logo=supabase&logoColor=white)](https://supabase.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.x-f55036)](https://groq.com/)

</div>

A retrieval-augmented generation system over every Huberman Lab episode (342 transcripts, ~7.3M words). Hybrid retrieval, contextual chunking, grounded generation with citations, and an LLM-judged evaluation harness — running end to end on free tiers.

## Ingestion pipeline

```mermaid
flowchart LR
    A[("342 transcripts<br/>~7.3M words")] --> B["parse<br/><i>title, date, match<br/>YouTube video id</i>"]
    B --> C["chunk<br/><i>sentence-aware<br/>~270 tok, 15% overlap<br/>+ word offsets</i>"]
    C --> D["context<br/><i>LLM episode overview<br/>prepended per chunk</i>"]
    D --> E["embed<br/><i>bge-small-en-v1.5<br/>384-d, local</i>"]
    E --> F[("Supabase<br/>pgvector + tsvector<br/>32,723 chunks")]

    style A fill:#1b2027,stroke:#4f9cf9,color:#e7ecf2
    style F fill:#14322a,stroke:#3ecf8e,color:#e7ecf2
    style D fill:#2a1f14,stroke:#f55036,color:#e7ecf2
    style E fill:#1d2b3f,stroke:#4f9cf9,color:#e7ecf2
```

## Query flow

```mermaid
flowchart LR
    Q["question"] --> EQ["embed query<br/><i>same bge-small</i>"]
    EQ --> H{{"match_hybrid()<br/>Postgres RPC"}}
    H --> SEM["semantic<br/>pgvector"]
    H --> KW["keyword<br/>tsvector FTS"]
    SEM --> RRF["Reciprocal<br/>Rank Fusion"]
    KW --> RRF
    RRF --> CTX["top-k chunks"]
    CTX --> LLM["Groq<br/><i>grounded, cited</i>"]
    LLM --> ANS["streamed answer<br/>+ timestamped sources"]

    style Q fill:#1d2b3f,stroke:#4f9cf9,color:#e7ecf2
    style H fill:#14322a,stroke:#3ecf8e,color:#e7ecf2
    style RRF fill:#2a2414,stroke:#e0b341,color:#e7ecf2
    style LLM fill:#2a1f14,stroke:#f55036,color:#e7ecf2
```

## Design decisions

**Chunking.** ~270-token, sentence-aware chunks with 15% overlap. The size is bounded by the embedder's 512-token limit minus the context header; overlap keeps facts that straddle a boundary retrievable. Caption-style transcripts (no punctuation) are hard-wrapped on word boundaries so one runaway "sentence" can't swallow an episode.

**Embedding model.** `bge-small-en-v1.5` — strong retrieval quality at 384 dimensions, runs locally for free. The same model embeds the corpus and the query; if those diverged, the vectors would live in different spaces and retrieval would quietly degrade.

**Vector store.** Supabase gives vector search (pgvector) and full-text search (`tsvector`) in one database, so hybrid fusion runs server-side in a single SQL function rather than being stitched together in app code.

**Retrieval quality.** Three layers, in order of impact: hybrid retrieval (semantic for meaning, keyword for exact terms) fused with RRF; contextual retrieval (each chunk carries a one-line episode overview); and a cross-encoder reranker over the hybrid candidates.

**Hallucination.** The model answers only from retrieved excerpts, cites every claim, and says so when the excerpts don't cover the question. A guardrail refuses when nothing relevant is retrieved, and an LLM judge scores answer-vs-context faithfulness in the eval.

## Evaluation

Questions are scored by an LLM judge over the pooled candidates of every method, so the comparison isn't biased toward lexical matching. Metrics: precision@1, hit-rate, MRR, nDCG.

<!-- EVAL_RESULTS -->
| method | P@1 | hit@6 | MRR | nDCG@6 |
|---|---|---|---|---|
| semantic only | 66.7% | 100.0% | 0.803 | 0.712 |
| keyword only | 72.2% | 94.4% | 0.824 | 0.632 |
| **hybrid (RRF)** | **77.8%** | **100.0%** | **0.875** | **0.732** |

Faithfulness (LLM-judged): 0.82. Hybrid wins every ranking metric; keyword-only has the weakest hit-rate and nDCG.
<!-- /EVAL_RESULTS -->

```bash
pnpm eval        # ablation + faithfulness
pnpm search "how should I use cold exposure?"
```

## Stack and cost

| Layer | Choice | Cost |
|---|---|---|
| Embeddings | bge-small-en-v1.5, local (ingest) / HF Inference API (query) | $0 |
| Vector + keyword store | Supabase — pgvector (HNSW) + Postgres FTS | $0 |
| LLM | Groq free tier (Llama 3.x) | $0 |
| Rate limit + cache | Upstash Redis + Vector | $0 |
| Hosting | Next.js on Vercel | $0 |

## Setup

```bash
pnpm install
cp .env.example .env          # GROQ + Supabase keys (HF + Upstash optional)
```

Ingest (one-time, local):

```bash
pnpm ingest:parse
pnpm ingest:chunk
pnpm ingest:context           # needs GROQ_API_KEY
pnpm ingest:embed             # ~30 min, local
# paste supabase/schema.sql into the Supabase SQL editor, then:
pnpm ingest:upload
```

Run: `pnpm dev`

## Notes and tradeoffs

- Timestamps are estimated from word position (~150 wpm); transcripts have no time codes. Exact timestamps would require aligning YouTube captions.
- Contextual retrieval is episode-scoped (one overview per episode) to fit free-tier LLM limits.
- The reranker runs in the eval and locally; the deployed serverless function uses hybrid retrieval and embeds the query via the HF Inference API, because bundling the ONNX runtime exceeds Vercel's 250 MB function limit.
- Transcripts are not included in this repo (copyright); the pipeline expects them in `huberman_transcripts/`.
