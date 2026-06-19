-- Huberman RAG — Supabase schema
-- Run this in the Supabase SQL editor (Dashboard -> SQL -> New query) before uploading.

-- pgvector for semantic search
create extension if not exists vector;

-- ── chunks table ────────────────────────────────────────────────────────────
create table if not exists chunks (
  id          text primary key,
  doc_id      text not null,
  chunk_index int  not null,
  date        date,
  title       text not null,
  video_id    text,
  url         text not null,          -- timestamp deep-link (…&t=NNNs)
  word_offset int,
  est_seconds int,
  content     text not null,          -- displayed chunk text (no context header)
  embedding   vector(384) not null,   -- bge-small-en-v1.5
  -- Generated full-text column: keyword side of hybrid search. Title is included
  -- so episode-topic words contribute to lexical matches.
  fts tsvector generated always as (
    to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))
  ) stored
);

-- HNSW for high-recall semantic search at this scale (no probe tuning needed).
create index if not exists chunks_embedding_idx
  on chunks using hnsw (embedding vector_cosine_ops);

-- GIN for full-text search.
create index if not exists chunks_fts_idx
  on chunks using gin (fts);

create index if not exists chunks_doc_id_idx on chunks (doc_id);

-- Read-only public access (anon key) via RLS.
alter table chunks enable row level security;
drop policy if exists "public read" on chunks;
create policy "public read" on chunks for select using (true);

-- ── hybrid search: semantic + keyword fused with Reciprocal Rank Fusion ──────
-- RRF score for a doc = Σ 1/(k + rank_in_list). Rank-based fusion avoids having
-- to normalize cosine distance against ts_rank (different, incomparable scales).
create or replace function match_hybrid(
  query_embedding vector(384),
  query_text      text,
  match_count     int   default 6,
  rrf_k           int   default 60,  -- RRF dampening constant
  pool            int   default 40   -- candidates pulled from each retriever
)
returns table (
  id            text,
  doc_id        text,
  title         text,
  date          date,
  url           text,
  est_seconds   int,
  content       text,
  semantic_rank int,
  keyword_rank  int,
  score         float
)
language sql stable
as $$
  with semantic as (
    select id, row_number() over (order by embedding <=> query_embedding) as rank
    from chunks
    order by embedding <=> query_embedding
    limit pool
  ),
  keyword as (
    select id,
           row_number() over (
             order by ts_rank_cd(fts, websearch_to_tsquery('english', query_text)) desc
           ) as rank
    from chunks
    where fts @@ websearch_to_tsquery('english', query_text)
    order by ts_rank_cd(fts, websearch_to_tsquery('english', query_text)) desc
    limit pool
  ),
  fused as (
    select
      coalesce(s.id, k.id) as id,
      s.rank as semantic_rank,
      k.rank as keyword_rank,
      coalesce(1.0 / (rrf_k + s.rank), 0.0)
        + coalesce(1.0 / (rrf_k + k.rank), 0.0) as score
    from semantic s
    full outer join keyword k on s.id = k.id
  )
  select c.id, c.doc_id, c.title, c.date, c.url, c.est_seconds, c.content,
         f.semantic_rank, f.keyword_rank, f.score
  from fused f
  join chunks c on c.id = f.id
  order by f.score desc
  limit match_count;
$$;

grant execute on function match_hybrid to anon, authenticated;
