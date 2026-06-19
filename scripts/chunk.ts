/**
 * Stage 2 — chunk.
 * Splits each document into overlapping, sentence-aware chunks and attaches the
 * metadata each chunk needs downstream: stable id, source episode, and an
 * estimated video timestamp derived from word position.
 *
 * Input:  data/documents.json
 * Output: data/chunks.json
 */
import { readFileSync, writeFileSync } from "node:fs";
import { chunkText } from "../lib/chunk.ts";

const WPM = 150; // Huberman's approx speaking rate, for timestamp estimation

type Doc = {
  id: string;
  date: string;
  title: string;
  videoId: string | null;
  url: string;
  wordCount: number;
  text: string;
};

const docs: Doc[] = JSON.parse(readFileSync("data/documents.json", "utf8"));

const chunks = docs.flatMap((doc) => {
  const parts = chunkText(doc.text);
  return parts.map((c, i) => {
    const estSeconds = Math.round((c.wordOffset / WPM) * 60);
    const tsUrl = doc.videoId
      ? `https://www.youtube.com/watch?v=${doc.videoId}&t=${estSeconds}s`
      : doc.url;
    return {
      id: `${doc.id}__${i}`,
      docId: doc.id,
      chunkIndex: i,
      date: doc.date,
      title: doc.title,
      videoId: doc.videoId,
      url: tsUrl,
      wordOffset: c.wordOffset,
      wordCount: c.wordCount,
      estSeconds,
      text: c.text,
    };
  });
});

writeFileSync("data/chunks.json", JSON.stringify(chunks, null, 2));

// --- stats / sanity checks ---
const wordCounts = chunks.map((c) => c.wordCount);
const avg = Math.round(wordCounts.reduce((a, b) => a + b, 0) / chunks.length);
const max = Math.max(...wordCounts);
const estTokensMax = Math.round(max * 1.33);
const overLimit = wordCounts.filter((w) => w * 1.33 > 460).length; // header+body budget

console.log(`chunks:        ${chunks.length.toLocaleString()}`);
console.log(`avg words:     ${avg}  (~${Math.round(avg * 1.33)} tokens body)`);
console.log(`max words:     ${max}  (~${estTokensMax} tokens body)`);
console.log(`chunks whose body alone may exceed ~460 tokens: ${overLimit}`);
console.log(`wrote data/chunks.json`);
