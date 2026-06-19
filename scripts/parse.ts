/**
 * Stage 1 — parse.
 * Reads the local transcript corpus (huberman_transcripts/*.txt), pulls date +
 * title out of each filename, and fuzzy-matches each episode to a YouTube video
 * ID (from the cached channel listing) so citations can deep-link to the video.
 *
 * Output: data/documents.json  [{ id, date, title, videoId, url, wordCount, text }]
 */
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const TRANSCRIPT_DIR = "huberman_transcripts";
const CHANNEL_FILE = "data/channel.txt";
const OUT_FILE = "data/documents.json";

// Boilerplate tokens that carry no matching signal between filename + yt title.
const STOP = new Set([
  "huberman", "lab", "podcast", "essentials", "essential", "clip", "quantal",
  "the", "a", "an", "of", "for", "to", "and", "with", "your", "you", "how",
  "in", "on", "is", "ep", "episode", "part", "dr", "&",
]);

function tokenize(s: string): Set<string> {
  return new Set(
    s
      .toLowerCase()
      .replace(/&/g, " and ")
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 1 && !STOP.has(t))
  );
}

/** Overlap coefficient: |A∩B| / min(|A|,|B|). Robust to subset titles. */
function overlap(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let inter = 0;
  for (const t of a) if (b.has(t)) inter++;
  return inter / Math.min(a.size, b.size);
}

type Channel = { videoId: string; title: string; tokens: Set<string> };

function loadChannel(): Channel[] {
  const raw = readFileSync(CHANNEL_FILE, "utf8").trim().split("\n");
  return raw.map((line) => {
    const idx = line.indexOf("|||");
    const videoId = line.slice(0, idx);
    const title = line.slice(idx + 3);
    return { videoId, title, tokens: tokenize(title) };
  });
}

function parseFilename(file: string): { date: string; title: string } {
  const base = file.replace(/\.txt$/, "");
  const m = base.match(/^(\d{4}-\d{2}-\d{2})_(.*)$/);
  if (!m) return { date: "", title: base.replace(/_/g, " ") };
  return { date: m[1], title: m[2].replace(/_/g, " ") };
}

function slugify(file: string): string {
  return file.replace(/\.txt$/, "").toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

const channel = loadChannel();
const files = readdirSync(TRANSCRIPT_DIR)
  // Only dated episode transcripts (YYYY-MM-DD_*.txt); skips scraper artifacts.
  .filter((f) => /^\d{4}-\d{2}-\d{2}_.*\.txt$/.test(f))
  .sort();

let matched = 0;
const docs = files.map((file) => {
  const { date, title } = parseFilename(file);
  const text = readFileSync(join(TRANSCRIPT_DIR, file), "utf8").trim();
  const wordCount = text.split(/\s+/).length;

  // Best video match by title-token overlap.
  const qTokens = tokenize(title);
  let best: Channel | null = null;
  let bestScore = 0;
  for (const c of channel) {
    const s = overlap(qTokens, c.tokens);
    if (s > bestScore) {
      bestScore = s;
      best = c;
    }
  }
  const isMatch = best !== null && bestScore >= 0.6;
  if (isMatch) matched++;

  const videoId = isMatch ? best!.videoId : null;
  const url = videoId
    ? `https://www.youtube.com/watch?v=${videoId}`
    : `https://www.youtube.com/@hubermanlab/search?query=${encodeURIComponent(title)}`;

  return {
    id: slugify(file),
    date,
    title,
    videoId,
    matchScore: Number(bestScore.toFixed(2)),
    url,
    wordCount,
    text,
  };
});

writeFileSync(OUT_FILE, JSON.stringify(docs, null, 2));

const totalWords = docs.reduce((s, d) => s + d.wordCount, 0);
console.log(`parsed   ${docs.length} transcripts`);
console.log(`matched  ${matched}/${docs.length} to a YouTube video id (>=0.6 overlap)`);
console.log(`unmatched (search-link fallback): ${docs.length - matched}`);
console.log(`total words: ${totalWords.toLocaleString()}`);
console.log(`wrote ${OUT_FILE}`);
