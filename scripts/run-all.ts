/**
 * Runs the full ingestion pipeline in order. Each stage is resumable/idempotent,
 * so re-running after an interruption picks up where it left off.
 *   parse -> chunk -> context (LLM) -> embed -> upload
 */
import { execSync } from "node:child_process";

const stages = [
  ["parse", "scripts/parse.ts"],
  ["chunk", "scripts/chunk.ts"],
  ["context", "scripts/context.ts"],
  ["embed", "scripts/embed.ts"],
  ["upload", "scripts/upload.ts"],
];

for (const [name, file] of stages) {
  console.log(`\n━━━ ${name} ━━━`);
  execSync(`tsx ${file}`, { stdio: "inherit" });
}
console.log("\n✅ pipeline complete");
