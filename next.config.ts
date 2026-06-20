import type { NextConfig } from "next";

// The live app embeds queries via the HF Inference API (lib/inference.ts), so the
// serverless function carries no ML runtime — no special bundling needed.
// Transformers.js is used only by the local ingest scripts.
const nextConfig: NextConfig = {};

export default nextConfig;
