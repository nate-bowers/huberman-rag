import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Keep transformers.js out of the webpack bundle (it loads native/wasm at runtime).
  serverExternalPackages: ["@huggingface/transformers"],
  // onnxruntime-node ships ~400MB of native binaries for every platform and blows
  // past Vercel's 250MB function limit. The serverless query path uses the WASM
  // backend instead (see lib/embeddings.ts), so exclude the native package.
  outputFileTracingExcludes: {
    "*": [
      "node_modules/.pnpm/onnxruntime-node*/**",
      "node_modules/onnxruntime-node/**",
      "node_modules/@img/sharp-libvips*/**",
    ],
  },
};

export default nextConfig;
