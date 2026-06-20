import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Keep transformers.js external (not webpack-bundled). NOTE: do NOT list
  // onnxruntime-node here — serverExternalPackages copies the whole package and
  // ignores outputFileTracingExcludes, which is what blew the 250MB limit.
  serverExternalPackages: ["@huggingface/transformers"],
  // onnxruntime-node ships native binaries for 6 platforms (~165MB of waste).
  // Vercel runs Linux x64, so drop the other five to fit the 250MB function limit
  // while keeping the native backend (more reliable than WASM in the function).
  outputFileTracingExcludes: {
    "*": [
      "**/onnxruntime-node/bin/napi-v3/darwin/**",
      "**/onnxruntime-node/bin/napi-v3/win32/**",
      "**/onnxruntime-node/bin/napi-v3/linux/arm64/**",
    ],
  },
};

export default nextConfig;
