import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // @huggingface/transformers ships native/onnx bits that should not be bundled
  // by the server build — keep them external so the runtime loads them directly.
  serverExternalPackages: ["@huggingface/transformers"],
};

export default nextConfig;
