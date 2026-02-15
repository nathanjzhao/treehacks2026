import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  turbopack: {},
  webpack: (config) => {
    // ONNX Runtime Web needs .wasm files served as assets
    config.module.rules.push({
      test: /\.wasm$/,
      type: "asset/resource",
    });
    return config;
  },
  // Allow WASM files from node_modules to be served
  serverExternalPackages: ["onnxruntime-web"],
};

export default nextConfig;
