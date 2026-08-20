import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    // Two lockfiles live above this directory; pin the root so Turbopack does
    // not guess the workspace and warn on every build.
    root: __dirname,
  },
};

export default nextConfig;
