import type { NextConfig } from "next";

/**
 * GitHub Pages often hosts a project site under /<repository-name>.
 * Set NEXT_PUBLIC_BASE_PATH (e.g. "/Mini_JEPA") at build time.
 * Leave empty for local preview at http://localhost:3000/.
 */
const rawBasePath = process.env.NEXT_PUBLIC_BASE_PATH?.trim() ?? "";
const basePath =
  rawBasePath === "/" || rawBasePath === ""
    ? ""
    : `/${rawBasePath.replace(/^\/+|\/+$/g, "")}`;

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  basePath,
  assetPrefix: basePath ? `${basePath}/` : undefined,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },
};

export default nextConfig;
