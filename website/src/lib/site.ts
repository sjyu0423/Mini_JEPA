export const siteConfig = {
  title: "Mini-JEPA",
  subtitle: "Learning Motion Through Self-Supervised Prediction",
  description:
    "A compact JEPA research project on whether latent prediction can learn useful motion representations from bouncing-ball video.",
  author: "sjyu0423",
  githubUrl: "https://github.com/sjyu0423/Mini_JEPA",
  /** Canonical published research site (project Pages URL). */
  publishedSiteUrl: "https://sjyu0423.github.io/Mini_JEPA",
};

/** Prefix a public asset or client fetch path with the configured Pages base path. */
export function withBasePath(path: string): string {
  const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
  if (!path.startsWith("/")) {
    return `${base}/${path}`;
  }
  return `${base}${path}`;
}

/**
 * Origin used for absolute sitemap/robots URLs.
 * Production workflow sets NEXT_PUBLIC_SITE_URL to https://<owner>.github.io
 * and NEXT_PUBLIC_BASE_PATH to /<repo>, which compose the canonical Pages URL.
 */
export function siteOrigin(): string | null {
  const configured = process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/$/, "");
  return configured || null;
}

/** Full canonical site root including optional base path. */
export function canonicalSiteRoot(): string {
  const origin = siteOrigin();
  const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
  if (origin) {
    return `${origin}${base}`.replace(/\/$/, "") || origin;
  }
  return siteConfig.publishedSiteUrl;
}
