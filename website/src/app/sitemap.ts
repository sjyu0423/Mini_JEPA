import type { MetadataRoute } from "next";
import { siteConfig, siteOrigin, withBasePath } from "@/lib/site";

export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  const origin = siteOrigin() ?? new URL(siteConfig.publishedSiteUrl).origin;
  const routes = ["/", "/architecture/", "/experiments/", "/engineering/"];
  return routes.map((route) => ({
    url: `${origin}${withBasePath(route)}`,
    lastModified: new Date("2026-11-18"),
    changeFrequency: "monthly" as const,
    priority: route === "/" ? 1 : 0.7,
  }));
}
