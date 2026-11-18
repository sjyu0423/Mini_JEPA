import type { MetadataRoute } from "next";
import { siteOrigin, withBasePath } from "@/lib/site";

export const dynamic = "force-static";

export default function robots(): MetadataRoute.Robots {
  const origin = siteOrigin();
  return {
    rules: {
      userAgent: "*",
      allow: "/",
    },
    ...(origin
      ? { sitemap: `${origin}${withBasePath("/sitemap.xml")}` }
      : {}),
  };
}
