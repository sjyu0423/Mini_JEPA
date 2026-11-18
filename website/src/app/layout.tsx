import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Source_Serif_4 } from "next/font/google";
import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { canonicalSiteRoot, siteConfig, siteOrigin, withBasePath } from "@/lib/site";
import "./globals.css";

const sans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-ibm-plex-sans",
  display: "swap",
});

const serif = Source_Serif_4({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-source-serif",
  display: "swap",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-ibm-plex-mono",
  display: "swap",
});

/** Host origin only — basePath is applied via withBasePath for asset URLs. */
const metadataOrigin =
  siteOrigin() ?? new URL(siteConfig.publishedSiteUrl).origin;
const canonicalRoot = canonicalSiteRoot();

export const metadata: Metadata = {
  metadataBase: new URL(`${metadataOrigin}/`),
  title: {
    default: `${siteConfig.title} — ${siteConfig.subtitle}`,
    template: `%s · ${siteConfig.title}`,
  },
  description: siteConfig.description,
  authors: [{ name: siteConfig.author, url: siteConfig.githubUrl }],
  creator: siteConfig.author,
  alternates: {
    canonical: `${canonicalRoot}/`,
  },
  openGraph: {
    title: siteConfig.title,
    description: siteConfig.description,
    type: "website",
    url: `${canonicalRoot}/`,
    siteName: siteConfig.title,
    images: [{ url: withBasePath("/plots/mini-jepa-demo.png"), alt: "Mini-JEPA interactive demo" }],
  },
  twitter: {
    card: "summary_large_image",
    title: siteConfig.title,
    description: siteConfig.description,
    images: [withBasePath("/plots/mini-jepa-demo.png")],
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} ${serif.variable} ${mono.variable}`}>
      <body
        className="antialiased"
        style={
          {
            "--font-body": "var(--font-ibm-plex-sans), 'Segoe UI', sans-serif",
            "--font-display": "var(--font-source-serif), 'Iowan Old Style', serif",
            "--font-mono": "var(--font-ibm-plex-mono), ui-monospace, monospace",
          } as React.CSSProperties
        }
      >
        <SiteHeader />
        <main>{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
