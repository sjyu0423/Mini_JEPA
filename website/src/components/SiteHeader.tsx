"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { siteConfig } from "@/lib/site";

const links = [
  { href: "/", label: "Overview" },
  { href: "/architecture/", label: "Architecture" },
  { href: "/experiments/", label: "Experiments" },
  { href: "/engineering/", label: "Engineering" },
];

export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-[color:var(--line)]/80 bg-[color:var(--bg-elevated)]/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <Link href="/" className="focus-ring display text-lg font-semibold tracking-tight">
          Mini-JEPA
        </Link>
        <nav aria-label="Primary" className="flex flex-wrap items-center justify-end gap-1 sm:gap-2">
          {links.map((link) => {
            const normalized = link.href.replace(/\/$/, "") || "/";
            const active =
              normalized === "/"
                ? pathname === "/"
                : pathname === normalized || pathname.startsWith(`${normalized}/`);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`focus-ring rounded-full px-3 py-1.5 text-sm transition ${
                  active
                    ? "bg-[color:var(--accent-soft)] text-[color:var(--accent)]"
                    : "text-[color:var(--ink-muted)] hover:text-[color:var(--ink)]"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
          <a
            href={siteConfig.githubUrl}
            className="focus-ring rounded-full border border-[color:var(--line)] px-3 py-1.5 text-sm text-[color:var(--ink-muted)] hover:text-[color:var(--ink)]"
            rel="noreferrer"
            target="_blank"
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
