import { siteConfig } from "@/lib/site";

export function SiteFooter() {
  return (
    <footer className="mt-20 border-t border-[color:var(--line)]">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-10 text-sm text-[color:var(--ink-muted)] sm:px-6 sm:flex-row sm:items-center sm:justify-between">
        <p>
          {siteConfig.title} — self-supervised motion research on a synthetic bouncing-ball
          domain. Position RMSE is in pixels; lower is better.
        </p>
        <div className="flex flex-col gap-1 sm:items-end">
          <a
            className="focus-ring underline-offset-2 hover:underline"
            href={siteConfig.githubUrl}
            rel="noreferrer"
            target="_blank"
          >
            GitHub repository
          </a>
          <p className="mono text-xs">Static research site · no model inference in the browser</p>
        </div>
      </div>
    </footer>
  );
}
