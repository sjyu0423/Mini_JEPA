import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
PUBLIC_MARKDOWN = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def test_relative_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
    missing: list[str] = []
    for document in PUBLIC_MARKDOWN:
        for target in link_pattern.findall(document.read_text()):
            if target.startswith(("http://", "https://", "#")):
                continue
            path = (document.parent / target.split("#", maxsplit=1)[0]).resolve()
            if not path.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "Broken documentation links:\n" + "\n".join(missing)


def test_public_docs_do_not_contain_private_machine_paths() -> None:
    forbidden = ("/Users/", "/Volumes/Projects/")
    violations = [
        str(document.relative_to(ROOT))
        for document in PUBLIC_MARKDOWN
        if any(item in document.read_text() for item in forbidden)
    ]
    assert not violations


def test_release_documents_and_screenshot_exist() -> None:
    required = (
        ROOT / "LICENSE",
        ROOT / "CITATION.cff",
        ROOT / "docs/research-overview.md",
        ROOT / "docs/portfolio-summary.md",
        ROOT / "docs/artifact-provenance.md",
        ROOT / "docs/model-card.md",
        ROOT / "docs/milestones/m5.md",
        ROOT / "docs/assets/mini-jepa-demo.png",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)


def test_generated_and_large_directories_are_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    for expected in (".venv/", "data/", "artifacts/", "dist/"):
        assert expected in ignored
