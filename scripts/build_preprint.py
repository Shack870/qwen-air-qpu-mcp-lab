from __future__ import annotations

import html
import subprocess
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
SOURCE = PAPER_DIR / "quantum_enhanced_legacy_moe_inference.md"
HTML_OUT = PAPER_DIR / "quantum_enhanced_legacy_moe_inference.html"
PDF_OUT = PAPER_DIR / "quantum_enhanced_legacy_moe_inference.pdf"


CSS = """
@page {
  size: Letter;
  margin: 0.7in;
}
body {
  max-width: 8.1in;
  margin: 0 auto;
  color: #111827;
  background: #ffffff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  font-size: 10.7pt;
  line-height: 1.48;
}
h1, h2, h3, h4 {
  color: #0f172a;
  line-height: 1.18;
  page-break-after: avoid;
}
h1 {
  font-size: 23pt;
  margin: 0 0 0.4em;
}
h2 {
  font-size: 16pt;
  border-bottom: 1px solid #d8dee9;
  padding-bottom: 0.18em;
  margin-top: 1.35em;
}
h3 {
  font-size: 12.7pt;
  margin-top: 1.15em;
}
p, li {
  orphans: 3;
  widows: 3;
}
a {
  color: #1d4ed8;
  text-decoration: none;
}
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #f6f8fa;
  border: 1px solid #d8dee9;
  border-radius: 5px;
  padding: 9px;
  font-size: 8.8pt;
}
code {
  font-family: "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 0.92em;
  background: #f6f8fa;
  padding: 0.08em 0.22em;
  border-radius: 3px;
}
pre code {
  padding: 0;
  background: transparent;
}
blockquote {
  border-left: 4px solid #93c5fd;
  margin: 1em 0;
  padding: 0.2em 0 0.2em 1em;
  color: #334155;
  background: #f8fbff;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1em 0;
  font-size: 9.3pt;
}
th, td {
  border: 1px solid #d8dee9;
  padding: 6px 7px;
  vertical-align: top;
}
th {
  background: #eef2f7;
}
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.6em auto;
  border: 1px solid #e5e7eb;
}
.meta {
  color: #475569;
  font-size: 10pt;
  margin-bottom: 1.4em;
}
.notice {
  border: 1px solid #f1c86a;
  background: #fff9e8;
  padding: 0.75em 0.9em;
  border-radius: 5px;
  margin: 1em 0 1.3em;
}
"""


def render_html() -> str:
    text = SOURCE.read_text()
    body = markdown.markdown(
        text,
        extensions=["extra", "sane_lists", "toc"],
        output_format="html5",
    )
    title = "Quantum-Enhanced Hyperparameter Tuning for High-Performance On-Device CPU-Only Inference of Mixture-of-Experts LLMs on Legacy Hardware"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="notice">
    Preprint draft generated from repository Markdown. The GitHub and Hugging Face
    artifacts are the source of truth for code, data, and reproduction commands.
  </div>
  {body}
</body>
</html>
"""


def print_pdf_with_chrome() -> None:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome.exists():
        raise SystemExit("Google Chrome not found; HTML preprint was still generated.")
    cmd = [
        str(chrome),
        "--headless",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--print-to-pdf={PDF_OUT}",
        HTML_OUT.as_uri(),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    HTML_OUT.write_text(render_html())
    print(f"wrote {HTML_OUT}")
    print_pdf_with_chrome()
    print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
