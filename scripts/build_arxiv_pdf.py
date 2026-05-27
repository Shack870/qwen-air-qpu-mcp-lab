from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
ARXIV_DIR = PAPER_DIR / "arxiv"
SOURCE = PAPER_DIR / "quantum_enhanced_legacy_moe_inference.md"
HTML_OUT = ARXIV_DIR / "quantum_enhanced_legacy_moe_inference_arxiv.html"
PDF_OUT = ARXIV_DIR / "quantum_enhanced_legacy_moe_inference_arxiv.pdf"


CSS = """
@page {
  size: Letter;
  margin: 0.85in 0.82in 0.9in 0.82in;
}
html {
  background: #ffffff;
}
body {
  max-width: 7.2in;
  margin: 0 auto;
  color: #111111;
  background: #ffffff;
  font-family: "Times New Roman", Times, serif;
  font-size: 10.6pt;
  line-height: 1.31;
  hyphens: auto;
}
h1, h2, h3, h4 {
  color: #111111;
  font-family: "Times New Roman", Times, serif;
  page-break-after: avoid;
  break-after: avoid;
}
h1 {
  font-size: 18pt;
  font-weight: 700;
  line-height: 1.14;
  text-align: center;
  margin: 0 0 0.55em 0;
}
h1 + p {
  text-align: center;
  margin: 0.12em 0;
  font-size: 10.8pt;
}
h1 + p + p {
  text-align: center;
  margin: 0.12em 0 1.15em 0;
  font-size: 10.2pt;
}
h1 + p + p + p {
  text-align: center;
  margin: 0.12em 0 1.2em 0;
  font-size: 9.5pt;
}
h2 {
  font-size: 12pt;
  font-weight: 700;
  margin: 1.05em 0 0.36em 0;
}
h2#abstract {
  text-align: center;
  font-size: 11pt;
  margin-top: 0.65em;
  margin-bottom: 0.25em;
}
h2#abstract + p,
h2#abstract + p + p,
h2#abstract + p + p + p {
  font-size: 9.8pt;
  line-height: 1.24;
  margin-left: 0.38in;
  margin-right: 0.38in;
}
h3 {
  font-size: 11pt;
  font-weight: 700;
  margin: 0.85em 0 0.25em 0;
}
h4 {
  font-size: 10.6pt;
  font-weight: 700;
  font-style: italic;
  margin: 0.7em 0 0.2em 0;
}
p {
  margin: 0.38em 0;
}
p, li {
  orphans: 3;
  widows: 3;
}
ol, ul {
  margin-top: 0.25em;
  margin-bottom: 0.45em;
  padding-left: 1.4em;
}
li {
  margin: 0.16em 0;
}
a {
  color: #111111;
  text-decoration: underline;
}
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #f7f7f7;
  border: 0.4pt solid #d8d8d8;
  border-radius: 0;
  padding: 6px 7px;
  font-size: 8pt;
  line-height: 1.18;
  page-break-inside: avoid;
  break-inside: avoid;
}
code {
  font-family: "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.86em;
  background: #f7f7f7;
  padding: 0.04em 0.18em;
}
pre code {
  background: transparent;
  padding: 0;
}
blockquote {
  margin: 0.65em 0 0.65em 0.18in;
  padding-left: 0.14in;
  border-left: 1.5pt solid #777777;
  color: #111111;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.7em 0;
  font-size: 9pt;
  page-break-inside: avoid;
  break-inside: avoid;
}
th, td {
  border: 0.4pt solid #c8c8c8;
  padding: 4px 5px;
  vertical-align: top;
}
th {
  background: #f1f1f1;
  font-weight: 700;
}
img {
  max-width: 95%;
  height: auto;
  display: block;
  margin: 0.55em auto;
  page-break-inside: avoid;
  break-inside: avoid;
}
hr {
  border: 0;
  border-top: 0.4pt solid #c8c8c8;
  margin: 0.8em 0;
}
"""


def prepare_markdown(text: str) -> str:
    text = text.replace("Draft date: 2026-05-26", "Preprint version: May 27, 2026")
    text = re.sub(r"!\[(.*?)\]\((figures/.*?)\)", r"![\1](../\2)", text)
    return text


def render_html() -> str:
    text = prepare_markdown(SOURCE.read_text())
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
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  {body}
</body>
</html>
"""


def print_pdf_with_chrome() -> None:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome.exists():
        raise SystemExit("Google Chrome not found; HTML paper was still generated.")
    cmd = [
        str(chrome),
        "--headless",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={PDF_OUT}",
        HTML_OUT.as_uri(),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ARXIV_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(render_html())
    print(f"wrote {HTML_OUT}")
    print_pdf_with_chrome()
    print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
