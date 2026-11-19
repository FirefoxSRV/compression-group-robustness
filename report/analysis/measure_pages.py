"""Measure the report two ways: total pages, and body-text pages.

The brief caps body text at 4.5 pages "excluding figures, tables and ref",
so the body measure compiles a variant with every float, the bibliography
and the contribution section stripped out, and counts what is left.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]
WORDS_PER_FULL_PAGE = 560


def pages(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    return int(re.search(r"^Pages:\s+(\d+)", out, re.M).group(1))


def words(pdf: Path, page: int | None = None) -> int:
    cmd = ["pdftotext"]
    if page:
        cmd += ["-f", str(page), "-l", str(page)]
    cmd += [str(pdf), "-"]
    return len(subprocess.run(cmd, capture_output=True, text=True).stdout.split())


def main() -> None:
    full = REPORT / "report.pdf"
    n = pages(full)
    print(f"TOTAL document: {n} pages")
    for p in range(1, n + 1):
        print(f"   page {p}: {words(full, p):4d} words")

    src = (REPORT / "report.tex").read_text()
    body = re.sub(r"\\begin\{table\}.*?\\end\{table\}\n", "", src, flags=re.S)
    body = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}\n", "", body, flags=re.S)
    body = re.sub(r"\\setlength\{\\bibsep\}.*\n", "", body)
    body = re.sub(r"\{\\(?:small|footnotesize)\\bibliography\{refs\}\}", "", body)
    body = body.replace("\\section*{Contribution of each member}", "")
    assert "\\begin{table}" not in body, "a float survived stripping"
    assert "\\bibliography{" not in body, "the bibliography survived stripping"

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "body.tex").write_text(body)
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "body.tex"],
                           cwd=d, capture_output=True)
        pdf = d / "body.pdf"
        m = pages(pdf)
        last = words(pdf, m)
        total_words = words(pdf)
        frac = (m - 1) + last / WORDS_PER_FULL_PAGE
        print(f"\nBODY text only: {frac:.2f} pages "
              f"({m - 1} full + {last} words), {total_words} words total")

    ok_total = n <= 6
    ok_body = frac <= 4.5
    print(f"\n  total <= 6 pages : {'PASS' if ok_total else f'FAIL ({n})'}")
    print(f"  body  <= 4.5 pages: {'PASS' if ok_body else f'FAIL ({frac:.2f})'}")
    sys.exit(0 if (ok_total and ok_body) else 1)


if __name__ == "__main__":
    main()
