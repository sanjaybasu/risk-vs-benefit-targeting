"""Build Word files of the manuscript and SI Appendix for co-author tracked changes.

Embeds Figures 1 to 3 above their legends, starts Tables, Figures, and References on new pages, and turns on
continuous line numbering. Output: notebooks/system-three/paper/*.docx
Usage: python3 scripts/47_build_docx.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parents[3] / "notebooks" / "system-three" / "paper"
BREAK = '\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'
MARGINS = 'w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"'
LN = '<w:lnNumType w:countBy="1" w:distance="360" w:restart="continuous"/>'


def section_end(landscape: bool) -> str:
    """Paragraph that ends a section; the section it closes is portrait or landscape."""
    size = '<w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/>' if landscape else '<w:pgSz w:w="12240" w:h="15840"/>'
    return (f'\n```{{=openxml}}\n<w:p><w:pPr><w:sectPr><w:type w:val="nextPage"/>{size}<w:pgMar {MARGINS}/>{LN}'
            '</w:sectPr></w:pPr></w:p>\n```\n')


BORDERS = ('<w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
           '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
           '<w:insideH w:val="single" w:sz="2" w:space="0" w:color="BFBFBF"/></w:tblBorders>')


def tables(xml: str) -> str:
    """Autofit columns to content, add horizontal rules, and set table text to 9 pt."""
    def one(mt):
        t = mt.group(0)
        t = t.replace('<w:tblLayout w:type="fixed" />', BORDERS + '<w:tblLayout w:type="autofit" />', 1)
        t = t.replace("<w:r><w:t", '<w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:t')
        t = re.sub(r"<w:r><w:rPr>((?:(?!</w:rPr>).)*?)</w:rPr>",
                   lambda m: m.group(0) if ("w:sz " in m.group(1) or "vertAlign" in m.group(1))
                   else f'<w:r><w:rPr>{m.group(1)}<w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>', t)
        return t
    return re.sub(r"<w:tbl>.*?</w:tbl>", one, xml, flags=re.S)


def prep(md: str) -> str:
    md = md.replace("\n## Tables\n", f"\n{section_end(False)}\n## Tables\n", 1)
    md = md.replace("\n## Figure legends\n", f"\n{section_end(True)}\n## Figure legends\n", 1)
    md = md.replace("\n## References\n", f"\n{BREAK}\n## References\n", 1)
    md = md.replace("## Figure legends", "## Figures")
    for i in (1, 2, 3):
        md = re.sub(rf"\n\*\*Figure {i}\.\*\*",
                    f"\n![](figures/figure{i}.png){{width=6.5in}}\n\n**Figure {i}.**", md, count=1)
    return md


def line_numbers(docx: Path) -> None:
    tmp = docx.with_suffix(".tmp")
    with zipfile.ZipFile(docx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode()
                sect = ('<w:sectPr><w:footnotePr><w:numRestart w:val="eachSect"/></w:footnotePr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" '
                        'w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
                        '' + LN + '</w:sectPr>')
                xml, n = re.subn(r"<w:sectPr>.*?</w:sectPr>", sect, xml, count=1, flags=re.S)
                assert n == 1
                xml = tables(xml)
                xml = xml.replace('<m:nor /><m:sty m:val="p" />', "<m:nor />")  # schema: nor and sty are alternatives
                data = xml.encode()
            if item.filename == "[Content_Types].xml" and b'Extension="png"' not in data:
                data = data.replace(b'<Default Extension="xml"', b'<Default Extension="png" ContentType="image/png" /><Default Extension="xml"', 1)
            if not item.filename.endswith("/"):
                zout.writestr(item, data)
    shutil.move(tmp, docx)


def build(src: str, out: str, embed: bool) -> None:
    md = (PAPER / src).read_text()
    if embed:
        md = prep(md)
    with tempfile.NamedTemporaryFile("w", suffix=".md", dir=PAPER, delete=False) as fh:
        fh.write(md)
    try:
        subprocess.run(["pandoc", fh.name, "-f", "markdown+superscript", "-o", str(PAPER / out),
                        "--resource-path", str(PAPER)], check=True)
    finally:
        Path(fh.name).unlink()
    line_numbers(PAPER / out)
    print("wrote", PAPER / out)


if __name__ == "__main__":
    build("risk_vs_benefit_targeting_22_trials_pnas_manuscript.md", "risk_vs_benefit_targeting_22_trials_manuscript.docx", True)
    build("risk_vs_benefit_targeting_22_trials_si_appendix.md", "risk_vs_benefit_targeting_22_trials_si_appendix.docx", False)
