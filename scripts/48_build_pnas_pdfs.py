"""Build publication-ready PNAS Research Report PDF (official pnas-new.cls) and single combined SI Appendix PDF.

Reads notebooks/system-three/paper/risk_vs_benefit_targeting_22_trials_pnas_manuscript.md and
notebooks/system-three/paper/risk_vs_benefit_targeting_22_trials_si_appendix.md.
Outputs:
  notebooks/system-three/paper/risk_vs_benefit_targeting_22_trials_pnas.pdf
  notebooks/system-three/paper/risk_vs_benefit_targeting_22_trials_si_appendix.pdf
Usage: python3 scripts/48_build_pnas_pdfs.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1] / "notebooks" / "system-three" / "paper"
LATEX = PAPER / "pnas_latex"


def tex_escape(s: str) -> str:
    """Convert markdown inline syntax (math, italics, bold, superscripts, unicode) to safe pdflatex."""
    # Preserve $...$ and $$...$$ math blocks
    parts = re.split(r"(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)", s)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            # Inside math mode: fix unicode minus/symbols
            m = p.replace("−", "-").replace("–", "-").replace("×", r"\times ")
            m = m.replace(r"\operatorname{logit}", r"\mathrm{logit}\,")
            out.append(m)
            continue
        # Handle markdown superscripts ^a^, ^b^
        t = p
        # Temporarily protect italics *...* and bold **...** and URLs
        tokens: list[tuple[str, str]] = []

        def stash(val: str) -> str:
            key = f"ZZTOK{len(tokens)}ZZ"
            tokens.append((key, val))
            return key

        t = re.sub(r"https?://[^\s),;]+", lambda m: stash(r"\url{" + m.group(0) + "}"), t)
        t = re.sub(r"\*\*([^*]+)\*\*", lambda m: stash(r"\textbf{" + escape_plain(m.group(1)) + "}"), t)
        t = re.sub(r"\*([^*]+)\*", lambda m: stash(r"\textit{" + escape_plain(m.group(1)) + "}"), t)
        t = re.sub(r"\^([^^]+)\^", lambda m: stash(r"\textsuperscript{" + escape_plain(m.group(1)) + "}"), t)
        t = escape_plain(t)
        for key, val in tokens:
            t = t.replace(key, val)
        out.append(t)
    return "".join(out)


def escape_plain(t: str) -> str:
    repl = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("−", r"$-$"),
        ("–", "--"),
        ("—", "---"),
        ("²", r"\textsuperscript{2}"),
        ("≤", r"$\le$"),
        ("≥", r"$\ge$"),
        ("×", r"$\times$"),
        ("→", r"$\to$"),
        ("é", r"\'{e}"),
        ("ö", r'\"o'),
        ("ü", r'\"u'),
        ("ł", r"\l{}"),
        ("ı", r"\i{}"),
        ("Ş", r"\c{S}"),
        ("’", "'"),
        ("‘", "`"),
        ("“", "``"),
        ("”", "''"),
    ]
    t = re.sub(r'"([^"]+)"', r"``\1''", t)
    for a, b in repl:
        t = t.replace(a, b)
    return t


def parse_md_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    rows = []
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if all(re.fullmatch(r"[-:\s]+", c) for c in cells):
            continue
        rows.append(cells)
    return rows[0], rows[1:]


def render_pnas_table(caption: str, panels: list[tuple[str | None, list[str], list[list[str]]]],
                      footnote: str, col_specs: list[str]) -> str:
    out = [r"\begin{table*}[tbhp]", r"\centering", r"\caption{" + tex_escape(caption) + "}"]
    for idx, (subhead, hdr, body) in enumerate(panels):
        if subhead:
            out.append(r"\par\vspace{3pt}\noindent\parbox{\textwidth}{\raggedright\sffamily\footnotesize\textit{" + tex_escape(subhead) + r"}}\par\vspace{2pt}")
        spec = col_specs[idx]
        out.append(r"\begin{tabular}{" + spec + "}")
        out.append(r"\toprule")
        out.append(" & ".join(r"\textbf{" + tex_escape(h) + "}" for h in hdr) + r" \\")
        out.append(r"\midrule")
        for r in body:
            out.append(" & ".join(tex_escape(c) for c in r) + r" \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}\par")
        if idx < len(panels) - 1:
            out.append(r"\vspace{6pt}")
    if footnote:
        out.append(r"\vspace{4pt}\noindent\parbox{\textwidth}{\raggedright\sffamily\fontsize{7}{9}\selectfont " + tex_escape(footnote) + r"}\par")
    out.append(r"\end{table*}")
    return "\n".join(out)


def build_main_pnas() -> Path:
    md = (PAPER / "risk_vs_benefit_targeting_22_trials_pnas_manuscript.md").read_text()
    title = md.splitlines()[0].lstrip("# ").strip()
    sig = md.split("## Significance Statement\n\n")[1].split("\n\n## Abstract")[0].strip()
    abs_ = md.split("## Abstract\n\n")[1].split("\n\n## Introduction")[0].strip()
    intro = md.split("## Introduction\n\n")[1].split("\n\n## Results")[0].strip()
    results = md.split("## Results\n\n")[1].split("\n\n## Discussion")[0].strip()
    disc_raw = md.split("## Discussion\n\n")[1].split("\n\n## Materials and Methods")[0].strip()
    methods_raw = md.split("## Materials and Methods\n\n")[1].split("\n\n## Tables")[0].strip()
    tables_raw = md.split("## Tables\n\n")[1].split("\n\n## Figure legends")[0].strip()
    figs_raw = md.split("## Figure legends\n\n")[1].split("\n\n## References")[0].strip()
    refs_raw = md.split("## References\n\n")[1].strip()

    # Extract Author contributions and Competing interests from Methods
    m_paras = [p.strip() for p in methods_raw.split("\n\n") if p.strip()]
    auth_contrib = ""
    coi = ""
    methods_keep = []
    for p in m_paras:
        if p.startswith("*Author contributions.*"):
            auth_contrib = p.replace("*Author contributions.*", "").strip()
        elif p.startswith("*Competing interests.*"):
            coi = p.replace("*Competing interests.*", "").strip()
        else:
            methods_keep.append(p)

    # Split Discussion around Box 1
    box_title = "Box 1. Proposed workflow for choosing a targeting rule in practice"
    before_box, rest = disc_raw.split("### " + box_title + "\n\n")
    box_block, after_box = rest.split("\n\n", 1)
    _, box_rows = parse_md_table(box_block.splitlines())

    # Format Box 1 as a clean full-width shaded box (figure* float or inline mdframed)
    box_tex_lines = [
        r"\begin{figure*}[tbhp]",
        r"\begin{mdframed}[backgroundcolor=black!4,linecolor=black!75,linewidth=0.7pt,innertopmargin=6pt,innerbottommargin=6pt,innerleftmargin=8pt,innerrightmargin=8pt]",
        r"{\sffamily\small\textbf{" + tex_escape(box_title) + r"}}\vspace{4pt}",
        r"\begin{enumerate}[leftmargin=14pt,itemsep=2pt,topsep=2pt,parsep=0pt]\sffamily\footnotesize",
    ]
    for step, act in box_rows:
        box_tex_lines.append(r"\item " + tex_escape(act))
    box_tex_lines += [
        r"\end{enumerate}",
        r"\end{mdframed}",
        r"\end{figure*}",
    ]
    box_tex = "\n".join(box_tex_lines)

    # Parse Table 1 and Table 2 from tables_raw
    t1_part, t2_part = tables_raw.split("### Table 2.")
    t1_lines = [l for l in t1_part.splitlines() if l.strip()]
    t1_cap = t1_lines[0].replace("### Table 1.", "").strip()
    t1_tbl = [l for l in t1_lines[1:] if l.strip().startswith("|")]
    t1_fn = " ".join(l for l in t1_lines[1:] if not l.strip().startswith("|"))
    t1_hdr, t1_body = parse_md_table(t1_tbl)
    table1_tex = render_pnas_table(
        t1_cap, [(None, t1_hdr, t1_body)], t1_fn,
        [r">{\raggedright\arraybackslash}p{8.2cm} r c r r"]
    )

    t2_lines = [l for l in ("### Table 2." + t2_part).splitlines() if l.strip()]
    t2_cap = t2_lines[0].replace("### Table 2.", "").strip()
    idx_b = next(i for i, l in enumerate(t2_lines) if l.startswith("*B."))
    t2a_sub = t2_lines[1].strip("*")
    t2a_tbl = [l for l in t2_lines[2:idx_b] if l.strip().startswith("|")]
    t2b_sub = t2_lines[idx_b].strip("*")
    t2b_tbl = [l for l in t2_lines[idx_b + 1:] if l.strip().startswith("|")]
    t2_fn = " ".join(l for l in t2_lines[idx_b + 1:] if not l.strip().startswith("|"))
    t2a_hdr, t2a_body = parse_md_table(t2a_tbl)
    t2b_hdr, t2b_body = parse_md_table(t2b_tbl)
    table2_tex = render_pnas_table(
        t2_cap,
        [(t2a_sub, t2a_hdr, t2a_body), (t2b_sub, t2b_hdr, t2b_body)],
        t2_fn,
        [r">{\raggedright\arraybackslash}p{2.3cm} >{\raggedright\arraybackslash}p{2.05cm} >{\raggedright\arraybackslash}p{2.15cm} >{\raggedright\arraybackslash}p{2.0cm} >{\raggedright\arraybackslash}p{2.35cm} >{\raggedright\arraybackslash}p{2.1cm} >{\raggedright\arraybackslash}p{2.0cm}",
         r">{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}p{2.9cm} >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{2.9cm} >{\raggedright\arraybackslash}p{2.8cm}"]
    )

    # Parse Figure legends
    fig_caps = {}
    for p in figs_raw.split("\n\n"):
        m = re.match(r"\*\*Figure (\d)\.\*\*\s*(.*)", p.strip(), re.S)
        if m:
            fig_caps[int(m.group(1))] = m.group(2).strip()

    def fig_tex(num: int) -> str:
        return (
            "\\begin{figure*}[tbhp]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\textwidth]{{../figures/figure{num}.pdf}}\n"
            f"\\caption{{{tex_escape(fig_caps[num])}}}\n"
            f"\\label{{fig:{num}}}\n"
            "\\end{figure*}\n"
        )

    # Format intro with dropcap on first letter (unheaded Introduction per PNAS instructions)
    intro_paras = [p.strip() for p in intro.split("\n\n") if p.strip()]
    first_p = tex_escape(intro_paras[0])
    first_word, rest_first = first_p.split(" ", 1)
    intro_tex = f"\\dropcap{{{first_word[0]}}}{first_word[1:]} {rest_first}\n\n" + "\n\n".join(tex_escape(p) for p in intro_paras[1:])

    # Results & Discussion paragraphs (insert Figures 1-4 and Tables 1-2 near first mention)
    res_paras = [p.strip() for p in results.split("\n\n") if p.strip()]
    res_tex_list = []
    for p in res_paras:
        res_tex_list.append(tex_escape(p))
        if p.startswith("*Question 1."):
            res_tex_list.append(fig_tex(1))
        elif p.startswith("*Question 2."):
            res_tex_list.append(table1_tex)
        elif p.startswith("To test the constant-odds-ratio mechanism"):
            res_tex_list.append(fig_tex(2))
        elif p.startswith("*Question 3."):
            res_tex_list.append(table2_tex)
            res_tex_list.append(fig_tex(3))
        elif p.startswith("Four post-unsealing exploratory analyses"):
            res_tex_list.append(fig_tex(4))
    results_tex = "\n\n".join(res_tex_list)

    disc_tex = (
        "\n\n".join(tex_escape(p.strip()) for p in before_box.split("\n\n") if p.strip())
        + "\n\n" + box_tex + "\n\n"
        + "\n\n".join(tex_escape(p.strip()) for p in after_box.split("\n\n") if p.strip())
    )

    methods_tex = "\n\n".join(tex_escape(p) for p in methods_keep)

    # References (manual thebibliography matching exact sequential numbers 1..31)
    ref_items = []
    for ln in refs_raw.splitlines():
        m = re.match(r"^(\d+)\.\s+(.*)$", ln.strip())
        if m:
            ref_items.append(f"\\bibitem{{ref{m.group(1)}}} {tex_escape(m.group(2))}")

    tex = rf"""\documentclass[9pt,twocolumn,twoside,lineno]{{pnas-new}}
\templatetype{{pnasresearcharticle}}
\setboolean{{displaywatermark}}{{false}}
\usepackage{{xurl}}
\usepackage{{array}}
\renewcommand{{\dbltopfraction}}{{0.85}}
\renewcommand{{\dblfloatpagefraction}}{{0.75}}
\renewcommand{{\textfraction}}{{0.12}}

\title{{{tex_escape(title)}}}

\author[a,b,1]{{Sanjay Basu}}
\author[b,c]{{Benjamin Q. Huynh}}
\author[d]{{Luyang Luo}}
\author[e]{{Pranav Rajpurkar}}

\affil[a]{{Department of Medicine, University of California San Francisco, San Francisco, CA 94143}}
\affil[b]{{Waymark, San Francisco, CA 94115}}
\affil[c]{{Department of Environmental Health and Engineering, Johns Hopkins Bloomberg School of Public Health, Baltimore, MD 21205}}
\affil[d]{{Department of Biostatistics, Yale School of Public Health, New Haven, CT 06520}}
\affil[e]{{Department of Biomedical Informatics, Harvard Medical School, Boston, MA 02115}}

\leadauthor{{Basu}}

\significancestatement{{{tex_escape(sig)}}}

\authorcontributions{{Author contributions: {tex_escape(auth_contrib)}}}
\authordeclaration{{Competing interests: {tex_escape(coi)}}}
\correspondingauthor{{\textsuperscript{{1}}To whom correspondence should be addressed. E-mail: sanjay.basu@ucsf.edu}}

\keywords{{targeting $|$ heterogeneous treatment effects $|$ risk prediction $|$ pre-registration $|$ replication}}

\begin{{abstract}}
{tex_escape(abs_)}
\end{{abstract}}

\dates{{Classification: Social Sciences (Economic Sciences); Physical Sciences (Statistics).}}
\doi{{Direct Submission}}

\begin{{document}}

\maketitle
\thispagestyle{{firststyle}}
\ifthenelse{{\boolean{{shortarticle}}}}{{\ifthenelse{{\boolean{{singlecolumn}}}}{{\abscontentformatted}}{{\abscontent}}}}{{}}

{intro_tex}

\section*{{Results}}

{results_tex}

\section*{{Discussion}}

{disc_tex}

\matmethods{{{methods_tex}}}
\showmatmethods

\section*{{References}}
\begin{{thebibliography}}{{99}}
{chr(10).join(ref_items)}
\end{{thebibliography}}

\end{{document}}
"""
    LATEX.mkdir(parents=True, exist_ok=True)
    tex_path = LATEX / "manuscript.tex"
    tex_path.write_text(tex)
    for _ in range(3):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
                       cwd=LATEX, check=True, stdout=subprocess.DEVNULL)
    out_pdf = PAPER / "risk_vs_benefit_targeting_22_trials_pnas.pdf"
    shutil.copy2(LATEX / "manuscript.pdf", out_pdf)
    print("wrote", out_pdf)
    return out_pdf


def build_si_pdf() -> Path:
    md = (PAPER / "risk_vs_benefit_targeting_22_trials_si_appendix.md").read_text()
    lines = md.splitlines()
    title = lines[0].lstrip("# ").strip()
    authors = lines[2].strip()

    # Tables with >= 7 columns go on landscape pages so nothing ever overflows
    WIDE_TABLES = {"S1", "S2", "S5", "S6", "S13", "S17", "S19", "S21"}

    body_chunks = []
    i = 4
    n_lines = len(lines)
    while i < n_lines:
        ln = lines[i]
        if ln.startswith("## Table "):
            m = re.match(r"^## Table (S\d+)\.\s*(.*)$", ln)
            tnum = m.group(1) if m else ""
            wide = tnum in WIDE_TABLES
            # Collect everything until next "## " heading
            j = i + 1
            while j < n_lines and not lines[j].startswith("## "):
                j += 1
            block = lines[i + 1:j]
            sec_out = []
            if wide:
                sec_out.append(r"\begin{landscape}")
            sec_out.append(r"\subsection*{Table " + tex_escape(tnum) + ". " + tex_escape(m.group(2) if m else ln[3:]) + "}")
            k = 0
            while k < len(block):
                if block[k].strip().startswith("|"):
                    t_start = k
                    while k < len(block) and block[k].strip().startswith("|"):
                        k += 1
                    hdr, rows = parse_md_table(block[t_start:k])
                    ncols = len(hdr)
                    # Build smart tabularx column spec
                    if tnum == "S1":
                        spec = r">{\raggedright\arraybackslash}p{2.8cm} c >{\raggedright\arraybackslash}p{1.5cm} >{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}p{3.5cm} >{\raggedleft\arraybackslash}p{1.6cm} >{\raggedleft\arraybackslash}p{1.2cm} >{\raggedleft\arraybackslash}p{1.7cm} >{\raggedright\arraybackslash}X"
                    elif tnum == "S2":
                        spec = r">{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}p{2.4cm} r r r >{\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash}X"
                    elif tnum == "S5":
                        spec = r">{\raggedright\arraybackslash}p{3.8cm} r r >{\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash}p{2.6cm} c"
                    elif tnum == "S6":
                        spec = r">{\raggedright\arraybackslash}p{3.6cm} >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{2.8cm}"
                    elif tnum == "S13":
                        spec = r">{\raggedright\arraybackslash}p{3.4cm} r r r r r r r"
                    elif tnum == "S14":
                        spec = r">{\raggedright\arraybackslash}p{5.0cm} >{\raggedright\arraybackslash}p{3.2cm} >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}X"
                    elif tnum == "S15":
                        spec = r">{\raggedright\arraybackslash}p{7.8cm} >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}X"
                    elif tnum == "S17":
                        spec = r">{\raggedright\arraybackslash}p{2.2cm} r >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{2.8cm}"
                    elif tnum == "S19" and ncols == 7:
                        spec = r">{\raggedright\arraybackslash}p{3.6cm} r r >{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}p{2.4cm} >{\raggedright\arraybackslash}p{2.4cm} >{\raggedright\arraybackslash}X"
                    elif tnum == "S19" and ncols == 6:
                        spec = r">{\raggedright\arraybackslash}p{4.6cm} >{\raggedright\arraybackslash}p{3.0cm} r >{\raggedright\arraybackslash}p{3.2cm} >{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}X"
                    elif tnum == "S21":
                        spec = r">{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}p{2.4cm} r r >{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}p{3.0cm} >{\raggedright\arraybackslash}X"
                    elif tnum == "S22":
                        spec = r">{\raggedright\arraybackslash}p{3.4cm} r >{\raggedleft\arraybackslash}p{2.3cm} >{\raggedleft\arraybackslash}p{2.3cm} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{1.6cm}"
                    elif ncols == 2:
                        spec = r">{\raggedright\arraybackslash}X r"
                    elif ncols == 3:
                        spec = r">{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{3.8cm} >{\raggedright\arraybackslash}p{3.2cm}"
                    elif ncols == 4:
                        spec = r">{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}p{3.2cm} >{\raggedright\arraybackslash}p{3.2cm}"
                    elif ncols == 5:
                        spec = r">{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{2.8cm} >{\raggedright\arraybackslash}p{2.5cm} >{\raggedright\arraybackslash}p{2.5cm} >{\raggedright\arraybackslash}p{2.5cm}"
                    else:
                        spec = " ".join([r">{\raggedright\arraybackslash}X"] + [r">{\raggedright\arraybackslash}p{2.3cm}"] * (ncols - 1))
                    sec_out.append(r"{\scriptsize\sffamily")
                    sec_out.append(r"\begin{tabularx}{\linewidth}{" + spec + "}")
                    sec_out.append(r"\toprule")
                    sec_out.append(" & ".join(r"\textbf{" + tex_escape(h) + "}" for h in hdr) + r" \\")
                    sec_out.append(r"\midrule")
                    for r in rows:
                        sec_out.append(" & ".join(tex_escape(c) for c in r) + r" \\")
                    sec_out.append(r"\bottomrule")
                    sec_out.append(r"\end{tabularx}}")
                    sec_out.append(r"\vspace{4pt}")
                elif block[k].strip():
                    ln_s = block[k].strip()
                    if ln_s.startswith("*") and ln_s.endswith("*") and ln_s.count("*") >= 2 and not ln_s.startswith("**"):
                        sec_out.append(r"\noindent{\footnotesize\textit{" + tex_escape(ln_s[1:-1]) + r"}}\par\vspace{4pt}")
                    else:
                        sec_out.append(r"\noindent{\footnotesize " + tex_escape(ln_s) + r"}\par\vspace{4pt}")
                    k += 1
                else:
                    k += 1
            if wide:
                sec_out.append(r"\end{landscape}")
            else:
                sec_out.append(r"\vspace{8pt}")
            body_chunks.append("\n".join(sec_out))
            i = j
        elif ln.startswith("## "):
            body_chunks.append(r"\section*{" + tex_escape(ln[3:].strip()) + "}")
            i += 1
        elif ln.strip().startswith("$$"):
            body_chunks.append(r"\[" + ln.strip().strip("$").replace("−", "-").replace(r"\operatorname{logit}", r"\mathrm{logit}\,") + r"\]")
            i += 1
        elif ln.strip():
            body_chunks.append(tex_escape(ln.strip()) + "\n")
            i += 1
        else:
            i += 1

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs,tabularx,pdflscape,array}}
\usepackage[colorlinks=true,allcolors=blue]{{hyperref}}
\usepackage{{fancyhdr}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\footnotesize\sffamily SI Appendix: Risk vs.\ Benefit Targeting in 22 Randomized Trials}}
\fancyhead[R]{{\footnotesize\sffamily Basu et al.}}
\fancyfoot[C]{{\footnotesize\sffamily S\thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{5pt}}

\begin{{document}}
\begin{{center}}
{{\Large\sffamily\bfseries {tex_escape(title)}}}\\[6pt]
{{\normalsize\sffamily {tex_escape(authors)}}}
\end{{center}}
\vspace{{6pt}}

{chr(10).join(body_chunks)}

\end{{document}}
"""
    tex_path = LATEX / "si_appendix.tex"
    tex_path.write_text(tex)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "si_appendix.tex"],
                       cwd=LATEX, check=True, stdout=subprocess.DEVNULL)
    out_pdf = PAPER / "risk_vs_benefit_targeting_22_trials_si_appendix.pdf"
    shutil.copy2(LATEX / "si_appendix.pdf", out_pdf)
    print("wrote", out_pdf)
    return out_pdf


def main() -> None:
    p1 = build_main_pnas()
    p2 = build_si_pdf()
    for p in (p1, p2):
        info = subprocess.check_output(["pdfinfo", str(p)], text=True)
        pages = next(l for l in info.splitlines() if l.startswith("Pages:"))
        print(p.name, "->", pages)


if __name__ == "__main__":
    main()
