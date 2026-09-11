#!/usr/bin/env python3
"""
build_full_manuscript_docx.py

Converts the complete final revision of the manuscript (HydroGraphEM_Manuscript.tex,
matching main (17).pdf) into an authoritative, publication-ready Microsoft Word (.docx) file.

Pipeline:
1. Preprocess HydroGraphEM_Manuscript.tex:
   - Resolve all \\label and \\ref tags to explicit sequential numbers (Tables 1-13, Figures 1-9, Eq. 16, Sec. 6.4).
   - Format Table captions as "Table N: <caption text>".
   - Format Figure captions as "Figure N: <caption text>".
   - Reconstruct the frontmatter (Title, Authors with affiliations, Corresponding author block, Abstract, Keywords).
2. Execute Pandoc with --resource-path=manuscript --bibliography=manuscript/references.bib --citeproc.
3. Post-process the generated DOCX with python-docx:
   - Page setup: A4 with 1.0-inch margins throughout.
   - Base typography: Times New Roman 11pt, 1.15 line spacing.
   - Format all 13 tables to strict 3-line academic booktabs styling (top/bottom heavy lines, sub-header line, no vertical lines).
   - Format headings with clean professional spacing.
   - Embed all 9 figures cleanly with centered captions.
   - Save output to:
     - manuscript/HydroGraphEM_Manuscript_Revision17.docx
     - manuscript/HydroGraphEM_Manuscript.docx
     - HydroGraphEM_Manuscript.docx (workspace root for easy access)
"""

import os
import re
import subprocess
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANUSCRIPT_DIR = os.path.join(BASE_DIR, "manuscript")
TEX_PATH = os.path.join(MANUSCRIPT_DIR, "HydroGraphEM_Manuscript.tex")
BIB_PATH = os.path.join(MANUSCRIPT_DIR, "references.bib")


def preprocess_latex(tex_content):
    """Resolve cross references, table numbers, figure numbers, and frontmatter."""
    # Define Table and Figure mappings matching the document's sequential order
    table_map = {
        'tab:novelty': 1,
        'tab:compute': 2,
        'tab:temporal': 3,
        'tab:extreme': 4,
        'tab:spatial': 5,
        'tab:ablation_encoder': 6,
        'tab:context_ablation': 7,
        'tab:pretrain_ablation': 8,
        'tab:t_test': 9,
        'tab:rolling_origin': 10,
        'tab:binary_classification': 11,
        'tab:mc_dropout': 12,
        'tab:ci': 13
    }

    fig_map = {
        'fig:study_area': 1,
        'fig:flowchart': 2,
        'fig:topology': 3,
        'fig:network': 4,
        'fig:projections': 5,
        'fig:attention': 6,
        'fig:cka': 7,
        'fig:importance': 8,
        'fig:performance': 9
    }

    content = tex_content

    # Fix compound range references first
    content = content.replace(r"Tables~\ref{tab:temporal}--\ref{tab:extreme}", "Tables 3–4")
    content = content.replace(r"Tables~\ref{tab:temporal} and \ref{tab:extreme}", "Tables 3 and 4")
    content = content.replace(r"Tables~\ref{tab:temporal} and [tab:extreme]", "Tables 3 and 4")
    content = content.replace(r"(Eq.~\eqref{eq:bias_correction})", "(Eq. 16)")
    content = content.replace(r"\eqref{eq:bias_correction}", "Eq. 16")
    content = content.replace(r"\ref{eq:bias_correction}", "16")
    content = content.replace(r"Section~\ref{sec:limitations}", "Section 6.4")
    content = content.replace(r"\ref{sec:limitations}", "6.4")

    # Replace individual table and figure refs
    for label, num in table_map.items():
        content = re.sub(rf'Table~\\ref{{{label}}}', f'Table {num}', content)
        content = re.sub(rf'\\ref{{{label}}}', str(num), content)

    for label, num in fig_map.items():
        content = re.sub(rf'Figure~\\ref{{{label}}}', f'Figure {num}', content)
        content = re.sub(rf'Fig\.~\\ref{{{label}}}', f'Fig. {num}', content)
        content = re.sub(rf'\\ref{{{label}}}', str(num), content)

    # Replace remaining \ref{...} safely if any
    content = re.sub(r'\\ref\{([^}]+)\}', r'\1', content)

    # Format Table captions explicitly: \caption{Text}\label{tab:...} -> \caption{\textbf{Table N.} Text}
    for label, num in table_map.items():
        # Match caption and label
        pattern = re.compile(rf'\\caption\{{([^}}]+)\}}\s*\\label\{{{label}\}}', re.DOTALL)
        content = pattern.sub(lambda m, n=num: f"\\caption{{\\textbf{{Table {n}.}} {m.group(1)}}}", content)

    # Format Figure captions explicitly: \caption{Text}\label{fig:...} -> \caption{\textbf{Figure N.} Text}
    for label, num in fig_map.items():
        pattern = re.compile(rf'\\caption\{{([^}}]+)\}}\s*\\label\{{{label}\}}', re.DOTALL)
        content = pattern.sub(lambda m, n=num: f"\\caption{{\\textbf{{Figure {n}.}} {m.group(1)}}}", content)

    # Remove \resizebox{\textwidth}{!}{...} wrapper around tabulars so Pandoc parses tables cleanly
    content = re.sub(r'\\resizebox\{\\textwidth\}\{!\}' + r'\s*\{', '', content)
    # Note: we need to balance the closing brace for resizebox.
    # Each resizebox ends before \end{table}. Let's clean that up:
    content = re.sub(r'(\\end\{tabular\})\s*\}', r'\1', content)

    # Reconstruct frontmatter for clean Pandoc title block
    frontmatter_replacement = r"""
\title{\textbf{HydroGraphEM: A Self-Supervised Graph Learning Framework for Multi-Task Watershed Emulation of the Indus Delta}}

\author{
\textbf{Mirza Muhammad Muzzamil}$^{1,*}$,
\textbf{Muhammad Ali Ismael}$^{1}$,
\textbf{Syed Imran Ahmad}$^{2}$,
\textbf{Rustam B. Rustomov}$^{3}$
}

\date{}

\maketitle

\noindent $^{1}$ Department of Computer and Information Systems Engineering (CISE), NED University of Engineering and Technology, University Road, Karachi 75270, Pakistan\\
$^{2}$ Department of Civil Engineering, NED University of Engineering and Technology, University Road, Karachi 75270, Pakistan\\
$^{3}$ Institute of Physics, Ministry of Science and Education of the Republic of Azerbaijan, 33 H. Javid Ave., Baku AZ1143, Azerbaijan\\
$^{*}$ \textbf{Corresponding Author:} Mirza Muhammad Muzzamil; Email: \texttt{mirzamuzzamil@neduet.edu.pk}; Tel: +92-334-2976696; ORCID: 0000-0001-5258-8959

\vspace{1em}

\section*{Abstract}
Conventional hydrological models like SWAT are trusted for watershed planning but face severe computational limitations and cannot ingest dense satellite observations. To bridge this gap, we present \textbf{HydroGraphEM}, a self-supervised spatiotemporal graph learning framework that serves as a surrogate emulator of SWAT-generated watershed responses. HydroGraphEM constructs a directed, hierarchical graph topology (mapping Hydrological Response Units to subbasins and reaches) and fuses SWAT simulations with satellite-derived meteorology and vegetation indices. A spatial-temporal attention graph transformer is pretrained via temporal mask reconstruction to learn routing mechanics and emergent mass-balance relationships without downstream labels. We evaluate HydroGraphEM on the Indus Delta (Pakistan) across a multi-task benchmark (flood susceptibility, water quality, and risk classification). Under chronological holdout (test set: 2019--2020), the model achieves $R^2 = 0.6708$ for flood susceptibility and $R^2 = 0.6850$ for stream Nitrate, outperforming randomly initialized baselines ($p < 0.001$). On a spatial block holdout of 105 reaches, it yields Pearson $r = 0.38$--$0.60$ for temporal routing dynamics, though absolute calibration requires post-hoc bias correction. A key limitation of the framework is that drought emulation is a documented negative result ($R^2 < 0$), as temporal context windows up to 60 months cannot capture multi-year groundwater memory. Furthermore, validation is currently restricted to a single basin. Saliency attributions trace physically plausible transport chains, confirming physical interpretability. These findings establish HydroGraphEM as an accurate, explainable emulation framework for multi-task watershed intelligence.

\vspace{0.5em}
\noindent \textbf{Keywords:} Graph learning; SWAT emulation; Representation learning; Watershed modelling; Flood susceptibility; Water quality; Indus Delta.

\vspace{1em}
"""

    # Replace \begin{frontmatter} ... \end{frontmatter}
    content = re.sub(r'\\begin\{frontmatter\}.*?\\end\{frontmatter\}', lambda m: frontmatter_replacement, content, flags=re.DOTALL)

    return content


def set_academic_table_borders(table):
    """
    Format table with standard 3-line academic booktabs style:
    - Heavy top border on header
    - Light bottom border on header
    - Heavy bottom border on last row
    - No vertical lines
    """
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    num_rows = len(table.rows)
    for r_idx, row in enumerate(table.rows):
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            # Determine borders
            if r_idx == 0:
                top_val, top_sz = "single", "12"
                bottom_val, bottom_sz = "single", "6"
            elif r_idx == num_rows - 1:
                top_val, top_sz = "none", "0"
                bottom_val, bottom_sz = "single", "12"
            else:
                top_val, top_sz = "none", "0"
                bottom_val, bottom_sz = "none", "0"

            tcBorders = parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                f'<w:top w:val="{top_val}" w:sz="{top_sz}" w:space="0" w:color="auto"/>'
                f'<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
                f'<w:bottom w:val="{bottom_val}" w:sz="{bottom_sz}" w:space="0" w:color="auto"/>'
                f'<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
                f'</w:tcBorders>'
            )
            tcPr.append(tcBorders)

            # Cell padding
            tcMar = parse_xml(
                f'<w:tcMar {nsdecls("w")}>'
                f'<w:top w:w="80" w:type="dxa"/>'
                f'<w:left w:w="120" w:type="dxa"/>'
                f'<w:bottom w:w="80" w:type="dxa"/>'
                f'<w:right w:w="120" w:type="dxa"/>'
                f'</w:tcMar>'
            )
            tcPr.append(tcMar)

            # Style text inside cell
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.05
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
                for run in p.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(9.5)
                    if r_idx == 0:
                        run.bold = True


def post_process_docx(docx_path):
    """Apply styling, margins, fonts, and clean table borders."""
    doc = docx.Document(docx_path)

    # 1. Page Margins & Setup
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        s.page_width = Inches(8.27)
        s.page_height = Inches(11.69)

    # 2. Base Normal style
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(4)

    # 3. Tables styling
    for table in doc.tables:
        set_academic_table_borders(table)

    # 4. Paragraph and Heading styling
    for p in doc.paragraphs:
        # If heading style
        if p.style.name.startswith('Heading 1'):
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(14)
                r.bold = True
                r.font.color.rgb = RGBColor(0, 0, 0)
        elif p.style.name.startswith('Heading 2'):
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(12)
                r.bold = True
                r.font.color.rgb = RGBColor(0, 0, 0)
        elif p.style.name.startswith('Heading 3'):
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(11)
                r.bold = True
                r.font.color.rgb = RGBColor(0, 0, 0)
        elif p.style.name == 'Title':
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(12)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(16)
                r.bold = True
                r.font.color.rgb = RGBColor(0, 0, 0)
        else:
            # Body text font check
            for r in p.runs:
                if not r.font.name:
                    r.font.name = 'Times New Roman'

        # Center figure and table captions
        if p.text.startswith(('Figure ', 'Fig. ', 'Table ')):
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)

    doc.save(docx_path)
    print(f"Post-processing complete for: {docx_path}")


def main():
    print(f"Reading source LaTeX: {TEX_PATH}")
    with open(TEX_PATH, "r", encoding="utf-8") as f:
        tex_raw = f.read()

    print("Preprocessing cross references, tables, figures, and frontmatter...")
    tex_clean = preprocess_latex(tex_raw)

    temp_tex = os.path.join(MANUSCRIPT_DIR, "_temp_clean_manuscript.tex")
    with open(temp_tex, "w", encoding="utf-8") as f:
        f.write(tex_clean)

    target_docx_rel = os.path.join(MANUSCRIPT_DIR, "HydroGraphEM_Manuscript_Revision17.docx")
    final_manuscript_docx = os.path.join(MANUSCRIPT_DIR, "HydroGraphEM_Manuscript.docx")
    root_manuscript_docx = os.path.join(BASE_DIR, "HydroGraphEM_Manuscript.docx")

    cmd = [
        "pandoc",
        temp_tex,
        "-s",
        f"--resource-path={MANUSCRIPT_DIR}",
        f"--bibliography={BIB_PATH}",
        "--citeproc",
        "-o",
        target_docx_rel
    ]

    print("Running Pandoc conversion with citeproc and high-res media embedding...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Pandoc error:\n{result.stderr}")
        sys.exit(result.returncode)

    if result.stderr:
        print(f"Pandoc warnings/notes:\n{result.stderr}")

    print("Applying publication typography and academic table styling...")
    post_process_docx(target_docx_rel)

    # Copy to primary targets
    import shutil
    shutil.copy2(target_docx_rel, final_manuscript_docx)
    shutil.copy2(target_docx_rel, root_manuscript_docx)
    print(f"Successfully generated final revision Word documents:")
    print(f"  1. {target_docx_rel}")
    print(f"  2. {final_manuscript_docx}")
    print(f"  3. {root_manuscript_docx}")

    # Remove temporary tex
    if os.path.exists(temp_tex):
        os.remove(temp_tex)

    print("Build complete!")


if __name__ == "__main__":
    main()
