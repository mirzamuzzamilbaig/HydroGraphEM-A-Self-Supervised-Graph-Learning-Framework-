#!/usr/bin/env python3
"""
build_ieee_manuscript_docx.py

Generates an authoritative, publication-grade IEEE Transactions / Journal Word document (.docx)
conforming strictly to IEEE authoring standards:

1. Title & Metadata (Section 1: 1-column full-width, 7.25 in):
   - Title: 20pt Bold, Centered, Times New Roman.
   - Author Line: Mirza Muhammad Muzzamil, Graduate Student Member, IEEE, Muhammad Ali Ismael, Senior Member, IEEE, Syed Imran Ahmad, and Rustam B. Rustomov.
   - Institutional Affiliations: Compact, italic, centered 8.5pt font with emails and corresponding author note.
2. Continuous Section Break on Page 1:
   - Inserted in <w:pPr> of the last affiliation paragraph (<w:sectPr><w:type w:val="continuous"/><w:cols w:num="1"/></w:sectPr>).
3. Body Text (Section 2: 2-column layout on Page 1 onwards, w:cols w:num="2" w:space="360"):
   - Column 1 on Page 1 starts with:
       * Abstract--- (Bold-italic 9pt) followed by abstract text (Bold 9pt).
       * Index Terms--- (Bold-italic 9pt) followed by index terms (9pt).
       * I. INTRODUCTION (Centered 10pt Bold All-Caps).
       * Body text flowing down Column 1 and into Column 2!
   - Section Headings: Centered, 10pt Bold All-Caps Roman numerals (I. INTRODUCTION, II. STUDY AREA AND DATASETS, etc.).
   - Subsection Headings: Left-aligned, 9.5pt Bold Italic Alphabetical (A. HST-PAGT Encoder, etc.).
   - Body Paragraphs: 9.5pt Times New Roman, 1.05 line spacing, 0.14 in first-line indent, Justified.
   - Figures: Centered with "Fig. N.  <caption text>" in 8pt font below the figure.
   - Tables: Centered with IEEE two-tier caption (TABLE N \n TITLE IN UPPERCASE), 3-line IEEE borders, 7.5pt font.
   - References: Centered "REFERENCES" heading, numbered IEEE citations [1], [2], hanging indent, 8pt font.
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
CSL_PATH = os.path.join(MANUSCRIPT_DIR, "ieee.csl")

ROMAN_NUMERALS = {
    1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
    6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X',
    11: 'XI', 12: 'XII', 13: 'XIII'
}

def clean_ieee_latex(raw_tex):
    """Preprocess LaTeX to produce clean, well-tagged IEEE source for Pandoc."""
    content = raw_tex

    # 1. Resolve cross references
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

    content = content.replace(r"Tables~\ref{tab:temporal}--\ref{tab:extreme}", "Tables III--IV")
    content = content.replace(r"Tables~\ref{tab:temporal} and \ref{tab:extreme}", "Tables III and IV")
    content = content.replace(r"Tables~\ref{tab:temporal} and [tab:extreme]", "Tables III and IV")
    content = content.replace(r"(Eq.~\eqref{eq:bias_correction})", "(16)")
    content = content.replace(r"\eqref{eq:bias_correction}", "(16)")
    content = content.replace(r"\ref{eq:bias_correction}", "(16)")
    content = content.replace(r"Section~\ref{sec:limitations}", "Section VI-D")
    content = content.replace(r"\ref{sec:limitations}", "VI-D")

    for label, num in table_map.items():
        roman = ROMAN_NUMERALS[num]
        content = re.sub(rf'Table~\\ref{{{label}}}', f'Table {roman}', content)
        content = re.sub(rf'\\ref{{{label}}}', roman, content)

    for label, num in fig_map.items():
        content = re.sub(rf'Figure~\\ref{{{label}}}', f'Fig. {num}', content)
        content = re.sub(rf'Fig\.~\\ref{{{label}}}', f'Fig. {num}', content)
        content = re.sub(rf'\\ref{{{label}}}', str(num), content)

    content = re.sub(r'\\ref\{([^}]+)\}', r'\1', content)

    # 2. Table Captions: IEEE Style (TABLE N \newline TITLE IN UPPERCASE)
    for label, num in table_map.items():
        roman = ROMAN_NUMERALS[num]
        pattern = re.compile(rf'\\caption\{{([^}}]+)\}}\s*\\label\{{{label}\}}', re.DOTALL)
        def table_repl(m, r=roman):
            title = m.group(1).strip().replace('\n', ' ')
            return f"\\caption{{\\textbf{{TABLE {r}}}\\\\ \\textsc{{{title}}}}}"
        content = pattern.sub(table_repl, content)

    # 3. Figure Captions: IEEE Style (Fig. N.  Caption)
    for label, num in fig_map.items():
        pattern = re.compile(rf'\\caption\{{([^}}]+)\}}\s*\\label\{{{label}\}}', re.DOTALL)
        def fig_repl(m, n=num):
            caption_text = m.group(1).strip().replace('\n', ' ')
            return f"\\caption{{\\textbf{{Fig. {n}.}} {caption_text}}}"
        content = pattern.sub(fig_repl, content)

    # Remove resizebox around tables
    content = re.sub(r'\\resizebox\{\\textwidth\}\{!\}' + r'\s*\{', '', content)
    content = re.sub(r'(\\end\{tabular\})\s*\}', r'\1', content)

    # 4. Section titles: IEEE Roman numerals
    sec_map = [
        (r'\section{Introduction}', r'\section{I. INTRODUCTION}'),
        (r'\section{Study Area and Datasets}', r'\section{II. STUDY AREA AND DATASETS}'),
        (r'\section{HydroGraphEM Framework and Architecture}', r'\section{III. HYDROGRAPHEM FRAMEWORK AND ARCHITECTURE}'),
        (r'\section{Experimental Design}', r'\section{IV. EXPERIMENTAL DESIGN}'),
        (r'\section{Results}', r'\section{V. RESULTS}'),
        (r'\section{Discussion}', r'\section{VI. DISCUSSION}'),
        (r'\section{Conclusion}', r'\section{VII. CONCLUSION}')
    ]
    for old, new in sec_map:
        content = content.replace(old, new)

    # Subsections: Letters A, B, C...
    # Section III
    content = content.replace(r'\subsection{HST-PAGT Encoder}', r'\subsection{A. HST-PAGT Encoder}')
    content = content.replace(r'\subsection{Self-Supervised Pretraining}', r'\subsection{B. Self-Supervised Pretraining}')
    content = content.replace(r'\subsection{Implementation and Reproducibility Details}', r'\subsection{C. Implementation and Reproducibility Details}')
    content = content.replace(r'\subsection{Computational Cost and Model Complexity}', r'\subsection{D. Computational Cost and Model Complexity}')
    # Section IV
    content = content.replace(r'\subsection{Downstream Task Formulations}', r'\subsection{A. Downstream Task Formulations}')
    content = content.replace(r'\subsection{Pretrain-Freeze-Transfer Protocol}', r'\subsection{B. Pretrain-Freeze-Transfer Protocol}')
    content = content.replace(r'\subsection{Statistical Significance Testing Framework}', r'\subsection{C. Statistical Significance Testing Framework}')
    content = content.replace(r'\subsection{Baseline Configurations}', r'\subsection{D. Baseline Configurations}')
    content = content.replace(r'\subsection{Rolling-Origin Cross-Validation Protocol}', r'\subsection{E. Rolling-Origin Cross-Validation Protocol}')
    content = content.replace(r'\subsection{Uncertainty Quantification via Monte Carlo Dropout}', r'\subsection{F. Uncertainty Quantification via Monte Carlo Dropout}')
    # Section V
    content = content.replace(r'\subsection{Temporal Generalization Performance}', r'\subsection{A. Temporal Generalization Performance}')
    content = content.replace(r'\subsection{Extreme Event Generalization}', r'\subsection{B. Extreme Event Generalization}')
    content = content.replace(r'\subsection{Spatial Representation Transfer Analysis}', r'\subsection{C. Spatial Representation Transfer Analysis}')
    content = content.replace(r'\subsection{Validation of the Underlying SWAT Simulation Model}', r'\subsection{D. Validation of the Underlying SWAT Simulation Model}')
    content = content.replace(r'\subsection{Spatial Representation Dynamics}', r'\subsection{E. Spatial Representation Dynamics}')
    content = content.replace(r'\subsection{Ablation Experiments}', r'\subsection{F. Ablation Experiments}')
    content = content.replace(r'\subsection{Temporal Context-Window Ablation}', r'\subsection{G. Temporal Context-Window Ablation}')
    content = content.replace(r'\subsection{Spatial Routing Attention Analysis}', r'\subsection{H. Spatial Routing Attention Analysis}')
    content = content.replace(r'\subsection{Structural Probing via Strahler Stream Order}', r'\subsection{I. Structural Probing via Strahler Stream Order}')
    content = content.replace(r'\subsection{CKA Representational Similarity Analysis}', r'\subsection{J. CKA Representational Similarity Analysis}')
    content = content.replace(r'\subsection{Explainability and Feature Attributions via Integrated Gradients}', r'\subsection{K. Explainability and Feature Attributions via Integrated Gradients}')
    content = content.replace(r'\subsection{Downstream Uncertainty Bounds via Monte Carlo Dropout}', r'\subsection{L. Downstream Uncertainty Bounds via Monte Carlo Dropout}')
    content = content.replace(r'\subsection{Bootstrapped Uncertainty Bounds on Test Performance}', r'\subsection{M. Bootstrapped Uncertainty Bounds on Test Performance}')
    # Section VI
    content = content.replace(r'\subsection{Physical Plausibility and Process Realism}', r'\subsection{A. Physical Plausibility and Process Realism}')
    content = content.replace(r'\subsection{Saliency Attribution and Transport Mechanics}', r'\subsection{B. Saliency Attribution and Transport Mechanics}')
    content = content.replace(r'\subsection{Practical Implications for Water Resource Operations}', r'\subsection{C. Practical Implications for Water Resource Operations}')
    content = content.replace(r'\subsection{Limitations and Future Work}', r'\subsection{D. Limitations and Future Work}')

    # 5. Extract Abstract and Keywords text from raw_tex
    abs_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', raw_tex, re.DOTALL)
    abstract_text = abs_match.group(1).strip().replace('\n', ' ') if abs_match else ""

    # Frontmatter structure:
    # Title & Authors & Affiliations
    # FOLLOWED BY marker %%% IEEE_CONTINUOUS_BREAK %%%
    # FOLLOWED BY Abstract and Index terms (which will be in Section 2, column 1!)
    ieee_front = f"""
\\title{{\\LARGE \\textbf{{HydroGraphEM: A Self-Supervised Graph Learning Framework for Multi-Task Watershed Emulation of the Indus Delta}}}}

\\author{{
\\textbf{{Mirza Muhammad Muzzamil}}, \\textit{{Graduate Student Member, IEEE}},
\\textbf{{Muhammad Ali Ismael}}, \\textit{{Senior Member, IEEE}},\\\\
\\textbf{{Syed Imran Ahmad}}, and
\\textbf{{Rustam B. Rustomov}}
}}

\\date{{}}

\\maketitle

\\noindent \\textit{{Department of Computer and Information Systems Engineering, NED University of Engineering and Technology, Karachi 75270, Pakistan}}\\\\
\\noindent \\textit{{Department of Civil Engineering, NED University of Engineering and Technology, Karachi 75270, Pakistan}}\\\\
\\noindent \\textit{{Institute of Physics, Ministry of Science and Education of the Republic of Azerbaijan, Baku AZ1143, Azerbaijan}}\\\\
\\noindent \\textit{{(Corresponding author: Mirza Muhammad Muzzamil, e-mail: mirzamuzzamil@neduet.edu.pk)}}

\\noindent %%% IEEE_CONTINUOUS_BREAK %%%

\\noindent \\textbf{{\\textit{{Abstract}}}}---{abstract_text}

\\vspace{{0.4em}}

\\noindent \\textbf{{\\textit{{Index Terms}}}}---Graph learning, SWAT emulation, self-supervised learning, spatio-temporal graph transformers, watershed modeling, flood susceptibility, water quality, Indus Delta.

\\vspace{{0.8em}}
"""
    content = re.sub(r'\\begin\{frontmatter\}.*?\\end\{frontmatter\}', lambda m: ieee_front, content, flags=re.DOTALL)

    return content


def format_table_ieee(table):
    """Format table strictly according to IEEE standards."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    num_rows = len(table.rows)
    for r_idx, row in enumerate(table.rows):
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            # IEEE Borders: Top 12 (1.5pt), Mid 6 (0.75pt), Bottom 12 (1.5pt), No verticals
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

            # Cell padding: tight IEEE 30 twips top/bot, 50 twips left/right
            tcMar = parse_xml(
                f'<w:tcMar {nsdecls("w")}>'
                f'<w:top w:w="30" w:type="dxa"/>'
                f'<w:left w:w="50" w:type="dxa"/>'
                f'<w:bottom w:w="30" w:type="dxa"/>'
                f'<w:right w:w="50" w:type="dxa"/>'
                f'</w:tcMar>'
            )
            tcPr.append(tcMar)

            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(7.5)
                    if r_idx == 0:
                        run.bold = True


def post_process_ieee_docx(docx_path):
    """Apply authentic IEEE two-column section break, typography, and styling."""
    doc = docx.Document(docx_path)

    # 1. Page Setup: IEEE Standard (8.5" x 11.0", Top 0.75", Bottom 1.0", Left/Right 0.625")
    # Base Normal Style
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.05
    normal.paragraph_format.space_after = Pt(2)

    # 2. Setup the Two-Column Section Break right before Abstract
    # Find paragraph containing '%%% IEEE_CONTINUOUS_BREAK %%%' or paragraph right before 'Abstract'
    break_p_idx = -1
    for idx, p in enumerate(doc.paragraphs):
        if '%%% IEEE_CONTINUOUS_BREAK %%%' in p.text:
            break_p_idx = idx
            break
        elif p.text.strip().startswith('Abstract') and idx > 0:
            break_p_idx = idx - 1
            break

    if break_p_idx != -1:
        target_p = doc.paragraphs[break_p_idx]
        pPr = target_p._p.get_or_add_pPr()

        # Insert Section 1 sectPr (1 column, continuous, full width)
        sectPr_sec1 = parse_xml(
            f'<w:sectPr {nsdecls("w")}>'
            f'<w:type w:val="continuous"/>'
            f'<w:cols w:num="1"/>'
            f'</w:sectPr>'
        )
        pPr.append(sectPr_sec1)

        # Clear marker text if present
        if '%%% IEEE_CONTINUOUS_BREAK %%%' in target_p.text:
            target_p.text = ""

        # Configure Section 2 (Body: 2 columns, space = 360 twips = 0.25", continuous)
        body_sectPr = doc._body._element.xpath('w:sectPr')
        if body_sectPr:
            b_sp = body_sectPr[0]
            # Remove existing type/cols/pgMar to avoid duplication
            for old_elem in b_sp.xpath('w:type') + b_sp.xpath('w:cols') + b_sp.xpath('w:pgMar') + b_sp.xpath('w:pgSz'):
                b_sp.remove(old_elem)

            # Insert type continuous first
            b_sp.insert(0, parse_xml(f'<w:type {nsdecls("w")} w:val="continuous"/>'))
            b_sp.append(parse_xml(f'<w:pgSz {nsdecls("w")} w:w="12240" w:h="15840"/>'))
            b_sp.append(parse_xml(f'<w:pgMar {nsdecls("w")} w:top="1080" w:right="900" w:bottom="1440" w:left="900"/>'))
            b_sp.append(parse_xml(f'<w:cols {nsdecls("w")} w:num="2" w:space="360"/>'))

    # 3. Format Tables
    for table in doc.tables:
        format_table_ieee(table)

    # 4. Format Paragraphs & Typography
    abstract_seen = False
    for idx, p in enumerate(doc.paragraphs):
        text_strip = p.text.strip()

        # Document Title
        if 'HydroGraphEM: A Self-Supervised' in text_strip and len(text_strip) < 140:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(6)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(20)
                r.bold = True

        # Author line
        elif text_strip.startswith('Mirza Muhammad Muzzamil'):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(11)

        # Affiliations (Header block)
        elif text_strip.startswith(('Department of Computer and Information Systems', 'Department of Civil Engineering', 'Institute of Physics, Ministry', '(Corresponding author:')):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(1)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(8.5)
                r.italic = True

        # Abstract (Left column of Section 2)
        elif text_strip.startswith('Abstract'):
            abstract_seen = True
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.first_line_indent = Inches(0)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(9)
                if 'Abstract' in r.text:
                    r.bold = True
                    r.italic = True

        # Index Terms (Left column of Section 2)
        elif text_strip.startswith('Index Terms'):
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.first_line_indent = Inches(0)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(9)
                if 'Index Terms' in r.text:
                    r.bold = True
                    r.italic = True

        # Level 1 Heading (e.g. I. INTRODUCTION)
        elif any(text_strip.startswith(f'{r}. ') for r in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']) or text_strip in ['REFERENCES', 'ACKNOWLEDGMENT']:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(9)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.first_line_indent = Inches(0)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(10)
                r.bold = True
                r.font.color.rgb = RGBColor(0, 0, 0)

        # Level 2 Heading (e.g. A. HST-PAGT Encoder)
        elif re.match(r'^[A-M]\.\s+[A-Za-z]', text_strip):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.first_line_indent = Inches(0)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(9.5)
                r.bold = True
                r.italic = True
                r.font.color.rgb = RGBColor(0, 0, 0)

        # Figure Caption
        elif text_strip.startswith(('Fig. ', 'Figure ')):
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.first_line_indent = Inches(0)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(8)

        # Table Caption (TABLE N)
        elif text_strip.startswith('TABLE '):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.first_line_indent = Inches(0)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(8)
                r.bold = True

        # Reference item: [1], [2]
        elif re.match(r'^\[\d+\]', text_strip):
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.left_indent = Inches(0.2)
            p.paragraph_format.first_line_indent = Inches(-0.2)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(8)

        # Body Text
        else:
            if abstract_seen:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.first_line_indent = Inches(0.14)
                p.paragraph_format.space_after = Pt(1)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                if not r.font.size:
                    r.font.size = Pt(9.5)

    doc.save(docx_path)
    print(f"IEEE formatting applied and saved: {docx_path}")


def main():
    print("Pre-processing LaTeX for authentic IEEE formatting...")
    with open(TEX_PATH, "r", encoding="utf-8") as f:
        raw_tex = f.read()

    ieee_tex = clean_ieee_latex(raw_tex)

    temp_tex = os.path.join(MANUSCRIPT_DIR, "_temp_ieee_clean.tex")
    with open(temp_tex, "w", encoding="utf-8") as f:
        f.write(ieee_tex)

    target_manuscript_docx = os.path.join(MANUSCRIPT_DIR, "HydroGraphEM_Manuscript_IEEE_Format.docx")
    target_root_docx = os.path.join(BASE_DIR, "HydroGraphEM_Manuscript_IEEE_Format.docx")

    cmd = [
        "pandoc",
        temp_tex,
        "-s",
        f"--resource-path={MANUSCRIPT_DIR}",
        f"--bibliography={BIB_PATH}",
        f"--csl={CSL_PATH}",
        "--citeproc",
        "-o",
        target_manuscript_docx
    ]

    print("Running Pandoc with IEEE CSL numeric style...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Pandoc error:\n{res.stderr}")
        sys.exit(res.returncode)

    print("Post-processing Word document for strict IEEE standards...")
    post_process_ieee_docx(target_manuscript_docx)

    import shutil
    shutil.copy2(target_manuscript_docx, target_root_docx)

    if os.path.exists(temp_tex):
        os.remove(temp_tex)

    print("IEEE Word document build complete!")


if __name__ == "__main__":
    main()
