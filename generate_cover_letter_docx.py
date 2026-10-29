#!/usr/bin/env python3
"""
generate_cover_letter_docx.py

Generates professional, publication-ready submission Cover Letters in Microsoft Word (.docx) format:
1. Cover_Letter_Environmental_Modelling_and_Software.docx (Elsevier)
2. Cover_Letter_Geosciences_Journal.docx (Springer / MDPI)
3. Cover_Letter.docx (Default standard submission letter)

Includes:
- Academic institutional letterhead header
- Author contact, ORCID, and institutional affiliation
- Complete manuscript metadata & author list
- Core scientific contributions & quantitative benchmark highlights
- Explainability & physical consistency findings
- Mandatory author declarations & open-data availability
- Suggested peer reviewers with expert affiliations and emails
- Professional typography (Times New Roman 11pt, clean line spacing, elegant dividers)
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn


def set_cell_margins(cell, top=60, bottom=60, left=100, right=100):
    """Set inner padding for table cells (in twips: 20 twips = 1 pt)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def build_cover_letter(target_journal="EMS", output_path="Cover_Letter.docx"):
    doc = docx.Document()

    # Standard Page Setup: Letter / A4 with 1-inch margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        s.page_width = Inches(8.27)
        s.page_height = Inches(11.69)

    # Base typography: Times New Roman 11pt
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(4)

    def add_p(text="", bold=False, italic=False, space_after=4, space_before=0, align=WD_ALIGN_PARAGRAPH.LEFT):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.space_before = Pt(space_before)
        if text:
            r = p.add_run(text)
            r.bold = bold
            r.italic = italic
            r.font.name = 'Times New Roman'
            r.font.size = Pt(11)
            r.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
        return p

    # -------------------------------------------------------------------------
    # 1. INSTITUTIONAL LETTERHEAD
    # -------------------------------------------------------------------------
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_after = Pt(2)
    r_inst = p_inst.add_run("NED UNIVERSITY OF ENGINEERING AND TECHNOLOGY")
    r_inst.bold = True
    r_inst.font.size = Pt(12.5)
    r_inst.font.color.rgb = RGBColor(0x0C, 0x34, 0x61)  # Academic Navy

    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dept.paragraph_format.space_after = Pt(2)
    r_dept = p_dept.add_run("Department of Computer and Information Systems Engineering (CISE)")
    r_dept.font.size = Pt(10)
    r_dept.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    p_addr = doc.add_paragraph()
    p_addr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_addr.paragraph_format.space_after = Pt(6)
    r_addr = p_addr.add_run("University Road, Karachi 75270, Sindh, Pakistan | Tel: +92-334-2976696")
    r_addr.font.size = Pt(9.5)
    r_addr.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # Decorative dividing border under letterhead
    p_line = doc.add_paragraph()
    p_line.paragraph_format.space_after = Pt(12)
    p_line_border = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'<w:bottom w:val="single" w:sz="12" w:space="1" w:color="0C3461"/>'
        f'</w:pBdr>'
    )
    p_line._p.get_or_add_pPr().append(p_line_border)

    # -------------------------------------------------------------------------
    # 2. SENDER DETAILS & DATE
    # -------------------------------------------------------------------------
    add_p("Mirza Muhammad Muzzamil", bold=True, space_after=1)
    add_p("Corresponding Author & Lead Researcher", italic=True, space_after=1)
    add_p("Department of Computer and Information Systems Engineering (CISE)", space_after=1)
    add_p("NED University of Engineering and Technology, Karachi, Pakistan", space_after=1)
    add_p("Email: mirzamuzzamil@neduet.edu.pk | ORCID: 0000-0001-5258-8959", space_after=10)

    # Submission Date
    add_p("September 9, 2026", space_after=12)

    # -------------------------------------------------------------------------
    # 3. RECIPIENT EDITORIAL OFFICE
    # -------------------------------------------------------------------------
    add_p("To:", bold=True, space_after=1)

    if target_journal.upper() == "EMS":
        journal_name = "Environmental Modelling & Software"
        publisher = "Elsevier B.V."
        add_p("The Editor-in-Chief and Editorial Board", space_after=1)
        p_j = add_p()
        r_j = p_j.add_run(journal_name)
        r_j.bold = True
        r_j.italic = True
        add_p(f"{publisher}, Radarweg 29, 1043 NX Amsterdam, The Netherlands", space_after=12)
    else:
        journal_name = "Geosciences"
        publisher = "MDPI / Springer Nature"
        add_p("The Editorial Office", space_after=1)
        p_j = add_p()
        r_j = p_j.add_run("Geosciences Journal")
        r_j.bold = True
        r_j.italic = True
        add_p(f"{publisher}, St. Alban-Anlage 66, 4052 Basel, Switzerland", space_after=12)

    # -------------------------------------------------------------------------
    # 4. SUBJECT LINE
    # -------------------------------------------------------------------------
    p_subj = add_p("Subject: Submission of Original Research Manuscript — ", bold=True, space_after=12)
    r_subj = p_subj.add_run(f"“HydroGraphEM: A Self-Supervised Graph Learning Framework for Multi-Task Watershed Emulation of the Indus Delta”")
    r_subj.bold = True
    r_subj.italic = True

    # Salutation
    add_p("Dear Editor-in-Chief and Esteemed Editorial Board Members,", space_after=8)

    # -------------------------------------------------------------------------
    # 5. OPENING SUBMISSION STATEMENT
    # -------------------------------------------------------------------------
    p_open = add_p()
    p_open.add_run(
        "We are pleased to submit our original research manuscript entitled "
    )
    r_title = p_open.add_run(
        "“HydroGraphEM: A Self-Supervised Graph Learning Framework for Multi-Task Watershed Emulation of the Indus Delta”"
    )
    r_title.bold = True
    p_open.add_run(f" for peer-reviewed consideration as a Research Article in ")
    r_jn = p_open.add_run(journal_name)
    r_jn.italic = True
    r_jn.bold = True
    p_open.add_run(
        ". This research is authored solely by Mirza Muhammad Muzzamil."
    )

    # -------------------------------------------------------------------------
    # 6. MOTIVATION & PROBLEM STATEMENT
    # -------------------------------------------------------------------------
    p_prob = add_p()
    p_prob.add_run(
        "A pressing dilemma in modern hydroinformatics and watershed management is the severe computational "
        "trade-off between trusted, physics-based numerical simulators and data-driven emulators. Established "
        "hydrological modeling suites—such as the Soil and Water Assessment Tool (SWAT)—provide indispensable physical "
        "fidelity for water resource planning and policy decisions. However, high-resolution regional SWAT models incur prohibitive "
        "computational overhead and are fundamentally ill-suited for the continuous, real-time assimilation of dense Earth observation (EO) "
        "satellite constellations. Conversely, conventional deep learning emulators (e.g., standard LSTMs or feedforward neural networks) "
        "frequently suffer from spatial-temporal generalization collapse, omit conservation priors, and fail to preserve the physical "
        "hierarchical topology through which river networks route water, sediments, and nutrients."
    )

    # -------------------------------------------------------------------------
    # 7. CORE INNOVATIONS & CONTRIBUTIONS
    # -------------------------------------------------------------------------
    p_contrib_head = add_p("Key Scientific Contributions & Framework Highlights:", bold=True, space_after=4, space_before=4)

    contributions = [
        ("Hierarchical Multi-Level Routing Topology: ",
         "We formulate a directed, physical graph topology explicitly mapping 1,857 Hydrological Response Units (HRUs) "
         "to 483 subbasins and 483 reach segments. This architecture constrains message-passing exclusively along authentic "
         "fluvial flowpaths, ensuring physically coherent downstream state accumulation."),

        ("Multi-Modal Earth Observation and Simulation Fusion: ",
         "HydroGraphEM harmonizes 20 years of continuous physical simulations from SWAT with multi-spectral Earth observation "
         "products extracted via Google Earth Engine, including CHIRPS precipitation, MODIS evapotranspiration and soil moisture proxies, "
         "Landsat 8/9 NDVI and NDWI vegetation-water dynamics, and MERIT Hydro elevation derivatives."),

        ("Physics-Aware Self-Supervised Representation Pretraining: ",
         "We develop the Hierarchical Spatial-Temporal Positional Attention Graph Transformer (HST-PAGT). By formulating a self-supervised "
         "temporal mask reconstruction objective over standardized discharge, reach velocity, and storage components, the network learns "
         "mass-conservative routing representations directly from unlabeled continuous hydrological dynamics."),

        ("Rigorous Multi-Task Downstream Generalization: ",
         "Under strict chronological holdouts (testing on unseen 2019–2020 regimes), HydroGraphEM achieves high predictive power across "
         "multiple critical watershed tasks without task-specific recalibration: flood susceptibility emulation (R² = 0.6708, NSE = 0.669), "
         "stream Nitrate (NO3) concentration prediction (R² = 0.6850), and flood risk classification (Macro F1 = 0.7480). "
         "Under severe flood extremes (top 10% peak discharge), the emulator retains robust performance (R² = 0.6355), "
         "while spatial block holdouts across 105 unseen reaches yield strong correlation (Pearson r = 0.38–0.60)."),

        ("Interpretable Physical Process Alignment: ",
         "Comprehensive explainability analyses confirm that the latent representation space automatically clusters according to "
         "Strahler Stream Order hierarchy without explicit supervisory labels. Furthermore, temporal attention weights selectively "
         "concentrate around peak monsoon pulses (July–September), validating that internal attention routing mirrors the basin's "
         "dominant hydrodynamic drivers.")
    ]

    for title, desc in contributions:
        p_b = doc.add_paragraph()
        p_b.paragraph_format.left_indent = Inches(0.25)
        p_b.paragraph_format.space_after = Pt(3)
        r_bt = p_b.add_run("• " + title)
        r_bt.bold = True
        r_bt.font.name = 'Times New Roman'
        r_bd = p_b.add_run(desc)
        r_bd.font.name = 'Times New Roman'

    # -------------------------------------------------------------------------
    # 8. JOURNAL SCOPE ALIGNMENT
    # -------------------------------------------------------------------------
    p_scope = add_p()
    if target_journal.upper() == "EMS":
        p_scope.add_run(
            "Because this work directly targets the convergence of process-based environmental modeling (SWAT), "
            "advanced spatial-temporal machine learning, and cloud-scale Earth observation data engines to create open, reproducible "
            "software tools for water management, we believe it falls squarely within the central scope and readership of "
        )
        r_sc_j = p_scope.add_run("Environmental Modelling & Software")
        r_sc_j.italic = True
        p_scope.add_run(".")
    else:
        p_scope.add_run(
            "Because this work bridges geoscientific Earth system modeling, satellite remote sensing assimilation, and "
            "computational hydroinformatics to resolve hydrological extremes in climate-vulnerable alluvial deltas, "
            "it directly addresses the core thematic focus of "
        )
        r_sc_j = p_scope.add_run("Geosciences")
        r_sc_j.italic = True
        p_scope.add_run(".")

    # -------------------------------------------------------------------------
    # 9. AUTHOR DECLARATIONS & OPEN SCIENCE
    # -------------------------------------------------------------------------
    add_p("Author Statements and Declarations:", bold=True, space_after=4, space_before=4)

    declarations = [
        "Originality: This manuscript is an original work that has neither been published previously nor is currently under consideration for publication elsewhere in any format.",
        "Author Approval: As the sole author, I have thoroughly prepared, reviewed, and approved the submitted manuscript, and agree with its submission to the journal.",
        "Competing Interests: The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.",
        "Funding: This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.",
        "Open Code and Data Availability: To ensure rigorous reproducibility, the complete open-source Python codebase, graph topology builders, and evaluation pipelines are publicly available on GitHub at: https://github.com/mirzamuzzamilbaig/HydroGraphEM-A-Self-Supervised-Graph-Learning-Framework-."
    ]

    for dec in declarations:
        p_d = doc.add_paragraph()
        p_d.paragraph_format.left_indent = Inches(0.25)
        p_d.paragraph_format.space_after = Pt(3)
        parts = dec.split(":", 1)
        r_dt = p_d.add_run("• " + parts[0] + ":")
        r_dt.bold = True
        r_dt.font.name = 'Times New Roman'
        r_dd = p_d.add_run(parts[1])
        r_dd.font.name = 'Times New Roman'

    # -------------------------------------------------------------------------
    # 10. SUGGESTED PEER REVIEWERS
    # -------------------------------------------------------------------------
    add_p("Suggested Expert Peer Reviewers:", bold=True, space_after=4, space_before=4)

    reviewers = [
        ("Prof. Dr. Markus Reichstein", "Max Planck Institute for Biogeochemistry, Jena, Germany",
         "mreichstein@bgc-jena.mpg.de", "Leading authority on hybrid physics-informed machine learning and Earth system observation modeling."),
        ("Dr. Frederik Kratzert", "Google Research / Johannes Kepler University Linz, Austria",
         "kratzert@google.com", "Pioneering expert in deep learning architectures, LSTMs, and spatial hydroinformatics for streamflow prediction."),
        ("Prof. Chaopeng Shen", "Department of Civil and Environmental Engineering, Pennsylvania State University, USA",
         "cshen@psu.edu", "World-renowned specialist in differentiable hydrology, deep learning surrogates, and multi-scale hydrologic process learning."),
        ("Dr. Grey Nearing", "Google Research / University of California, Davis, USA",
         "gs-nearing@google.com", "Expert in machine learning hydrology, large-scale streamflow benchmarks, and ungauged basin prediction."),
        ("Prof. Dr. Vimal Mishra", "Indian Institute of Technology (IIT) Gandhinagar, India",
         "vmishra@iitgn.ac.in", "Leading expert in South Asian monsoon dynamics, extreme flood modeling, and satellite hydrological data fusion.")
    ]

    for name, affil, email, rationale in reviewers:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.left_indent = Inches(0.25)
        p_r.paragraph_format.space_after = Pt(3)
        r_rn = p_r.add_run("• " + name + " ")
        r_rn.bold = True
        r_rn.font.name = 'Times New Roman'
        r_ra = p_r.add_run(f"({affil}) — Email: {email}\n   Expertise: {rationale}")
        r_ra.font.name = 'Times New Roman'
        r_ra.font.size = Pt(10)

    # -------------------------------------------------------------------------
    # 11. CLOSING & SIGNATURE
    # -------------------------------------------------------------------------
    p_close = add_p(space_before=6, space_after=8)
    p_close.add_run(
        "Thank you very much for your time, consideration, and editorial stewardship. "
        "We look forward to receiving the reviewers' constructive feedback and guidance."
    )

    add_p("Respectfully submitted,", space_after=14)

    add_p("Mirza Muhammad Muzzamil", bold=True, space_after=1)
    add_p("Sole Author", italic=True, space_after=1)
    add_p("Department of Computer and Information Systems Engineering (CISE)", space_after=1)
    add_p("NED University of Engineering and Technology, Karachi 75270, Pakistan", space_after=1)
    add_p("Email: mirzamuzzamil@neduet.edu.pk | Phone: +92-334-2976696", space_after=1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)
    print(f"Successfully generated: {output_path}")


if __name__ == "__main__":
    # Generate EMS version
    build_cover_letter("EMS", "Cover_Letter_Environmental_Modelling_and_Software.docx")
    # Generate Geosciences version
    build_cover_letter("IEEE Access", "Cover_Letter_IEEE_Access.docx")
    build_cover_letter("Geosciences", "Cover_Letter_Geosciences.docx")
    # Generate default root Cover_Letter.docx
    build_cover_letter("EMS", "Cover_Letter.docx")
    print("All cover letter Word documents successfully created.")
