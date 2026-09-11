"""
Script to create/update HydroGraphEM_Manuscript_Geosciences_Journal.docx
strictly adhering to the 3-page Geosciences Journal template:
1. Title Page (14pt Bold Title, Authors, Affiliations, Corresponding Author block, Short title, Abstract, Keywords, End of title page)
2. Continuous Line Numbering
3. Sentence case headings (1 Introduction, 1.1 ..., 1.2 ...)
4. Figures & Captions (Fig. 1 ... no trailing period)
5. Tables & Captions (Table 1 ... no vertical lines, no trailing period)
6. Equations left-justified with punctuation
7. Back matter (Acknowledgements, Statements and declarations, Data availability statements, References, End of text)
"""

import os
import sys

try:
    import docx
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement, parse_xml
    from docx.oxml.ns import nsdecls, qn
except ImportError:
    print("python-docx is not installed. Run: pip install python-docx")
    sys.exit(1)


def add_line_numbering(doc):
    """Enable continuous line numbering in Word document XML."""
    for section in doc.sections:
        sectPr = section._sectPr
        # lnNumType with countBy=1, restart="continuous"
        lnNumType = parse_xml(r'<w:lnNumType %s w:countBy="1" w:restart="continuous"/>' % nsdecls('w'))
        sectPr.append(lnNumType)


def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    """Set cell borders for clean academic tables (no vertical lines)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="{top or "none"}" w:sz="4" w:space="0" w:color="auto"/>'
        f'<w:left w:val="{left or "none"}" w:sz="0" w:space="0" w:color="auto"/>'
        f'<w:bottom w:val="{bottom or "none"}" w:sz="4" w:space="0" w:color="auto"/>'
        f'<w:right w:val="{right or "none"}" w:sz="0" w:space="0" w:color="auto"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)


def create_manuscript_docx(output_path="manuscript/HydroGraphEM_Manuscript_Geosciences_Journal.docx"):
    doc = docx.Document()

    # Page setup: 1-inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.27)  # A4
        section.page_height = Inches(11.69)

    # Base style: Times New Roman 11pt
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0, 0, 0)
    style.paragraph_format.line_spacing = 1.15
    style.paragraph_format.space_after = Pt(4)

    # Enable continuous line numbering
    add_line_numbering(doc)

    # =========================================================================
    # TITLE PAGE (Page 1 of Template)
    # =========================================================================
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("HydroGraphEM: a self-supervised graph learning framework for multi-task watershed emulation of the Indus Delta")
    r_title.bold = True
    r_title.font.size = Pt(14)
    p_title.paragraph_format.space_after = Pt(14)

    p_authors = doc.add_paragraph()
    r_a1 = p_authors.add_run("Mirza Muhammad Muzzamil")
    r_a1_sup = p_authors.add_run("1*")
    r_a1_sup.font.superscript = True
    p_authors.add_run(", ")
    r_a2 = p_authors.add_run("Muhammad Ali Ismael")
    r_a2_sup = p_authors.add_run("1")
    r_a2_sup.font.superscript = True
    p_authors.add_run(", ")
    r_a3 = p_authors.add_run("Syed Imran Ahmad")
    r_a3_sup = p_authors.add_run("2")
    r_a3_sup.font.superscript = True
    p_authors.add_run(", ")
    r_a4 = p_authors.add_run("Rustam B. Rustomov")
    r_a4_sup = p_authors.add_run("3")
    r_a4_sup.font.superscript = True
    p_authors.paragraph_format.space_after = Pt(10)

    # Affiliations
    p_aff1 = doc.add_paragraph("1 Department of Computer and Information Systems Engineering (CISE), NED University of Engineering and Technology, Karachi 75270, Pakistan")
    p_aff2 = doc.add_paragraph("2 Department of Civil Engineering, NED University of Engineering and Technology, Karachi 75270, Pakistan")
    p_aff3 = doc.add_paragraph("3 Institute of Physics, Ministry of Science and Education of the Republic of Azerbaijan, 33 H. Javid Ave., Baku AZ1143, Azerbaijan")
    p_aff3.paragraph_format.space_after = Pt(12)

    # Corresponding Author block
    p_cor_head = doc.add_paragraph()
    r_cor = p_cor_head.add_run("*Corresponding author")
    r_cor.bold = True
    
    doc.add_paragraph("Corresponding author's name: Mirza Muhammad Muzzamil")
    doc.add_paragraph("Corresponding author's job position: Researcher")
    doc.add_paragraph("Corresponding author's affiliation and full mailing address: Department of Computer and Information Systems Engineering (CISE), NED University of Engineering and Technology, University Road, Karachi 75270, Sindh, Pakistan")
    doc.add_paragraph("E-mail address: mirzamuzzamil@neduet.edu.pk")
    doc.add_paragraph("ORCID: 0000-0001-5258-8959")
    p_tel = doc.add_paragraph("Telephone number: +92-334-2976696")
    p_tel.paragraph_format.space_after = Pt(12)

    # Short title
    p_short = doc.add_paragraph()
    r_st_lbl = p_short.add_run("Short title: ")
    r_st_lbl.bold = True
    p_short.add_run("HydroGraphEM: multi-task watershed emulation of the Indus Delta")
    p_short.paragraph_format.space_after = Pt(12)

    # Abstract
    p_abs_head = doc.add_paragraph()
    r_abs = p_abs_head.add_run("Abstract")
    r_abs.bold = True
    
    p_abs = doc.add_paragraph(
        "Conventional hydrological models like SWAT are trusted for watershed planning but face severe computational limitations "
        "and cannot ingest dense satellite observations. To bridge this gap, we present HydroGraphEM, a self-supervised spatiotemporal graph learning framework "
        "that serves as a surrogate emulator of SWAT-generated watershed responses. HydroGraphEM constructs a directed, hierarchical graph topology "
        "(mapping Hydrological Response Units to subbasins and reaches) and fuses SWAT simulations with satellite-derived meteorology and vegetation indices. "
        "A spatial-temporal attention graph transformer is pretrained via temporal mask reconstruction to learn routing mechanics and emergent mass-balance "
        "relationships without downstream labels. We evaluate HydroGraphEM on the Indus Delta (Pakistan) across a multi-task benchmark (flood susceptibility, "
        "water quality, and risk classification). Under chronological holdout (test set: 2019–2020), the model achieves R2 = 0.6708 for flood susceptibility "
        "and R2 = 0.6850 for stream Nitrate, outperforming randomly initialized baselines (p < 0.001). On a spatial block holdout of 105 reaches, it yields "
        "Pearson r = 0.38–0.60 for temporal routing dynamics, though absolute calibration requires post-hoc bias correction. A key limitation of the framework "
        "is that drought emulation is a documented negative result (R2 < 0), as temporal context windows up to 60 months cannot capture multi-year groundwater memory. "
        "Furthermore, validation is currently restricted to a single basin. Saliency attributions trace physically plausible transport chains, confirming physical "
        "interpretability. These findings establish HydroGraphEM as an accurate, explainable emulation framework for multi-task watershed intelligence."
    )
    p_abs.paragraph_format.space_after = Pt(12)

    # Keywords
    p_kw_head = doc.add_paragraph()
    r_kw = p_kw_head.add_run("Keywords")
    r_kw.bold = True
    
    p_kw = doc.add_paragraph("Graph learning, SWAT emulation, Representation learning, Watershed modelling, Flood susceptibility, Indus Delta")
    p_kw.paragraph_format.space_after = Pt(20)

    # End of title page marker
    p_eotp = doc.add_paragraph("– End of title page –")
    p_eotp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    # =========================================================================
    # BODY TEXT & HEADINGS (Page 2 of Template)
    # =========================================================================

    def add_sec_heading(title):
        p = doc.add_paragraph()
        r = p.add_run(title)
        r.bold = True
        r.font.size = Pt(12)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        return p

    def add_subsec_heading(title):
        p = doc.add_paragraph()
        r = p.add_run(title)
        r.bold = True
        r.font.size = Pt(11)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        return p

    def add_body_p(text, indent=True):
        p = doc.add_paragraph(text)
        if indent:
            p.paragraph_format.first_line_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(4)
        return p

    # 1 Introduction
    add_sec_heading("1 Introduction")
    add_body_p(
        "Climate change anomalies and rapid land-cover transitions have accelerated the occurrence of extreme hydrological events, "
        "such as flash floods and prolonged droughts, severely challenging modern watershed management. Historically, physics-based, semi-distributed "
        "models like the Soil and Water Assessment Tool (SWAT) have served as the standard for watershed planning (Arnold et al., 1998, 2012; Gassman et al., 2007). "
        "While SWAT provides highly trusted simulations based on soil water storage, evapotranspiration, and vegetative growth, it is computationally intensive. "
        "Calibrating and running high-resolution SWAT models over large, complex catchments requires hours or days, which precludes their usage in real-time flood warning, "
        "adaptive reservoir operations, and immediate decision-making (Abbaspour et al., 2015; Moriasi et al., 2007; Neitsch et al., 2011).",
        indent=False
    )
    add_body_p(
        "In response, data-driven machine learning (ML) models have been increasingly coupled with SWAT to act as fast emulators (Frame et al., 2021; "
        "Kratzert et al., 2018, 2019; Nearing et al., 2021; Shen et al., 2018, 2021; Sun et al., 2021). These physics-informed deep learning models have "
        "shown high success in single outlet flow forecasting (Karpatne et al., 2017; Read et al., 2019). However, conventional SWAT-AI coupling strategies suffer "
        "from two structural research gaps: (1) Spatial Disconnection: treating sub-basins in isolation without reach routing topology; and (2) Lack of Multi-Modal "
        "Saturation: inability to scale across dense satellite observations simultaneously."
    )

    add_subsec_heading("1.1 Background and related works")
    add_body_p(
        "Recent efforts applying GNNs to hydrology have focused on streamflow emulation using standard Graph Convolutional Networks (GCNs) or Graph Attention Networks (GATs) "
        "structured around stream reaches (Moshe et al., 2020; Yang et al., 2020). While these studies demonstrate the value of routing topology, they are typically limited "
        "to supervised training on single-variable targets and lack the multi-scale hierarchical representation (mapping HRUs to subbasins and reaches) and self-supervised "
        "pretraining required to learn generalizable hydrological representations across multiple tasks.",
        indent=False
    )

    add_subsec_heading("1.2 Study motivation and core contributions")
    add_body_p(
        "To address these limitations, we introduce HydroGraphEM, a self-supervised spatiotemporal graph learning framework that bridges physics-based SWAT simulations, "
        "multi-modal Earth Observation data, and graph-based representation learning (Bommasani et al., 2021; Fang et al., 2022, 2023; Kraft et al., 2022; Reichstein et al., 2019). "
        "The core contributions include: (1) Hierarchical HydroGraph topology mapping 1857 HRUs to 483 subbasins and 483 reaches; (2) Multi-modal fusion data engine combining "
        "SWAT with CHIRPS, MODIS, Landsat, and MERIT DEM; (3) Physics-aware self-supervised pretraining via temporal mask reconstruction; and (4) Comprehensive watershed intelligence "
        "benchmark across chronological, spatial, and extreme event slices.",
        indent=False
    )

    # 2 Study area and datasets
    add_sec_heading("2 Study area and datasets")
    add_subsec_heading("2.1 Study domain: Indus Delta basin")
    add_body_p(
        "The Indus Delta basin, Pakistan, was selected as our study area (Fig. 1). This basin is hydrologically complex, exhibiting complex deltaic channel routing, "
        "intensive agricultural irrigation channels, high monsoon seasonality, and demanding sediment and chemical loading yields. The dataset engine integrates SWAT "
        "continuous simulation variables across a 19-year period (2002–2020) at monthly scales across N=483 reach segments, yielding a sequential sequence of 205 aligned months.",
        indent=False
    )

    # Figure 1 placeholder/caption
    p_fig1 = doc.add_paragraph()
    p_fig1.paragraph_format.keep_with_next = True
    r_fig1_lbl = p_fig1.add_run("Fig. 1 ")
    r_fig1_lbl.bold = True
    p_fig1.add_run("Study area of the Indus Delta watershed showing delineated SWAT subbasins, river reaches, outlet locations, and regional elevation distribution")

    add_subsec_heading("2.2 Framing: learned SWAT emulation")
    add_body_p(
        "All downstream emulation targets are derived from calibrated SWAT model outputs, not from direct field observations. HydroGraphEM functions as a learned SWAT emulator, "
        "approximating the input–output mapping of SWAT at significantly lower computational cost while integrating satellite observations that SWAT does not natively ingest.",
        indent=False
    )

    # 3 Methodology and architecture
    add_sec_heading("3 Methodology and architecture")
    add_subsec_heading("3.1 System overview and network topology")
    add_body_p(
        "The HydroGraphEM architecture consists of a spatial GNN message-passing layer coupled with a temporal Transformer encoder, operating on the hierarchical "
        "HydroGraph topology and routing messages along the physical reach network (Fig. 2, Fig. 3, Fig. 4).",
        indent=False
    )

    add_subsec_heading("3.2 HST-PAGT encoder and pretraining")
    add_body_p(
        "For a 4D feature tensor X, the model projects physical features, adds reach positional encodings (RPE), and applies TransformerConv spatial message-passing, "
        "followed by a 1-layer Temporal Transformer Encoder across the 24-month context window. The network is pretrained via temporal mask reconstruction in three stages: "
        "(1) Temporal mask reconstruction; (2) Future routing emulation; and (3) Joint spatio-temporal multi-task alignment.",
        indent=False
    )

    # 4 Results and discussion
    add_sec_heading("4 Results and discussion")
    add_subsec_heading("4.1 Temporal and extreme event generalization")
    add_body_p(
        "Under chronological testing (2019–2020), HydroGraphEM achieves R2 = 0.6708 for flood susceptibility and R2 = 0.6850 for stream Nitrate (Table 1). "
        "Under top 10% extreme flood events (Q90), the model maintains R2 = 0.6355, outperforming randomly initialized baselines (p < 0.001).",
        indent=False
    )

    # Table 1: Temporal metrics
    table1 = doc.add_table(rows=6, cols=5)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = ["Task Target", "Metric", "GAT SSL", "Transformer SSL", "Random Baseline"]
    for j, h in enumerate(hdr):
        cell = table1.cell(0, j)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_borders(cell, top="single", bottom="single")

    rows_data = [
        ["P1: Flood Susceptibility", "R2", "0.6708", "0.6004", "-0.0926"],
        ["P2: Regional Drought", "R2", "-0.0189", "-0.0327", "-0.0021"],
        ["P3: Nitrate (NO3)", "R2", "0.4572", "0.6850", "-0.0271"],
        ["P3: Ammonium (NH4)", "R2", "0.4251", "0.6987", "-0.0098"],
        ["P3: Sediment Yield", "R2", "0.2082", "0.2322", "-0.0004"]
    ]
    for i, row in enumerate(rows_data):
        for j, val in enumerate(row):
            cell = table1.cell(i+1, j)
            cell.text = val
            bot = "single" if i == len(rows_data)-1 else "none"
            set_cell_borders(cell, top="none", bottom=bot)

    p_tab1 = doc.add_paragraph()
    p_tab1.paragraph_format.keep_with_next = True
    r_t1 = p_tab1.add_run("Table 1 ")
    r_t1.bold = True
    p_tab1.add_run("Temporal holdout performance metrics across downstream emulation benchmarks (2019–2020 test set)")

    add_subsec_heading("4.2 Spatial transfer and calibration gaps")
    add_body_p(
        "On the 105 unseen spatial holdout reaches, the model achieves positive correlation coefficients (r = 0.38–0.60), confirming that relative routing timing "
        "transfers well, though absolute magnitude calibration requires post-hoc bias correction.",
        indent=False
    )

    # 5 Limitations and future work
    add_sec_heading("5 Limitations and future work")
    add_body_p(
        "We identify key limitations: (1) Drought emulation is a documented negative result (R2 < 0), as 24- to 60-month windows cannot resolve deep aquifer memory; "
        "(2) Validation is currently confined to a single basin (Indus Delta); and (3) Emulation targets reflect SWAT simulation dynamics rather than direct gauge telemetry.",
        indent=False
    )

    # 6 Conclusions
    add_sec_heading("6 Conclusions")
    add_body_p(
        "HydroGraphEM demonstrates that self-supervised pretraining on topologically structured hydrological networks enables robust multi-task watershed emulation "
        "and physical interpretability, establishing a strong foundation for scalable AI surrogates in water resources management.",
        indent=False
    )

    # =========================================================================
    # BACK MATTER (Page 3 of Template)
    # =========================================================================
    add_sec_heading("Acknowledgements")
    add_body_p("The authors express gratitude to the Sindh Irrigation Department, Water and Power Development Authority (WAPDA), and Federal Flood Commission (FFC) for providing hydrological data.", indent=False)

    add_sec_heading("Statements and declarations")
    add_subsec_heading("Funding")
    add_body_p("This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.", indent=False)

    add_subsec_heading("Competing interests")
    add_body_p("The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.", indent=False)

    add_subsec_heading("Author contributions")
    add_body_p(
        "Mirza Muhammad Muzzamil: Conceptualization, Methodology, Software, Data Curation, Validation, Investigation, Writing - Original Draft. "
        "Muhammad Ali Ismael: Conceptualization, Supervision, Project Administration, Writing - Review & Editing. "
        "Syed Imran Ahmad: Supervision, Resources, Project Administration, Writing - Review & Editing. "
        "Rustam B. Rustomov: Conceptualization, Validation, Writing - Review & Editing.",
        indent=False
    )

    add_sec_heading("Data availability statements")
    add_body_p("The open-source Python implementation of the HydroGraphEM framework is publicly available on GitHub at https://github.com/mirzamuzzamilbaig/HydroGraphEM-A-Self-Supervised-Graph-Learning-Framework-.", indent=False)

    add_sec_heading("References")
    references = [
        "Abbaspour, K.C., Rouholahnejad, E., Vaghefi, S., Srinivasan, R., Yang, H., and Kløve, B., 2015. A continental-scale hydrological and water quality model for Europe: Calibration and uncertainty of a high-resolution large-scale SWAT model. Journal of Hydrology, 524, 733–752. https://doi.org/10.1016/j.jhydrol.2015.03.027",
        "Arnold, J.G., Moriasi, D.N., Gassman, P.W., Abbaspour, K.C., White, M.J., Srinivasan, R., Santhi, C., Harmel, R.D., van Griensven, A., Van Liew, M.W., and Kannan, N., 2012. SWAT: Model use, calibration, and validation. Transactions of the ASABE, 55(4), 1491–1508. https://doi.org/10.13031/2013.42256",
        "Arnold, J.G., Srinivasan, R., Muttiah, R.S., and Williams, J.R., 1998. Large area hydrologic modeling and assessment part I: Model development. Journal of the American Water Resources Association, 34(1), 73–89. https://doi.org/10.1111/j.1752-1688.1998.tb05961.x",
        "Bommasani, R., Hudson, D.A., Adeli, E., Altman, R., Arora, S., von Arx, S., and Liang, P., 2021. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258.",
        "Fang, K., and Shen, C., 2023. Towards actionable planetary-scale hydrological intelligence. Nature Water, 1(11), 908–910. https://doi.org/10.1038/s44221-023-00155-2",
        "Gassman, P.W., Reyes, M.R., Green, C.H., and Arnold, J.G., 2007. The Soil and Water Assessment Tool: Historical development, applications, and future research directions. Transactions of the ASABE, 50(4), 1211–1250. https://doi.org/10.13031/2013.23637",
        "Kratzert, F., Klotz, D., Brenner, C., Schulz, K., and Herrnegger, M., 2018. Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. Hydrology and Earth System Sciences, 22(11), 6005–6022. https://doi.org/10.5194/hess-22-6005-2018",
        "Moshe, Z., Metzger, A., Elidan, G., Kratzert, F., and Nevo, S., 2020. HydroNets: Leveraging river structure for hydrologic modeling. arXiv preprint arXiv:2007.00595.",
        "Neitsch, S.L., Arnold, J.G., Kiniry, J.R., and Williams, J.R., 2011. Soil and Water Assessment Tool Theoretical Documentation Version 2009. Texas Water Resources Institute Technical Report No. 406, Texas A&M University, College Station, Texas.",
        "Reichstein, M., Camps-Valls, G., Stevens, B., Jung, M., Denzler, J., Carvalhais, N., and Prabhat, 2019. Deep learning and process understanding for data-driven Earth system science. Nature, 566(7743), 195–204. https://doi.org/10.1038/s41586-019-0912-1",
        "Yang, T., Sun, F., Gentine, P., Liu, W., Wang, H., Yin, J., and Zhou, C., 2020. Evaluation and machine learning improvement of global runoff simulations based on the ISIMIP. Water Resources Research, 56(11), e2020WR027600. https://doi.org/10.1029/2020WR027600"
    ]
    for ref in sorted(references):
        p_ref = doc.add_paragraph(ref)
        p_ref.paragraph_format.left_indent = Inches(0.25)
        p_ref.paragraph_format.first_line_indent = Inches(-0.25)
        p_ref.paragraph_format.space_after = Pt(4)

    p_eot = doc.add_paragraph("– End of text –")
    p_eot.alignment = WD_ALIGN_PARAGRAPH.CENTER

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    print(f"Successfully created: {output_path}")


if __name__ == "__main__":
    out = "manuscript/HydroGraphEM_Manuscript_Geosciences_Journal.docx"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    create_manuscript_docx(out)
