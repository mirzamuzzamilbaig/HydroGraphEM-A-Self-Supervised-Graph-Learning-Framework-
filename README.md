# HydroGraphEM: A Self-Supervised Graph Learning Framework for Multi-Task Watershed Emulation of the Indus Delta

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

Official implementation of **HydroGraphEM**, a self-supervised spatiotemporal graph learning framework designed as a surrogate emulator for physics-based Soil and Water Assessment Tool (SWAT) simulations coupled with multi-modal Earth Observation (EO) remote sensing data.

---

## 📖 Abstract

Conventional hydrological models like SWAT are trusted for watershed planning but face severe computational limitations and cannot ingest dense satellite observations. To bridge this gap, we present **HydroGraphEM**, a self-supervised spatiotemporal graph learning framework that serves as a surrogate emulator of SWAT-generated watershed responses. HydroGraphEM constructs a directed, hierarchical graph topology (mapping Hydrological Response Units to subbasins and reaches) and fuses SWAT simulations with satellite-derived meteorology and vegetation indices. A spatial-temporal attention graph transformer is pretrained via temporal mask reconstruction to learn routing mechanics and emergent mass-balance relationships without downstream labels. We evaluate HydroGraphEM on the Indus Delta (Pakistan) across a multi-task benchmark (flood susceptibility, water quality, and risk classification).

---

## 🌟 Key Methodological Highlights

1. **Three-Level Hierarchical HydroGraph Topology**: Captures watershed flow paths from Hydrological Response Units (HRUs) $\rightarrow$ Subbasins $\rightarrow$ River Reaches, respecting strict physical drainage divides.
2. **Multi-Modal Earth Observation Fusion**: Integrates Google Earth Engine (GEE) satellite products (NDVI, NDWI, precipitation, temperature) with SWAT physical parameters across hundreds of catchments.
3. **Self-Supervised Masked Representation Learning**: Pretrained via temporal mask reconstruction over extended time series to learn fundamental downstream mass transport and reach routing dynamics.
4. **Multi-Task Watershed Intelligence**: Emulates flood susceptibility, stream water quality ($NO_3^-$, sediment load), and hydrological risk classification.

---

## 📁 Repository Structure

```
HydroGraphEM/
├── prism/                         # LaTeX Source Files for Journal Manuscript
│   ├── main.tex                   # Principal manuscript text
│   ├── references.bib             # Complete BibTeX citations
│   ├── highlights.tex             # Journal highlights
│   └── *.png                      # High-resolution manuscript figures
├── paper_figures/                 # Generated experimental visualisations & SHAP figures
├── feature_dictionary_hrus.csv    # Feature definitions for HRUs
├── feature_dictionary_reaches.csv # Feature definitions for River Reaches
├── feature_dictionary_subbasins.csv# Feature definitions for Subbasins
├── master_results.csv             # Benchmark evaluation metrics across tasks
├── ablation_summary.csv           # Ablation study performance comparison
├── pretraining_metrics.csv        # Pretraining convergence metrics
│
├── extract_gee.py                 # Google Earth Engine data extraction pipeline
├── parse_swat.py                  # SWAT simulation output parser & graph generator
├── gen_network_topology.py        # Watershed adjacency & routing matrix construction
├── align_datasets.py              # Spatiotemporal dataset alignment & feature scaling
├── scalers.py                     # Custom physical parameter normalization utilities
│
├── build_pretrain.py              # Self-supervised temporal mask dataset preparation
├── train_pretrain.py              # Spatial-Temporal Attention Graph Transformer pretraining
│
├── benchmark_finetuning.py        # Fine-tuning downstream evaluation script
├── benchmark_drought_t60.py       # Extended context window drought emulation benchmark
├── benchmark_generalization.py    # Spatial block holdout & temporal generalization tests
├── benchmark_mcdropout.py         # Monte Carlo Dropout uncertainty estimation
│
├── attention_analysis.py          # Spatial-temporal attention weight visualization
├── embedding_analysis.py          # Latent space representations & CKA similarity
├── shap_analysis.py               # SHAP feature importance & physical interpretability
├── generate_custom_figures.py     # Publication-grade chart generation
├── LICENSE                        # MIT License
└── README.md                      # Repository documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric (PyG)
- earthengine-api

### Installation

```bash
git clone https://github.com/mirzamuzzamil/HydroGraphEM-A-Self-Supervised-Graph-Learning-Framework-.git
cd HydroGraphEM-A-Self-Supervised-Graph-Learning-Framework-
pip install -r requirements.txt
```

---

## ⚙️ Workflow Execution Pipeline

### 1. Data Processing & Topology Generation
```bash
# Extract Earth Observation data from GEE
python extract_gee.py

# Parse SWAT simulation outputs and build watershed graph
python parse_swat.py

# Construct network routing topology
python gen_network_topology.py

# Align temporal resolution and normalize features
python align_datasets.py
```

### 2. Self-Supervised Pretraining
```bash
# Build masked temporal sequences
python build_pretrain.py

# Pretrain spatial-temporal graph transformer
python train_pretrain.py
```

### 3. Downstream Benchmarking & Evaluation
```bash
# Evaluate multi-task fine-tuning performance
python benchmark_finetuning.py

# Run spatial holdout & generalization benchmarks
python benchmark_generalization.py

# Estimate prediction uncertainty via MC Dropout
python benchmark_mcdropout.py
```

### 4. Interpretability & Visualization
```bash
# Generate attention heatmap analysis
python attention_analysis.py

# Generate SHAP feature attributions
python shap_analysis.py

# Render manuscript visualisations
python generate_custom_figures.py
```

---

## 📊 Benchmark Results

Under chronological holdout (Test Period: 2019–2020) on the Indus Delta watershed benchmark:

| Emulation Target | Evaluation Metric | HydroGraphEM (Ours) | Baseline (Random Init) |
| :--- | :---: | :---: | :---: |
| **Flood Susceptibility** | $R^2$ | **0.6708** | 0.4120 |
| **Stream Nitrate ($NO_3^-$)** | $R^2$ | **0.6850** | 0.3890 |
| **Reach Discharge Routing** | Pearson $r$ | **0.38 – 0.60** | 0.18 – 0.32 |

---

## 📝 Citation & Contact

If you use HydroGraphEM in your research, please cite our paper:

```bibtex
@article{muzzamil2026hydrographem,
  title={HydroGraphEM: A Self-Supervised Graph Learning Framework for Multi-Task Watershed Emulation of the Indus Delta},
  author={Muzzamil, Mirza Muhammad },
  year={2026}
}
```

For questions or collaborations, please contact **Mirza Muhammad Muzzamil** at `mirzamuzzamil@neduet.edu.pk`.

---
## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
