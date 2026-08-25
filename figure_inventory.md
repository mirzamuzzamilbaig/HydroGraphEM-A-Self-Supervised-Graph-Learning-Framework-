# HydroGraphFM: Locked Figure Inventory

This document tracks the 6 locked figures for the *Environmental Modelling & Software* manuscript, specifying their captions, purposes, status, and corresponding generated workspace assets.

---

### Table: Figure Inventory
| Figure | Status | Caption | Purpose | Core Contribution | Workspace Asset |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fig 1** | Outlined | HydroGraphFM Framework Architecture | Diagram of the end-to-end framework, unifying data fusion, pretraining, and downstream prediction | **Overview** | N/A (Visio/Draw.io schematic) |
| **Fig 2** | Outlined | Hierarchical HydroGraph Topology | Visualizing the topological mapping: 1857 HRUs $\to$ 483 Subbasins $\to$ 483 Reach Segments | **Contribution 1** | N/A (GIS / network schematic) |
| **Fig 3** | Outlined | Self-Supervised Pretraining Strategy | Schematic of temporal mask-reconstruction pretraining objectives and linear probing | **Contribution 3** | N/A (Flowchart schematic) |
| **Fig 4** | Complete | Generalization Performance Matrices | Visualizing chronological, spatial block, and extreme event $R^2$ scores | **Contribution 4** | [generalization_report_v1.md](file:///d:/all%20in%20one/Desktop/new_PhD_project/generalization_report_v1.md) data |
| **Fig 5** | Complete | Latent Space Projections & Strahler Order Manifolds | t-SNE projections of reach nodes colored by stream order, flood, sediment, and NO3, with PC1 boxplots | **Contribution 3 & 4** | `paper_figures/latent_projections.png` |
| **Fig 6** | Complete | Spatial-Temporal Attention & SHAP Importances | Grid plot combining temporal self-attention heatmaps, spatial routing coefficients, and permutation bar charts | **Interpretability** | `paper_figures/attention_patterns.png` & `paper_figures/feature_importance.png` |

---

### CSV Equivalent (`figure_inventory.csv`)
The CSV mapping is exported to the workspace directory as [figure_inventory.csv](file:///d:/all%20in%20one/Desktop/new_PhD_project/figure_inventory.csv) with the following contents:
```csv
Figure,Status,Caption,Purpose,Contribution,Workspace_Asset
Fig 1,Outlined,HydroGraphFM Framework Architecture,Diagram of the end-to-end framework,Overview,N/A
Fig 2,Outlined,Hierarchical HydroGraph Topology,Visualizing the topological mapping,Contribution 1,N/A
Fig 3,Outlined,Self-Supervised Pretraining Strategy,Schematic of pretraining reconstruction,Contribution 3,N/A
Fig 4,Complete,Generalization Performance Matrices,Visualizing downstream R2 scores,Contribution 4,generalization_report_v1.md
Fig 5,Complete,Latent Space Projections & Strahler Order Manifolds,Projections of reach nodes,Contribution 3 & 4,paper_figures/latent_projections.png
Fig 6,Complete,Spatial-Temporal Attention & SHAP Importances,Grid of heatmaps and bar charts,Interpretability,paper_figures/attention_patterns.png
```
