
# Precise: Predictive Response Analysis from Single-Cell Expression

[![DOI](https://img.shields.io/badge/Published%20in-npj%20Precision%20Oncology-blue)](https://www.nature.com/articles/s41698-025-00883-z)

**Precise** is a Python-based computational framework for analyzing single-cell RNA sequencing (scRNA-seq) data to predict immune checkpoint inhibitor (ICI) responses. It integrates advanced feature selection, machine learning, and reinforcement learning methodologies, offering a streamlined workflow for both exploratory data analysis and predictive modeling.

The framework is designed for researchers working in immunotherapy, oncology, and computational biology, facilitating insights into predictive biomarkers and mechanisms of response or non-response.

---

## Table of Contents
- [Background](#background)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Workflow Overview](#workflow-overview)
  - [Key Modules](#key-modules)
- [Input Requirements](#input-requirements)
- [Output](#output)
- [Example](#example)
- [Citation](#citation)


---

## Background
Precise builds upon methods described in our publication in *npj Precision Oncology*: [10.1038/s41698-025-00883-z](https://www.nature.com/articles/s41698-025-00883-z).
It leverages single-cell data to:
- Select features (genes) relevant to ICI response using **Boruta**.
- Visualize feature importance via **SHAP** analysis.
- Generate predictions using **XGBoost**.
- Apply **reinforcement learning** to refine cell-level labels and uncover patterns in response/non-response predictions.

<img src="https://github.com/user-attachments/assets/bb72a0df-5ee9-495c-99b9-e8e2ccc4bade" alt="EACR Poster Workflow" width="650" />

---

## Installation

### 🔧 Using Conda (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yizhak-lab-ccg/PRECISE-.git
cd PRECISE-
```

2. Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate precise_env
```

---

## Usage

### Workflow Overview
1. Prepare your scRNA-seq data as an [AnnData](https://anndata.readthedocs.io/en/latest/) before inputting it into Precise.
2. Run feature selection, modeling, and visualization using **Precise**.

### Key Modules
#### 1. **BorutaAnalyzer**
   - Performs feature selection using Boruta with an XGBoost classifier.
   - Outputs selected features for downstream analysis.

#### 2. **SHAPVisualizer**
   - Generates SHAP-based feature importance plots.

#### 3. **ReinforcementLearningAnalyzer**
   - Refines predictions iteratively using reinforcement learning.
   - Outputs updated cell-level scores.

#### 4. **PredictionAnalyzer**
   - Conducts predictive modeling and leave-one-out (LOO) cross-validation.
   - Outputs feature importance scores and prediction results.

---

## Input Requirements
The primary input is an `AnnData` object, with the following `.obs` columns:
- `response`: Binary labels indicating response (`1`) or non-response (`0`).
- `sample`: Unique identifiers for samples.
Optional:
- `celltype` (optional): A column name corresponding to a cell type (boolean or 1/0 values) used for cell-type–specific analyses.

---

## Output
Precise generates:
- **Feature Importance Scores**: Ranked genes relevant to ICI response.
- **SHAP Visualizations**: Bar plots, summary plots, and dependence plots.
- **Predictions**: Cell- and sample-level predictions in CSV format.
- **Annotated AnnData**: Updated object with new columns for cell scores.

All outputs are saved in the specified `output_dir` (default: `../results`).

---

## Example
```python
from Precise import Precise
import scanpy as sc

adata = sc.read_h5ad("path_to_your_data.h5ad")
precise = Precise(adata, output_dir="./results", target_column="response", sample_column="sample")

# Step 1: Boruta feature selection
_, chosen_features, _ = precise.run_boruta()

# Step 2: Reinforcement learning
refined_adata = precise.run_reinforcement_learning(chosen_features=chosen_features)

# Step 3: SHAP visualization
precise.run_shap_visualizations(top_k=20)

# Step 4: Leave-One-Out (LOO) prediction
results, auc_score = precise.cv_prediction()
print(f"LOO ROC AUC: {auc_score:.3f}")
```

---

## Citation
If you use **Precise** in your research, please cite our published article:

Pinhasi, Y., & Yizhak, K. (2025). Precise: Predictive Response Analysis from Single-Cell Expression. npj Precision Oncology, https://doi.org/10.1038/s41698-025-00883-z