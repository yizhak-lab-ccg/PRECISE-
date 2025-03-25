
# Precise: Predictive Response Analysis from Single-Cell Expression

[![DOI](https://img.shields.io/badge/bioRxiv-10.1101%2F2024.11.16.623986v1-blue)](https://www.biorxiv.org/content/10.1101/2024.11.16.623986v1)

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
Precise builds upon methods described in our preprint on bioRxiv: [10.1101/2024.11.16.623986v1](https://www.biorxiv.org/content/10.1101/2024.11.16.623986v1). It leverages single-cell data to:
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
git clone https://github.com/asafpinhasitechnion/precise.git
cd precise
```

2. Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate precise_env
```

---

## Usage

### Workflow Overview
1. Preprocess your scRNA-seq data into an [AnnData](https://anndata.readthedocs.io/en/latest/) object.
2. Run feature selection, modeling, and visualization using **Precise**.

### Key Modules
#### 1. **BorutaAnalyzer**
   - Performs feature selection using Boruta with an XGBoost classifier.
   - Outputs selected features for downstream analysis.
   - Example:
     ```python
     from boruta_analysis import BorutaAnalyzer
     analyzer = BorutaAnalyzer(adata)
     subset_adata, chosen_features, _ = analyzer.run_boruta()
     ```

#### 2. **SHAPVisualizer**
   - Generates SHAP-based feature importance plots.
   - Example:
     ```python
     from shap_analysis import SHAPVisualizer
     shap_vis = SHAPVisualizer(adata)
     shap_vis.shapely_score_barplot(top_k=20)
     ```

#### 3. **ReinforcementLearningAnalyzer**
   - Refines predictions iteratively using reinforcement learning.
   - Outputs updated cell-level scores.
   - Example:
     ```python
     from reinforcement_learning import ReinforcementLearningAnalyzer
     rl_analyzer = ReinforcementLearningAnalyzer(adata)
     refined_adata = rl_analyzer.run_reinforcement_learning()
     ```

#### 4. **PredictionAnalyzer**
   - Conducts predictive modeling and leave-one-out (LOO) cross-validation.
   - Outputs feature importance scores and prediction results.
   - Example:
     ```python
     from xgboost_analysis import PredictionAnalyzer
     pred_analyzer = PredictionAnalyzer(adata)
     results_df, auc = pred_analyzer.run_loo_prediction()
     ```

---

## Input Requirements
The primary input is an `AnnData` object, with the following `.obs` columns:
- `response`: Binary labels indicating response (`1`) or non-response (`0`).
- `sample`: Unique identifiers for samples.

Ensure the data is preprocessed (e.g., log-normalized, variable features selected) before using Precise.

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

# Load your preprocessed AnnData object
adata = sc.read_h5ad("path_to_your_data.h5ad")

# Initialize the Precise framework
precise = Precise(adata, output_dir="./results", target_column="response", sample_column="sample")

# Run Boruta feature selection
_, chosen_features, _ = precise.run_boruta()

# Run reinforcement learning
refined_adata = precise.run_reinforcement_learning(chosen_features=chosen_features)

# Generate SHAP visualizations
precise.run_shap_visualizations(top_k=20)

# Perform LOO prediction
results, auc_score = precise.run_loo_prediction()
print(f"LOO ROC AUC: {auc_score}")
```

---

## Citation
If you use **Precise** in your research, please cite our bioRxiv preprint:
> Pinhasi, Y., & Yizhak, K. (2024). Precise: Predictive Response Analysis from Single-Cell Expression. bioRxiv. https://doi.org/10.1101/2024.11.16.623986v1
