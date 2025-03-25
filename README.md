
# Precise: Predictive Response Analysis from Single-Cell Expression

[![DOI](https://img.shields.io/badge/bioRxiv-10.1101%2F2024.11.16.623986v1-blue)](https://www.biorxiv.org/content/10.1101/2024.11.16.623986v1)

**Precise** is a Python-based computational framework for analyzing single-cell RNA sequencing (scRNA-seq) data to predict immune checkpoint inhibitor (ICI) responses. It integrates feature selection, explainable machine learning, and reinforcement learning to offer a reproducible and interpretable pipeline.

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

Precise builds upon methods described in our preprint on bioRxiv:  
🔗 [10.1101/2024.11.16.623986v1](https://www.biorxiv.org/content/10.1101/2024.11.16.623986v1)

It leverages single-cell data to:
- Select predictive genes using **Boruta**
- Visualize feature importance via **SHAP**
- Perform predictive modeling using **XGBoost**
- Apply **reinforcement learning** to refine cell-level labels

---

## Features

- ✅ Boruta feature selection
- ✅ SHAP-based interpretability
- ✅ LOO / k-fold cross-validation
- ✅ Reinforcement learning for per-cell scores
- ✅ Clean plots, modular code, and reproducible output

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

1. Prepare your AnnData object (filtered, normalized).
2. Run feature selection, prediction, SHAP analysis, and reinforcement learning using `Precise`.

### Key Modules

#### 1. **BorutaAnalyzer**
```python
from boruta_analysis import BorutaAnalyzer
analyzer = BorutaAnalyzer(adata)
subset_adata, chosen_features, _ = analyzer.run_boruta()
```

#### 2. **PredictionAnalyzer**
```python
from prediction_analysis import PredictionAnalyzer
analyzer = PredictionAnalyzer(adata, model, model_name="XGBoost", celltype=None)
results_df, auc_score, estimators = analyzer.cv_prediction()
```

#### 3. **SHAPVisualizer**
```python
from shap_analysis import SHAPVisualizer
vis = SHAPVisualizer(adata, model, model_name="XGBoost")
vis.shapely_score_barplot(top_k=20)
```

#### 4. **ReinforcementLearningAnalyzer**
```python
from reinforcement_learning import ReinforcementLearningAnalyzer
rl = ReinforcementLearningAnalyzer(adata, model, model_name="XGBoost")
refined_adata = rl.run_reinforcement_learning(chosen_features=chosen_features)
```

---

## Input Requirements

Your AnnData `.obs` must contain:
- `response`: Binary outcome (`0` = non-response, `1` = response)
- `sample`: Sample or patient ID

Ensure that data is log-normalized and contains highly variable genes.

---

## Output

- 📊 Feature importance CSVs
- 📈 SHAP plots and summary visualizations
- 📄 Cell- and sample-level predictions
- 📁 Annotated AnnData with added `.obs` fields:
  - `prediction`
  - `proba_prediction`
  - `RL_cell_scores`

Results are saved to `../results` or the directory you specify.

---

## Example

```python
import scanpy as sc
from Precise import Precise

adata = sc.read_h5ad("path_to_your_data.h5ad")

precise = Precise(adata, output_dir="./results", target_column="response", sample_column="sample")

# Step 1: Feature selection
_, chosen_features, _ = precise.run_boruta()

# Step 2: Reinforcement learning
refined_adata = precise.run_reinforcement_learning(chosen_features=chosen_features)

# Step 3: SHAP visualization
precise.run_shap_visualizations(top_k=20)

# Step 4: Prediction (LOO or k-fold)
results, auc_score = precise.run_loo_prediction()
print(f"LOO ROC AUC: {auc_score:.3f}")
```

---

## Citation

If you use **Precise** in your research, please cite our bioRxiv preprint:
> Pinhasi, Y., & Yizhak, K. (2024). **Precise: Predictive Response Analysis from Single-Cell Expression**. *bioRxiv*. https://doi.org/10.1101/2024.11.16.623986v1
