import copy
import os
import json
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import DPI, TITLE_FONT_SIZE, AXIS_LABELS_FONT_SIZE, AXIS_TICKS_FONT_SIZE, validate_anndata, validate_response_column



class ReinforcementLearningAnalyzer:
    def __init__(self, adata, model, model_name, output_dir="../results", plots_dir = 'plots', target_column="response", sample_column="sample", celltype=None, verbose=True):
        """
        Initialize the ReinforcementLearningAnalyzer.

        Parameters:
            adata (AnnData): Input AnnData object.
            output_dir (str): Directory to save results.
            verbose (bool): Whether to print progress.
        """
        self.adata = adata.copy()
        self.model = model
        self.model_name = model_name
        self.celltype = celltype
        self.output_dir = output_dir if not celltype else os.path.join(output_dir, celltype)
        self.plots_dir = os.path.join(self.output_dir, plots_dir)
        self.verbose = verbose
        self.target_column=target_column
        self.sample_column=sample_column
        

    def run_reinforcement_learning(
        self,
        chosen_features=["GZMH", "LGALS1", "GBP5", "HLA-DRB5", "CCR7", "IFI6", "GAPDH", "HLA-B", "EPSTI1", "CD38", "STAT1"],
        celltype=None,
        n_iters=200,
        learning_rate=0.1,
    ):
        """
        Perform reinforcement learning on the AnnData object to compute cell-level labels.
        """
        if not self.model.is_regressor:
            raise ValueError(f"Model {self.model_name} is not supported for RL analysis.")

        adata = self.adata

        if celltype is None and self.celltype is not None:
            celltype = self.celltype
        
        if celltype:
            if celltype not in adata.obs.columns:
                print(f"[WARNING] Cell type '{celltype}' not found in adata.obs columns. "
                    f"When using 'celltype' parameter, '{celltype}' must be a boolean or 0/1 column in obs.")
                print(f"[WARNING] Using all cells for RL analysis.")
            elif adata.obs[celltype].dtype == bool or set(adata.obs[celltype].unique()) <= {0, 1}:
                if self.verbose:
                    print(f"Filtering for cell type: {celltype}.")
                adata = adata[adata.obs[celltype].astype(int) == 1]
            else:
                print(f"[WARNING] Column '{celltype}' is not boolean or 0/1. Using all cells.")


        validate_anndata(self.adata, [self.target_column, self.sample_column])
        validate_response_column(self.adata, self.target_column)

        # Filter by chosen features
        adata = adata[:, chosen_features]
        
        if self.verbose:
            print(f"Starting RL iterations with {adata.shape[0]} cells and {adata.shape[1]} features.")
        rl_scores = 'RL_cell_scores'

        # Initialize biased labels
        labels = adata.obs[self.target_column].values
        adata.obs["labels"] = labels
        adata.obs[rl_scores] = np.where(labels == 0, -1, 1)
        pos_sum = adata.obs[rl_scores][adata.obs[rl_scores] > 0].sum()
        neg_sum = adata.obs[rl_scores][adata.obs[rl_scores] < 0].sum()

        # RL iterations
        sample_names = adata.obs[self.sample_column].unique()
        rl_labels = []

        for iteration in range(n_iters):
            if self.verbose:
                print(f"Iteration {iteration + 1}/{n_iters}")
            rl_labels.append(list(adata.obs[rl_scores].copy()))

            for sample in sample_names:
                train_set = adata[adata.obs[self.sample_column] != sample]
                test_set = adata[adata.obs[self.sample_column] == sample]

                X_train, y_train = train_set.to_df(), train_set.obs[rl_scores].values
                X_test = test_set.to_df()

                regressor = copy.deepcopy(self.model)
                regressor.fit(X_train, y_train)

                predictions = regressor.predict(X_test)
                sample_mask = adata.obs[self.sample_column] == sample
                updated_labels = np.where(
                    ((test_set.obs["labels"] == 1) == (predictions > 0)),
                    test_set.obs[rl_scores] + predictions * learning_rate,
                    test_set.obs[rl_scores],
                )
                adata.obs.loc[sample_mask, rl_scores] = updated_labels

            # Normalize labels
            pos_factor = adata.obs[rl_scores][adata.obs[rl_scores] > 0].sum() / pos_sum
            neg_factor = adata.obs[rl_scores][adata.obs[rl_scores] < 0].sum() / neg_sum
            adata.obs[rl_scores] = adata.obs[rl_scores].apply(
                lambda x: x / pos_factor if x > 0 else x / neg_factor
            )

        # Save results
        cluster_name = celltype.replace(' ', '_') if celltype else "all_cells"
        results_dir = self.output_dir
        os.makedirs(results_dir, exist_ok=True)

        adata_file = os.path.join(results_dir, f"{cluster_name}_rl_adata.h5ad")
        labels_file = os.path.join(results_dir, f"{cluster_name}_rl_labels.json")

        adata.write(adata_file)
        with open(labels_file, "w") as f:
            json.dump(rl_labels, f)

        if self.verbose:
            print(f"Results saved to {results_dir}")

        return adata

    def plot_reinforcement_labels_distribution(self, iter=None, log_scale=False, save_plots=True, celltype=None):
        """
        Plot the distribution of reinforcement learning predictions.
        """
        if celltype is None and self.celltype is not None:
            celltype = self.celltype
        cluster_name = celltype.replace(' ', '_') if celltype else "all_cells"
        labels_file = os.path.join(self.output_dir, f"{cluster_name}_rl_labels.json")

        # Load RL predictions from the JSON file
        with open(labels_file, "r") as f:
            all_rl_labels = json.load(f)

        # Determine which iteration to plot
        if not iter or iter >= len(all_rl_labels):
            iter = len(all_rl_labels) - 1  # Default to the last iteration

        rl_predictions = all_rl_labels[iter]

        # Plot the histogram
        plt.figure(figsize=(3.8, 3))
        plt.hist(rl_predictions, bins=100, color="#BAC6FF", edgecolor="black", linewidth=0.5)

        plt.xlabel("Predictions", fontsize=AXIS_LABELS_FONT_SIZE)
        plt.ylabel("Frequency", fontsize=AXIS_LABELS_FONT_SIZE)
        plt.title("Distribution of RL Predictions", fontsize=TITLE_FONT_SIZE)
        plt.tick_params(axis="both", which="major", labelsize=AXIS_TICKS_FONT_SIZE)

        if log_scale:
            plt.yscale("log")
            plt.ylabel("Frequency (log scale)", fontsize=AXIS_LABELS_FONT_SIZE)

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save or show the plot
        save_path = os.path.join(self.plots_dir, f"rl_labels_hist_{iter}.png")
        os.makedirs(os.path.dirname(self.plots_dir), exist_ok=True)

        if save_plots:
            plt.savefig(save_path, dpi=DPI)
            print(f"Plot saved at {save_path}")
            
        plt.show()
        plt.close()

        return rl_predictions