import os
import shap
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ML_models import get_model
from xgb_estimator import XGBEstimator
from utils import validate_anndata, validate_response_column


class SHAPVisualizer:
    def __init__(self, adata, model, model_name, args = None, output_dir='../results', target_column="response", sample_column="sample", colormap = 'plasma'):
        """
        Initialize the SHAPVisualizer with data, model, and directories.

        Parameters:
            adata (AnnData): AnnData object containing the data.
            model: Trained model (e.g., XGBoost or RandomForest).
            output_dir (str): Directory to save plots.
        """
        self.adata = adata
        self.target_column = target_column
        self.sample_column = sample_column
        self.model_name = model_name
        self.model = model
        self.args = args
        validate_anndata(self.adata, [self.target_column, self.sample_column])
        validate_response_column(self.adata, self.target_column)
        
        self.df = adata.to_df()
        model.fit(self.df, adata.obs[target_column].values)
        self.output_dir = output_dir
        
        self.gene_names = list(self.df.columns)
        self.colormap = colormap
        os.makedirs(self.output_dir, exist_ok=True)

        if self.model_name in ['XGBClassifier', 'LightGBMClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier']:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(self.df)
            if self.model_name == 'LightGBMClassifier':
                self.shap_values = self.shap_values[1]
        elif 'Neural' in self.model_name:
            tensor_df = torch.tensor(self.df.values, dtype=torch.float32)
            self.explainer = shap.DeepExplainer(self.model.model, tensor_df)
            self.shap_values = self.explainer.shap_values(tensor_df)
        else:
            self.explainer = shap.Explainer(self.model, self.df)
            self.shap_values = self.explainer(self.df).values
            

    def shapely_score_barplot(self, top_k=20, title='Highest Shapely Score Genes', save_path=None):
        """
        Generate a bar plot for SHAP scores.
        """
        mean_df = self._create_mean_df()
        shapely_score_df = mean_df.reset_index().rename(columns={'index': 'Gene', 'mean_of_abs': 'Shapely score'})

        plt.figure(figsize=(14, 7))
        sns.barplot(
            x='Gene',
            y='Shapely score',
            palette="Blues_d",
            data=shapely_score_df.iloc[:top_k]
        )
        plt.title(title, fontsize=20)
        plt.yticks(fontsize=14)
        plt.xticks(rotation=30, ha='right', fontsize=14)
        plt.xlabel('Gene', fontsize=16)
        plt.ylabel('Shapely Score', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Bar plot saved to {save_path}")
            plt.close()
        else:
            plt.show()


    def shap_summary_plot(self, max_display=20, save_path=None):
        """
        Generate a SHAP summary plot with customized font sizes and colorbar.
        """
        shap.summary_plot(
            self.shap_values, features=self.df, feature_names=self.gene_names, max_display=max_display, cmap=self.colormap, show=False
        )

        fig = plt.gcf()
        ax = plt.gca()

        ax.set_xlabel(ax.get_xlabel() + ' Expression', fontsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)
        ax.set_title(ax.get_title(), fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)

        cbar = fig.axes[-1]
        cbar.set_ylabel('Gene Expression', fontsize=16)
        cbar.tick_params(labelsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Summary plot saved to {save_path}")
        else:
            plt.show()

    def shap_dependence_plot(self, gene1, gene2=None, save_path=None):
        """
        Generate a SHAP dependence plot.
        """
        cmap = plt.get_cmap(self.colormap) if isinstance(self.colormap, str) else self.colormap

        if gene2:
            shap.dependence_plot(
                self.gene_names.index(gene1),
                self.shap_values,
                self.df,
                feature_names=self.gene_names,
                cmap=cmap,
                interaction_index=gene2,
                show=False,
                dot_size=20
            )
        else:
            shap.dependence_plot(
                self.gene_names.index(gene1),
                self.shap_values,
                self.df,
                feature_names=self.gene_names,
                cmap=cmap,
                show=False,
                dot_size=20
            )

        fig = plt.gcf()
        ax = plt.gca()

        if gene2:
            ax.set_title(f'Interaction between {gene1} and {gene2}', fontsize=18)
        else:
            ax.set_title(f'Strongest Interaction with {gene1}', fontsize=18)
        ax.set_xlabel(ax.get_xlabel() + ' Expression', fontsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

        cbar = fig.axes[-1]
        cbar.set_ylabel(cbar.get_ylabel() + ' Expression', fontsize=16)
        cbar.tick_params(labelsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Dependence plot saved to {save_path}")
        plt.show()
        plt.close()


    def shap_waterfall_plot(self, sample, save_folder=None):
        """
        Create a SHAP waterfall plot for a specified sample and save or display it.
        """
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)

        sdata = self.adata[self.adata.obs[self.sample_column] != sample]
        temp = self.adata[self.adata.obs[self.sample_column] == sample]
        df = sdata.to_df()

        # Get and fit the model
        ce = get_model(self.model_name, self.args)
        ce.fit(df, sdata.obs[self.target_column])

        # Select SHAP Explainer
        if self.model_name in ['XGBoostClassifier', 'LightGBMClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier']:
            explainer = shap.TreeExplainer(ce.model, df)
        elif 'Neural' in self.model_name:
            explainer = shap.DeepExplainer(ce.model, df)
        else:
            explainer = shap.Explainer(ce.model, df)


        # Compute SHAP values
        exp = explainer(temp.to_df())

        if isinstance(exp, list):
            exp = exp[1]  # Take class 1 SHAP values
        
        aggr_exp = exp.mean(axis = 0)
        # Generate Waterfall Plot
        shap.plots.waterfall(aggr_exp, show=False)
        plt.tight_layout()

        if save_folder:
            plot_path = os.path.join(save_folder, f'{sample}_mean.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Waterfall plot saved to {plot_path}")
        else:
            plt.show()


    def _create_mean_df(self):
        """
        Create a DataFrame summarizing SHAP values.
        """
        shap_df = pd.DataFrame(self.shap_values, index=self.adata.obs_names, columns=self.adata.var_names)
        mean_df = pd.DataFrame(
            [
                shap_df.abs().mean().sort_values(ascending=False),
                shap_df.mean().abs().sort_values(ascending=False)
            ],
            index=['mean_of_abs', 'abs_of_mean']
        ).T
        mean_df['mean_of_abs_rank'] = mean_df['mean_of_abs'].rank(ascending=False).astype(int)
        mean_df['abs_of_mean_rank'] = mean_df['abs_of_mean'].rank(ascending=False).astype(int)
        mean_df['rank_dif'] = (mean_df['mean_of_abs_rank'] - mean_df['abs_of_mean_rank']).abs()

        total_genes = len(mean_df)
        mean_df['mean_of_abs_rank'] = total_genes - mean_df['mean_of_abs_rank']
        mean_df['abs_of_mean_rank'] = total_genes - mean_df['abs_of_mean_rank']

        return mean_df
