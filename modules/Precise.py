import os
import anndata as ad
import scanpy as sc
from boruta_analysis import BorutaAnalyzer
from reinforcement_learning import ReinforcementLearningAnalyzer
from shap_analysis import SHAPVisualizer
from xgboost_analysis import PredictionAnalyzer
from utils import validate_anndata, validate_response_column

class Precise:
    def __init__(self, adata, output_dir="../results", target_column="response", sample_column="sample", verbose = True):
        """
        Initialize the Analysis Manager with AnnData and configurations.

        Parameters:
            adata_path (str): Path to the AnnData file.
            output_dir (str): Directory to save results.
            target_column (str): Column in `adata.obs` containing the response labels.
            sample_column (str): Column in `adata.obs` containing sample identifiers.
        """
        # Load AnnData object
        self.adata = adata
        self.output_dir = output_dir
        self.target_column = target_column
        self.sample_column = sample_column
        self.verbose = verbose
        
        # Validate necessary columns
        validate_anndata(self.adata, [target_column, sample_column])
        validate_response_column(self.adata, target_column)
        
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize analyzers
        self.boruta_analyzer = BorutaAnalyzer(self.adata, output_dir=self.output_dir,verbose=verbose)
        self.rl_analyzer = ReinforcementLearningAnalyzer(self.adata, output_dir=self.output_dir, verbose=verbose)
        self.shap_visualizer = SHAPVisualizer(self.adata, output_dir=self.output_dir)
        self.loo_analyzer = PredictionAnalyzer(self.adata, target_column=target_column, 
                                                  sample_column=sample_column, results_folder=self.output_dir, verbose=verbose)

    def run_boruta(self):
        """
        Run the Boruta analysis.
        """
        return self.boruta_analyzer.run_boruta()

    def run_reinforcement_learning(self, chosen_features=None, n_iters=200, learning_rate=0.1):
        """
        Run reinforcement learning analysis.
        
        Parameters:
            n_iters (int): Number of RL iterations.
            learning_rate (float): Learning rate for RL updates.
            chosen_features (list): Features to include in RL analysis.
        """
        return self.rl_analyzer.run_reinforcement_learning(n_iters=n_iters, learning_rate=learning_rate, chosen_features=chosen_features)
    
    def get_rl_distribution(self, iter = None):
        return self.rl_analyzer.plot_reinforcement_labels_distribution(iter=iter)

    def run_shap_visualizations(self, top_k=20):
        """
        Run SHAP visualizations for the AnnData.
        
        Parameters:
            top_k (int): Number of top features to visualize.
        """
        self.shap_visualizer.shapely_score_barplot(top_k=top_k)
        self.shap_visualizer.shap_summary_plot(max_display=top_k)

    def run_loo_prediction(self):
        """
        Run Leave-One-Out (LOO) prediction analysis.
        """
        results, auc_score = self.loo_analyzer.run_loo_prediction()
        print(f"LOO Prediction Results: {results}")
        print(f"Overall ROC AUC Score: {auc_score}")
        return (results, auc_score)
    

    def create_feature_plots(self, k = 500):
        """
        Generate feature importance and intersection plots.
        """
        self.loo_analyzer.create_feature_plots(k)

    def choose_top_features(self, k):
        """
        Select top k intersecting features based on importance analysis.

        Parameters:
            k (int): Number of top intersecting features to select.
        """
        return self.loo_analyzer.choose_top_features(k)
    
    def subset_highly_variable(self, n_top_genes = 2000):
        sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes, inplace=True)
        self.adata = self.adata[:, self.adata.var['highly_variable']]
        # Initialize analyzers
        self.boruta_analyzer = BorutaAnalyzer(self.adata, output_dir=self.output_dir)
        self.shap_visualizer = SHAPVisualizer(self.adata, output_dir=self.output_dir)
        self.loo_analyzer = PredictionAnalyzer(self.adata, target_column=self.target_column, 
                                                  sample_column=self.sample_column, results_folder=self.output_dir)
        return self.adata
