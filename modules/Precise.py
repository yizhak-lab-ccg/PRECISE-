import argparse
import copy
import os
import anndata as ad
import scanpy as sc
from boruta_analysis import BorutaAnalyzer
from reinforcement_learning import ReinforcementLearningAnalyzer
from shap_analysis import SHAPVisualizer
from prediction_analysis import PredictionAnalyzer
from ML_models import *
from utils import validate_anndata, validate_response_column

class Precise:
    def __init__(self, adata, model_name, output_dir="../results", target_column="response", sample_column="sample", celltype = None, args=None):
        """
        Initialize Precise with AnnData and configurations.
        """
        self.adata = adata
        self.output_dir = output_dir
        self.target_column = target_column
        self.sample_column = sample_column
        self.model_name = model_name
        self.celltype = celltype
        self.args = args or argparse.Namespace()
        
        defaults = {
            "max_depth": None,
            "learning_rate": None,
            "epochs": None,
            "seed": 42,
            "k_folds": None,
            "verbose": False,
            "weighted_prediction": False,
            'cell_type': None,
            "input_dim": self.adata.shape[1],  # Set input dimension dynamically
        }
        
        for key, default_value in defaults.items():
            setattr(self.args, key, getattr(self.args, key, default_value))

        validate_anndata(self.adata, [target_column, sample_column])
        validate_response_column(self.adata, target_column)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize models dynamically based on arguments
        self.args.input_dim = self.adata.shape[1]  # Set input dimension
        self.model = get_model(self.model_name, self.args)
        model = copy.deepcopy(self.model)
        self.shap_visualizer = SHAPVisualizer(self.adata, model.model if isinstance(model, SklearnModelWrapper) else model, self.model_name, self.args, output_dir=self.output_dir)
        self.prediction_analyzer = PredictionAnalyzer(
            self.adata,
            model=model,
            model_name=self.model_name,
            celltype=self.celltype,
            target_column=self.target_column,
            sample_column=self.sample_column,
            results_folder=self.output_dir,
            verbose=self.args.verbose
        )
        self.boruta_analyzer = BorutaAnalyzer(self.adata, model, model_name, self.celltype, output_dir=self.output_dir, verbose=self.args.verbose)
        self.rl_analyzer = ReinforcementLearningAnalyzer(self.adata, model, model_name, output_dir=self.output_dir, verbose=self.args.verbose)
    
    def cv_prediction(self, k_folds=None, seed = 42, verbose = None):
        """
        Perform K-Fold cross-validation or LOO prediction.
        """
        self.prediction_analyzer = PredictionAnalyzer(
            self.adata,
            model=self.model,
            model_name=self.model_name,
            celltype=self.celltype,
            target_column=self.target_column,
            sample_column=self.sample_column,
            results_folder=self.output_dir,
            verbose=self.args.verbose
        )
        return self.prediction_analyzer.cv_prediction(k_folds=k_folds, seed=seed,
                                               weighted_prediction=self.args.weighted_prediction,
                                               sample_column=self.sample_column, response_column=self.target_column,
                                               verbose=self.args.verbose)
    
    def run_boruta(self, verbose = False):
        """
        Run the Boruta analysis.
        """
        self.boruta_analyzer = BorutaAnalyzer(self.adata, output_dir=self.output_dir, verbose=verbose)
        return self.boruta_analyzer.run_boruta()

    def run_reinforcement_learning(self, model = None, model_name = None, chosen_features=None, n_iters=200, learning_rate=0.1, verbose = False):
        """
        Run reinforcement learning analysis.
        
        Parameters:
            n_iters (int): Number of RL iterations.
            learning_rate (float): Learning rate for RL updates.
            chosen_features (list): Features to include in RL analysis.
        """
        if not model:
            model = get_model(self.args.model_name, self.args, is_regressor=True)
        self.rl_analyzer = ReinforcementLearningAnalyzer(self.adata, model, self.args.model_name, output_dir=self.output_dir, verbose=verbose)
        return self.rl_analyzer.run_reinforcement_learning(n_iters=n_iters, learning_rate=learning_rate, chosen_features=chosen_features)
    
    def get_rl_distribution(self, iter = None):
        return self.rl_analyzer.plot_reinforcement_labels_distribution(iter=iter)

    def run_shap_visualizations(self, model = None, top_k=20):
        """
        Run SHAP visualizations for the AnnData.
        
        Parameters:
            top_k (int): Number of top features to visualize.
        """
        if not model:
            model = get_model(self.model_name, self.args, is_regressor=True)
        self.shap_visualizer = SHAPVisualizer(self.adata, model.model if isinstance(model, SklearnModelWrapper) else model, self.model_name, self.args, output_dir=self.output_dir)
        self.shap_visualizer.shapely_score_barplot(top_k=top_k)
        self.shap_visualizer.shap_summary_plot(max_display=top_k)

    def create_feature_plots(self, k = 500):
        """
        Generate feature importance and intersection plots.
        """
        self.prediction_analyzer.create_feature_plots(k)

    def choose_top_features(self, k):
        """
        Select top k intersecting features based on importance analysis.

        Parameters:
            k (int): Number of top intersecting features to select.
        """
        return self.prediction_analyzer.choose_top_features(k)
    
    def subset_highly_variable(self, n_top_genes = 2000):
        sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes, inplace=True)
        self.adata = self.adata[:, self.adata.var['highly_variable']]
        model = copy.deepcopy(self.model)
        self.shap_visualizer = SHAPVisualizer(self.adata, model.model if isinstance(model, SklearnModelWrapper) else model, self.model_name, self.args, output_dir=self.output_dir)
        self.prediction_analyzer = PredictionAnalyzer(
            self.adata,
            model=model,
            model_name=self.model_name,
            celltype=self.celltype,
            target_column=self.target_column,
            sample_column=self.sample_column,
            results_folder=self.output_dir,
            verbose=self.args.verbose
        )
        self.boruta_analyzer = BorutaAnalyzer(self.adata, model, self.model_name, self.celltype, output_dir=self.output_dir, verbose=self.args.verbose)
    
        return self.adata
    

def create_output_folder(base_folder, args):
    """
    Create a unique folder for the model input based on the parameters.

    Parameters:
    - base_folder: The base folder to create the input directory.
    - args: Arguments containing model parameters.

    Returns:
    - str: Path to the created folder.
    """
    folder_name_parts = [
        f"model_{args.model_name}",
        f"k_{args.k_folds}" if args.k_folds else "loo",
        f"seed_{args.seed}",
        f"weight_{args.weighted_prediction}",
    ]

    if args.max_depth:
        folder_name_parts.append(f"md_{args.max_depth}")
    if args.learning_rate:
        folder_name_parts.append(f"lr_{args.learning_rate}")
    if args.epochs:
        folder_name_parts.append(f"epoch_{args.epochs}")

    folder_name = "_".join(folder_name_parts)
    folder = os.path.join(base_folder, folder_name)
    os.makedirs(folder, exist_ok=True)
    return folder

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run Precise analysis pipeline")
    # parser.add_argument("--adata_path", type=str, required=True, help="Path to the AnnData file (H5AD format)")
    # parser.add_argument("--output_folder", type=str, default="./results", help="Path to the results folder")
    # parser.add_argument("--model_name", type=str, default="XGBoostClassifier", help="Model name to use")
    # parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the model")
    # parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for the model")
    # parser.add_argument("--epochs", type=int, default=None, help="Number of epochs for training neural networks")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # parser.add_argument("--k_folds", type=int, default=None, help="Number of folds for cross-validation")
    # parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    # parser.add_argument("--weighted_prediction", action="store_true", help="Enable weighted predictions")
    
    # args = parser.parse_args()
    
    # # Load AnnData
    # adata = sc.read_h5ad(args.adata_path)
    # output_folder = create_output_folder(args.output_folder, args)
    # args.output_folder = output_folder
    
    # # Create output folder
    # os.makedirs(args.output_folder, exist_ok=True)
    
    # # Initialize Precise and run CV prediction
    # precise = Precise(adata, model_name=args.model_name, output_dir=args.output_folder, target_column="response", sample_column="sample", args=args)
    # results, auc = precise.cv_prediction(model_name=args.model_name, k_folds=args.k_folds)
    # print(f"Finished cross-validation prediction. AUC: {auc:.4f}")

    adata = ad.read_h5ad("../examples/sample_melanoma_adata.h5ad")
    adata.obs['sample'] = adata.obs['sample_name']
    ## Continue with preprocessing adata...
    output_dir = "../results/"
    precise = Precise(
    adata,
    # model_name="XGBoostClassifier",
    model_name="NeuralNet_1",
    output_dir=output_dir,
    target_column="response",  # Adjust based on your dataset
    sample_column="sample"
    )
    
    # results_df, auc_score, estimators = precise.cv_prediction()
    # print(f"Finished cross-validation prediction. AUC: {auc_score:.4f}")
    precise.run_shap_visualizations()
    precise.shap_visualizer.shap_waterfall_plot('Pre_P3')