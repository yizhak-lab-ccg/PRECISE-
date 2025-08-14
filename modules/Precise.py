import argparse
import copy
import os
import anndata as ad
import numpy as np
import scanpy as sc
from boruta_analysis import BorutaAnalyzer
from reinforcement_learning import ReinforcementLearningAnalyzer
from shap_analysis import SHAPVisualizer
from prediction_analysis import PredictionAnalyzer
from ML_models import *
from utils import validate_anndata, validate_response_column
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

class Precise:
    def __init__(self, adata, model_name, output_dir="../results", target_column="response", sample_column="sample", celltype = None, weighted_prediction = False, args=None):
        """
        Initialize Precise with AnnData and configurations.
        """
        self.adata = adata
        self.output_dir = output_dir
        self.target_column = target_column
        self.sample_column = sample_column
        self.model_name = model_name
        self.celltype = celltype
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        self.args = args or argparse.Namespace()
        self.weighted_prediction = weighted_prediction
        defaults = {
            "max_depth": None,
            "learning_rate": None,
            "epochs": None,
            "seed": 42,
            "k_folds": None,
            "verbose": False,
            "weighted_prediction": weighted_prediction,
            'celltype': None,
            "input_dim": self.adata.shape[1],  # Set input dimension dynamically
            'use_gpu': False,
            'model_name': model_name
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

        self.boruta_analyzer = BorutaAnalyzer(self.adata,
            model=self.model,
            model_name=self.model_name,
            celltype=self.celltype,
            response_col=self.target_column,
            output_dir=self.output_dir,
            sample_column=self.sample_column,
            verbose=self.args.verbose)
        # self.boruta_analyzer = BorutaAnalyzer(self.adata, output_dir=self.output_dir, verbose=verbose)
        return self.boruta_analyzer.run_boruta()

    def run_reinforcement_learning(self, model_name = None, chosen_features=None, n_iters=200, learning_rate=0.1, verbose = False):
        """
        Run reinforcement learning analysis.
        
        Parameters:
            n_iters (int): Number of RL iterations.
            learning_rate (float): Learning rate for RL updates.
            chosen_features (list): Features to include in RL analysis.
        """
        model = get_model(self.args.model_name, self.args, is_regressor=True)
        self.rl_analyzer = ReinforcementLearningAnalyzer(self.adata, model, self.args.model_name.replace('Classifier', ''), output_dir=self.output_dir, verbose=verbose)
        return self.rl_analyzer.run_reinforcement_learning(n_iters=n_iters, learning_rate=learning_rate, chosen_features=chosen_features)
    
    def get_rl_distribution(self, iter = None):
        return self.rl_analyzer.plot_reinforcement_labels_distribution(iter=iter)

    def run_shap_visualizations(self, model = None, top_k=20, save_path = None):
        """
        Run SHAP visualizations for the AnnData.
        
        Parameters:
            top_k (int): Number of top features to visualize.
        """
        if not model:
            model = get_model(self.model_name, self.args, is_regressor=True)
        self.shap_visualizer = SHAPVisualizer(self.adata, model.model if isinstance(model, SklearnModelWrapper) else model, self.model_name, self.args, output_dir=self.output_dir)
        self.shap_visualizer.shapely_score_barplot(top_k=top_k, save_path = os.path.join(save_path, 'shap_barplot.png') if save_path else None)
        self.shap_visualizer.shap_summary_plot(max_display=top_k, save_path=os.path.join(save_path, 'summary_plot.png') if save_path else None)

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


    def prune_decision_tree(self, max_depth=4, min_samples_leaf=20, random_state=42, save_path = None):
        """
        Train and prune a DecisionTreeClassifier on AnnData.

        Parameters:
        -----------
        max_depth : int, optional
            Maximum depth of the decision tree (default: 4).
        min_samples_leaf : int, optional
            Minimum samples required at a leaf node (default: 20).
        random_state : int, optional
            Random seed for reproducibility.

        Returns:
        --------
        pruned_tree : DecisionTreeClassifier
            The trained and pruned decision tree.
        """

        # Convert AnnData to DataFrame for sklearn compatibility
        df = self.adata.to_df()
        labels = self.adata.obs[self.target_column]

        # Train decision tree
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
        trained_tree = tree.fit(df, labels)

        # Make a deep copy for pruning
        pruned_tree = copy.deepcopy(trained_tree)

        def prune_tree(clf, node_index=0):
            """
            Recursively prune the decision tree by removing nodes that do not contribute to meaningful splits.

            Parameters:
            -----------
            clf : DecisionTreeClassifier
                A trained sklearn DecisionTreeClassifier.
            node_index : int, optional
                Index of the current node in the tree (default: 0, root node).
            """
            tree = clf.tree_

            def is_leaf(node):
                """Check if a node is a leaf."""
                return tree.children_left[node] == tree.children_right[node] == -1

            def recursive_prune(node):
                """Recursively prune the tree and return ('pure', class) or ('mixed', None)."""
                if is_leaf(node):
                    return ("pure", np.argmax(tree.value[node]))  # Return class of leaf

                left_child = tree.children_left[node]
                right_child = tree.children_right[node]

                # Recursively prune left and right subtrees
                left_result = recursive_prune(left_child) if left_child != -1 else ("pure", None)
                right_result = recursive_prune(right_child) if right_child != -1 else ("pure", None)

                # If any side is 'mixed', we cannot prune this node
                if left_result[0] == "mixed" or right_result[0] == "mixed":
                    return ("mixed", None)

                # If both children predict the same class, prune this node
                if left_result[1] == right_result[1]:
                    tree.children_left[node] = -1
                    tree.children_right[node] = -1
                    tree.value[node] = tree.value[left_child]  # Assign same prediction as children
                    return ("pure", left_result[1])  # Return class

                return ("mixed", None)  # If left and right are different, keep the split

            recursive_prune(node_index)

        # Apply pruning
        prune_tree(pruned_tree)

        # Visualize the pruned tree with a title
        plt.figure(figsize=(15, 7), dpi=150)
        plot_tree(
            pruned_tree,
            feature_names=df.columns,
            class_names=["NR", "R"],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title(f"Decision Tree of Data", fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi = 300)
        else:
            plt.show()

        return pruned_tree  # Return the pruned tree


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
    parser = argparse.ArgumentParser(description="Run Precise analysis pipeline")
    # parser.add_argument("--adata_path", type=str, required=True, help="Path to the AnnData file (H5AD format)")
    parser.add_argument("--adata_path", type=str, default='examples/sample_melanoma_adata.h5ad', required=False, help="Path to the AnnData file (H5AD format)")
    parser.add_argument("--output_folder", type=str, default="./results", help="Path to the results folder")
    parser.add_argument("--model_name", type=str, default="XGBoostClassifier", help="Model name to use")
    parser.add_argument("--celltype", type=str, default='T cells', help="Subset by cell type")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the model")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for the model")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs for training neural networks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--k_folds", type=int, default=None, help="Number of folds for cross-validation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--weighted_prediction", action="store_true", help="Enable weighted predictions")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for XGBoost (see explanation in Git)")

    args = parser.parse_args()
    args.verbose=True
    
    # Load AnnData
    adata = sc.read_h5ad(args.adata_path)
    output_folder = create_output_folder(args.output_folder, args)
    args.output_folder = output_folder
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Initialize Precise and run CV prediction
    precise = Precise(adata, model_name=args.model_name, output_dir=args.output_folder, target_column="response", sample_column="sample", celltype=args.celltype, args=args)
    
    prediction_results = precise.cv_prediction(k_folds=args.k_folds)
    results_df, auc_score, estimators = prediction_results
    print(f"Finished cross-validation prediction. AUC: {auc_score:.4f}")
    precise.run_shap_visualizations(save_path=os.path.join(args.output_folder, 'plots'))
    precise.shap_visualizer.shap_waterfall_plot('Pre_P3', save_folder = os.path.join(args.output_folder, 'plots'))