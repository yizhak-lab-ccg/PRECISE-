import copy
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

# from xgb_estimator import XGBEstimator
from utils import (
    validate_anndata,
    validate_response_column,
    DPI,
    TITLE_FONT_SIZE,
    AXIS_LABELS_FONT_SIZE,
    AXIS_TICKS_FONT_SIZE,
)


class PredictionAnalyzer:
    def __init__(self, adata, model, model_name, celltype=None, target_column="response", sample_column="sample", results_folder="../results", plots_folder='plots', verbose=True):
        """
        Initialize the LOO Prediction Analyzer.

        Parameters:
            adata (AnnData): AnnData object containing the data.
            results_folder (str): Directory to store results.
            verbose (bool): Whether to print logs.
        """
        self.adata = adata
        self.target_column = target_column
        self.sample_column = sample_column
        self.results_folder = results_folder
        self.celltype = celltype
        self.plots_folder = os.path.join(results_folder, plots_folder)
        self.importance_folder = os.path.join(results_folder, "importance_scores_" +celltype if celltype else "importance_scores")
        self.verbose = verbose
        self.model_name = model_name
        self.model = model

        # Default colormap for feature plots
        self.default_colormap = LinearSegmentedColormap.from_list(
            "custom_cmap",
            ['#c44569', '#c44569', '#c44569', '#f78fb3', '#f78fb3', '#f78fb3', '#f5cd79', '#ffebbf'][::-1],
            N=256
        )

        # Ensure results and importance folders exist
        os.makedirs(self.plots_folder, exist_ok=True)
        os.makedirs(self.importance_folder, exist_ok=True)

    def simple_prediction(self, train_adata, test_adata, celltype=None):
        """
        Perform a simple prediction using XGBEstimator.

        Parameters:
            train_adata (AnnData): Training AnnData object.
            test_adata (AnnData): Test AnnData object.
            celltype (str, optional): Filter for a specific cell type.

        Returns:
            tuple: A dictionary with prediction scores and a DataFrame of feature importances.
        """
        # Validate input AnnData objects
        validate_anndata(train_adata, [self.target_column, self.sample_column])
        validate_anndata(test_adata, [self.sample_column])
        validate_response_column(train_adata, self.target_column)

        celltype = self.celltype if celltype is None else celltype
        # Filter by cell type if specified
        if celltype:
            print(f"Filtering for cell type: {celltype}.")
            if celltype in train_adata.obs:
                train_adata = train_adata[train_adata.obs[celltype] == 1]
            else:
                raise ValueError(f"Cell type '{celltype}' not found in train_adata.")
            
            if celltype in test_adata.obs:
                test_adata = test_adata[test_adata.obs[celltype] == 1]
            else:
                raise ValueError(f"Cell type '{celltype}' not found in test_adata.")
        if self.verbose:
            print(f"Starting Simple Prediction.")
            print(f"Training on {train_adata.shape[0]} cells from {train_adata.obs[self.sample_column].nunique()} samples.")
            print(f"Predicting for {test_adata.shape[0]} cells from {test_adata.obs[self.sample_column].nunique()} samples.")
        scores = {}

        # Prepare training data
        X_train, y_train = train_adata.X, train_adata.obs[self.target_column].values

        # Train the model
        estimator = copy.deepcopy(self.model)
        estimator.fit(X_train, y_train)


        feature_importance_df = None
        if hasattr(estimator, "feature_importances_") and estimator.feature_importances_ is not None:
            # Extract feature names from adata.var if available
            feature_names = train_adata.var_names.tolist()  # Feature names from adata

            # Create a DataFrame with feature importances
            feature_importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": estimator.feature_importances_,
            }).sort_values(by="Importance", ascending=False)


        # Iterate over samples in the test data
        for sample in test_adata.obs[self.sample_column].unique():
            sample_data = test_adata[test_adata.obs[self.sample_column] == sample].X
            
            # Make predictions
            predictions = estimator.predict(sample_data)
            scores[sample] = np.mean(predictions)
        if self.verbose:
            print(f"Predicted scores for samples: {list(scores.keys())}")
        return (scores, feature_importance_df)


    def cv_prediction(self, k_folds=None, seed=None, weighted_prediction = False, sample_column = 'sample_name', response_column = 'response', celltype = None, save_adata_with_predictions = False, verbose=None):
        """
        Perform either K-Fold cross-validation or Leave-One-Out (LOO) prediction with stratification
        at the sample level for binary classification.

        Parameters:
        - adata: AnnData object containing single-cell data.
        - output_folder: Path to the output folder.
        - k_folds: Number of folds for cross-validation (default: None, which uses LOO).
        - seed: Random seed for reproducibility (default: None).
        - verbose: Whether to print progress and results (default: True).

        Returns:
        - results_df: DataFrame containing fold-level results.
        - auc_score: Overall AUC score.
        - estimators: List of trained estimators.
        """
        if verbose is None:
            verbose = self.verbose
        
        # Ensure output directory for feature importances exists
        importance_folder = self.importance_folder
        os.makedirs(importance_folder, exist_ok=True)

        validate_anndata(self.adata, [self.target_column, self.sample_column])
        validate_response_column(self.adata, self.target_column)

        celltype = self.celltype if celltype is None else celltype
        adata = self.adata.copy()
        if celltype:
            if self.verbose:
                print(f"Filtering for cell type: {celltype}.")
            adata = adata[adata.obs[celltype] == 1]           

        if self.verbose:
            print(f"Starting LOO Prediction with {adata.shape[0]} cells and {adata.shape[1]} features.")
            print(f"Sample column: {self.sample_column}, Target column: {self.target_column}.")
        adata = self.adata
        # Extract sample-level labels
        sample_labels = adata.obs.groupby(sample_column, observed=False)[response_column].mean()
        sample_names = sample_labels.index
        sample_responses = sample_labels.values

        # Choose cross-validation strategy
        if k_folds is None:
            cv = LeaveOneOut()
            n_splits = cv.get_n_splits(sample_names)
            if verbose:
                print(f"Using Leave-One-Out cross-validation with {n_splits} splits.")
        else:
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            if verbose:
                print(f"Using {k_folds}-Fold cross-validation with stratification.")

        # Initialize results containers
        sample_scores = []
        sample_labels = []
        estimators = []
        predictions = []
        results = []
        adata.obs_names_make_unique()

        adata.obs['prediction'] = -1
        adata.obs['proba_prediction'] = -1
        adata.obs["proba_prediction"] = adata.obs["proba_prediction"].astype("float64")

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(sample_names, sample_responses)):
            train_samples = sample_names[train_idx]
            test_samples = sample_names[test_idx]

            train_set = adata[adata.obs[sample_column].isin(train_samples)]
            test_set = adata[adata.obs[sample_column].isin(test_samples)]
            test_set.obs = test_set.obs.copy()

            # Prepare data for training and testing
            X_train, y_train = train_set.X, train_set.obs[self.target_column].values
            X_test, y_test = test_set.X, test_set.obs[self.target_column].values
            
            model = copy.deepcopy(self.model)
            model.fit(X_train, y_train)
            estimators.append(model)

            if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
                # Extract feature names from adata.var if available
                feature_names = train_set.var_names.tolist()  # Feature names from adata

                # Create a DataFrame with feature importances
                feature_importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": model.feature_importances_,
                }).sort_values(by="Importance", ascending=False)

                # Save to CSV
                feature_importance_df.to_csv(
                    os.path.join(importance_folder, f"fold_{fold_idx + 1}_feature_importances.csv"),
                    index=False,
                )

            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)
            # Make predictions
            test_set.obs["proba_prediction"] = y_pred_prob
            test_set.obs["prediction"] = y_pred

            # Append results to the main AnnData object
            adata.obs.loc[test_set.obs.index, "proba_prediction"] = y_pred_prob.astype("float")
            adata.obs.loc[test_set.obs.index, "prediction"] = y_pred.astype("int")

            prediction_column = "prediction"
            if weighted_prediction:
                prediction_column = "proba_prediction"
            
            mean_score_per_sample = test_set.obs.groupby(sample_column, observed=False)[prediction_column].mean()
            mean_label_per_sample = test_set.obs.groupby(sample_column, observed=False)[response_column].mean()

            sample_scores.extend(mean_score_per_sample)
            sample_labels.extend(mean_label_per_sample)
            results.extend([(fold_idx + 1, sample, label, score) for sample, label, score in zip(mean_score_per_sample.index, mean_label_per_sample, mean_score_per_sample)])

            if verbose:
                print(f"Fold {fold_idx + 1}: Processed {len(mean_score_per_sample)} samples.")
                print(mean_score_per_sample.index, mean_label_per_sample, mean_score_per_sample)

        # Save results
        results_df = pd.DataFrame(results, columns=["Fold", "Sample", "Label", "Score"])
        results_file = os.path.join(self.results_folder, "loo_results.csv" if k_folds is None else "kfold_results.csv")
        results_df.to_csv(results_file, index=False)

        # Calculate overall AUC
        auc_score = roc_auc_score(sample_labels, sample_scores)
        if verbose:
            print(f"Overall AUC: {auc_score:.4f}")
        if save_adata_with_predictions:
            adata.obs['prediction'] = adata.obs['prediction'].astype('int')
            adata.obs['proba_prediction'] = adata.obs['proba_prediction'].astype('float')
            adata.write(os.path.join(self.results_folder, 'adata_with_prediction.h5ad'))
        return (results_df, auc_score, estimators)


    def feature_importance_colored_bar(self, top_n_features, colormap=None, save_path=None):
        """
        Plot a horizontal bar chart of feature importance.

        Parameters:
            top_n_features (pd.DataFrame): DataFrame containing 'Feature' and 'Importance' columns.
            colormap (str or Colormap): Colormap for the bars (uses default if None).
            save_path (str): Path to save the plot.
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise NotImplementedError("The model does not support feature importances.")
        top_n_features = top_n_features.sort_values(by="Importance", ascending=False)
        norm = plt.Normalize(top_n_features["Importance"].min(), top_n_features["Importance"].max())
        colors = (plt.get_cmap(colormap) if colormap else self.default_colormap)(norm(top_n_features["Importance"]))

        plt.figure(figsize=(5, 4))
        plt.barh(top_n_features.index, top_n_features["Importance"], color=colors, edgecolor="w")
        plt.xlabel("Feature Importance", fontsize=AXIS_LABELS_FONT_SIZE)
        plt.title("Top Feature Importances", fontsize=TITLE_FONT_SIZE)
        plt.xticks(fontsize=AXIS_TICKS_FONT_SIZE)
        plt.yticks(fontsize=AXIS_TICKS_FONT_SIZE)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=DPI)
            print(f"Feature importance bar plot saved to {save_path}.")
        
        plt.show()
        plt.close()
    

    def create_feature_plots(self, n_intersecting_genes = 500):
        """
        Generate feature importance and intersection plots using default settings.
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise NotImplementedError("The model does not support feature importances.")
        try:
            files = os.listdir(self.importance_folder)
            files = [file for file in files if file.endswith(".csv")]
            features_lists, df_lists = [], []

            # Load feature importances
            for file in files:
                temp_df = pd.read_csv(os.path.join(self.importance_folder, file))
                df_lists.append(temp_df)
                features_lists.append(temp_df["Feature"].values)

        except Exception as e:
            print(f"An error occurred while loading feature importance files, did you run LOO?: {str(e)}")
        # Compute mean importance scores

        mean_scores = pd.concat(df_lists).groupby("Feature").mean().sort_values("Importance", ascending=False)

        # Intersection analysis
        max_k = len(features_lists[0])
        k_values = range(1, min(n_intersecting_genes, max_k + 1))
        intersection_counts = [
            len(set.intersection(*[set(features[:k]) for features in features_lists]))
            for k in k_values
        ]

        os.makedirs(self.plots_folder, exist_ok=True)

        # Plot intersection results
        plt.figure(figsize=(6, 5))
        plt.plot(
            k_values,
            intersection_counts,
            marker="o",
            linestyle="-",
            color="#BAC6FF",
            linewidth=4,
            label="Intersection Size"
        )
        plt.title("Feature Intersection Lengths", fontsize=TITLE_FONT_SIZE)
        plt.xlabel("Number of Intersecting Genes", fontsize=AXIS_LABELS_FONT_SIZE)
        plt.ylabel("Intersection Size", fontsize=AXIS_LABELS_FONT_SIZE)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        cluster_name = self.celltype.replace(' ', '_') if self.celltype else "all_cells"
        plt.savefig(os.path.join(self.plots_folder,  cluster_name+ "_intersection_plot.png"), dpi=DPI)
        print(f"Intersection plot saved to {self.plots_folder}.")
        plt.show()
        plt.close()

        # Plot feature importance
        self.feature_importance_colored_bar(
            mean_scores.head(25),
            save_path=os.path.join(self.plots_folder, cluster_name + "_feature_importance_barplot.png")
        )
        
    def choose_top_features(self, k):
        """
        Select the top-k intersecting features across feature importance files.

        Parameters:
        - k (int): The number of intersecting features to select.

        Returns:
        - set: A set of intersecting features containing at least k features.
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise NotImplementedError("The model does not support feature importances.")
        try:
            # Validate importance folder
            if not os.path.exists(self.importance_folder):
                raise FileNotFoundError("Importance folder not found. Have you run LOO yet?")
            
            files = os.listdir(self.importance_folder)
            if not files:
                raise ValueError("No feature importance files found in the importance folder.")

            features_lists = []

            # Load feature importance
            for file in files:
                temp_df = pd.read_csv(os.path.join(self.importance_folder, file))
                if "Feature" not in temp_df.columns:
                    raise ValueError(f"Feature column missing in {file}.")
                features_lists.append(temp_df["Feature"].values)

            # Compute intersecting features
            intersection_len = 0
            n_features = 0
            max_features = len(features_lists[0])
            
            while intersection_len < k:
                # Compute intersection with current n_features
                intersecting_features = set.intersection(*[set(features[:n_features]) for features in features_lists])
                intersection_len = len(intersecting_features)

                # Adjust step size based on n_features
                if n_features < 100 and n_features < max_features:
                    n_features += 1
                elif n_features < 800 and n_features < max_features:
                    n_features += 10
                elif n_features < max_features:
                    n_features += 100
                else:
                    break  # Avoid infinite loop if k > total features

            if intersection_len < k:
                raise ValueError(f"Unable to find {k} intersecting features. Found {intersection_len}.")

            return list(intersecting_features)

        except FileNotFoundError as e:
            print(str(e))
        except ValueError as e:
            print(str(e))
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")



