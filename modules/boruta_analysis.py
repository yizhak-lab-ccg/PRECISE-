import os
import json
import numpy as np
from boruta import BorutaPy
from utils import validate_anndata, validate_response_column


class BorutaAnalyzer:
    def __init__(self, adata, model, model_name, celltype = None, output_dir="../results", response_col="response", sample_col="sample", verbose=True):
        """
        Initialize the BorutaAnalyzer.

        Parameters:
            adata (AnnData): Input AnnData object.
            output_dir (str): Directory to save Boruta analysis results.
            response_col (str): Column in `obs` with response labels ('R'/'NR' or 1/0).
            sample_col (str): Column in `obs` with sample identifiers.
            verbose (bool): Whether to print detailed logs.
        """
        self.adata = adata
        self.model = model
        self.model_name = model_name
        self.celltype = celltype
        self.output_dir = output_dir
        self.response_col = response_col
        self.sample_col = sample_col
        self.verbose = verbose

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def run_boruta(
        self,
        cluster_col=None,
        test_adata=None,
        subset_features=False,
        estimator=None,
        importance_file="boruta_chosen_features.json",
    ):
        """
        Run Boruta feature selection and optionally make predictions.

        Parameters:
            cluster_col (str, optional): Filter for a specific cluster if provided.
            test_adata (AnnData, optional): Test AnnData object to make predictions.
            subset_features (bool): If True, returns a subsetted AnnData with selected features.
            estimator (sklearn.BaseEstimator, optional): Custom estimator to use in Boruta. Defaults to XGBoostClassifier.
            importance_file (str): Path to save selected features as a JSON file.

        Returns:
            AnnData (optional): Subsetted AnnData object with selected features if `subset_features` is True.
            dict: Dictionary of chosen features.
            pd.DataFrame (optional): Prediction results on test AnnData if provided.
        """
        if self.model_name not in ['XGBClassifier', 'LightGBMClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier']:
            raise ValueError(f"Model {self.model_name} is not supported for Boruta analysis.")

        # Validate AnnData and columns
        validate_anndata(self.adata, [self.response_col, self.sample_col])
        validate_response_column(self.adata, self.response_col)

        if self.celltype:
            adata = adata[adata.obs[self.celltype] == 1]

        # Filter by cluster column if provided
        adata = self.adata
        if cluster_col:
            validate_anndata(adata, [cluster_col])
            adata = adata[adata.obs[cluster_col] == 1]

        # Prepare data for Boruta
        X = adata.to_df().values
        y = adata.obs[self.response_col].values


        estimator = self.model.model
        # Monkey-patch np.int to avoid deprecation issue
        if not hasattr(np, "int"):  
            np.int = int
        
        if not hasattr(np, "bool"):  
            np.bool = bool

        if not hasattr(np, "float"):  
            np.float = float

        # Run Boruta
        feat_selector = BorutaPy(estimator, n_estimators="auto", verbose=2 if self.verbose else 0, random_state=1)
        feat_selector.fit(X, y)

        # Save selected features
        chosen_features = {
            "confirmed": list(adata.var_names[feat_selector.support_]),
            "tentative": list(adata.var_names[feat_selector.support_weak_]),
        }
        cluster_name = self.celltype.replace(' ','_') if self.celltype else "all_cells"
        importance_path = os.path.join(self.output_dir, cluster_name+'_'+importance_file)
        with open(importance_path, "w") as f:
            json.dump(chosen_features, f)

        if self.verbose:
            print(f"Selected features saved to {importance_path}")

        # Subset AnnData if requested
        subset_adata = None
        if subset_features:
            subset_adata = adata[:, feat_selector.support_]

        # Predictions on test data
        prediction_results = {}
        if test_adata is not None:
            
            validate_anndata(test_adata, [self.response_col])
            validate_response_column(test_adata, self.response_col)
            
            # Ensure the test AnnData uses the same selected features
            test_adata = test_adata[:, chosen_features["confirmed"]]

            # Fit the estimator on training data
            adata = adata[:, chosen_features["confirmed"]]
            estimator.fit(adata.to_df(), adata.obs[self.response_col])
            # Iterate over samples in the test data
            for sample in test_adata.obs[self.sample_col].unique():
                sample_data = test_adata[test_adata.obs[self.sample_col] == sample].to_df()
                
                # Make predictions
                predictions = estimator.predict(sample_data)
                prediction_results[sample] = np.mean(predictions)

        return (subset_adata, chosen_features, prediction_results) if subset_features else (None, chosen_features, prediction_results)
