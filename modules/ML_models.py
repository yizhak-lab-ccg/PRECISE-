from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from scipy.sparse import issparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor, DMatrix
from lightgbm import LGBMClassifier, LGBMRegressor
# import cupy as cp  # For GPU-based arrays
# from scipy.sparse import csr_matrix  # Ensure correct type checking
# from cupyx.scipy.sparse import csr_matrix as csr_gpu

import torch
import xgboost as xgb
from sklearn.base import BaseEstimator

class SklearnModelWrapper(BaseEstimator):
    def __init__(self, model, is_regressor=False, use_gpu = False):
        self.model = model
        self.feature_importances_ = None
        self.is_regressor = is_regressor
        self.use_gpu = use_gpu

    def fit(self, X, y):
        """Convert data to GPU tensors before fitting."""
        if isinstance(self.model, (xgb.XGBRegressor, xgb.XGBClassifier)) and self.use_gpu:
            # ✅ Convert X and y to PyTorch tensors and keep on GPU
            X_tensor = torch.as_tensor(X.toarray(), dtype=torch.float32, device="cuda") if hasattr(X, "toarray") else torch.as_tensor(X, dtype=torch.float32, device="cuda")
            y_tensor = torch.as_tensor(y, dtype=torch.float32, device="cuda")

            self.model.fit(X_tensor, y_tensor)  # No need for CPU conversion
        else:
            self.model.fit(X, y)  # Use CPU for non-XGBoost models

        if hasattr(self.model, "coef_"):
            self.feature_importances_ = self.model.coef_.flatten()
        elif hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_
        else:
            self.feature_importances_ = None

    def predict(self, X):
        """Convert data to GPU tensors before prediction."""
        if isinstance(self.model, (xgb.XGBRegressor, xgb.XGBClassifier)) and self.use_gpu:
            X_tensor = torch.as_tensor(X.toarray(), dtype=torch.float32, device="cuda") if hasattr(X, "toarray") else torch.as_tensor(X, dtype=torch.float32, device="cuda")
            y_pred = self.model.predict(X_tensor)
            return y_pred if self.is_regressor else y_pred.astype(int)
        return self.model.predict(X)

    def predict_proba(self, X):
        """Convert data to GPU tensors before probability prediction."""
        if self.is_regressor:
            return self.predict(X)  # Regression models don't have probability outputs
        
        if isinstance(self.model, xgb.XGBClassifier) and self.use_gpu:
            X_tensor = torch.as_tensor(X.toarray(), dtype=torch.float32, device="cuda") if hasattr(X, "toarray") else torch.as_tensor(X, dtype=torch.float32, device="cuda")
            return self.model.predict_proba(X_tensor)[:, 1]

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        else:
            raise NotImplementedError("This model does not support probability predictions.")

    def score(self, X, y):
        """Use appropriate scoring metric depending on model type."""
        return self.model.score(X, y)



# class SklearnModelWrapper(BaseEstimator):
#     def __init__(self, model, is_regressor=False):
#         self.model = model
#         self.feature_importances_ = None
#         self.is_regressor = is_regressor

#     def fit(self, X, y):
#         self.model.fit(X, y)
#         if hasattr(self.model, "coef_"):
#             self.feature_importances_ = self.model.coef_.flatten()
#         elif hasattr(self.model, "feature_importances_"):
#             self.feature_importances_ = self.model.feature_importances_
#         else:
#             self.feature_importances_ = None

#     def predict(self, X):
#         """Predict outputs, handling both classification and regression."""
#         y_pred = self.model.predict(X)
#         return y_pred if self.is_regressor else y_pred.astype(int)

#     def predict_proba(self, X):
#         """Predict probabilities for classifiers; return predictions for regressors."""
#         if self.is_regressor:
#             return self.predict(X)  # Regression models don't have probability outputs
#         if hasattr(self.model, "predict_proba"):
#             return self.model.predict_proba(X)[:, 1]
#         else:
#             raise NotImplementedError("This model does not support probability predictions.")

#     def score(self, X, y):
#         """Use appropriate scoring metric depending on model type."""
#         return self.model.score(X, y)


class PytorchModelWrapper(BaseEstimator):
    def __init__(self, input_dim, layers, is_regressor=False, learning_rate=0.001, batch_size=32, epochs=10, device=None, verbose=True):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        self.is_regressor = is_regressor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model = self._build_model(input_dim)
        self.criterion = nn.MSELoss() if is_regressor else nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def _build_model(self, input_dim):
        model = nn.Sequential()
        for i, neurons in enumerate(self.layers):
            if i == 0:
                model.add_module(f"Layer_{i+1}", nn.Linear(input_dim, neurons))
            else:
                model.add_module(f"Layer_{i+1}", nn.Linear(self.layers[i-1], neurons))
            model.add_module(f"Activation_{i+1}", nn.ReLU())
            model.add_module(f"Dropout_{i+1}", nn.Dropout(0.2))
        model.add_module("Output", nn.Linear(self.layers[-1], 1))
        if not self.is_regressor:
            model.add_module("Output_Activation", nn.Sigmoid())
        return model.to(self.device)

    def fit(self, X, y):
        if issparse(X): 
            X = X.toarray()
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch).view(-1)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        if issparse(X): X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).cpu().numpy().flatten()
        return y_pred if self.is_regressor else (y_pred > 0.5).astype(int)


    def predict_proba(self, X):
        """Predict probabilities for the positive class."""
        from scipy.sparse import issparse

        # Handle DataFrame or sparse input
        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert DataFrame to NumPy array
        elif issparse(X):
            X = X.toarray()  # Convert sparse matrix to dense array

        # Convert to PyTorch tensor
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Perform prediction
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).cpu().numpy().flatten()

        return y_pred


    def save_model(self, path):
        """Save the PyTorch model to a file."""
        torch.save(self.model.state_dict(), path)
        if self.verbose:
            print(f"Model saved to {path}")

    def load_model(self, path, input_dim):
        """Load a PyTorch model from a file."""
        self.model = self._build_model(input_dim)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        if self.verbose:
            print(f"Model loaded from {path}")


def get_model(model_name, args, is_regressor=False):
    base_name = model_name.replace("Regressor", "").replace("Classifier", "")
    use_gpu = args.use_gpu

    use_gpu = torch.cuda.is_available() and use_gpu
    device = "cuda" if use_gpu else "cpu"  # New XGBoost 2.0+ way to enable GPU
    tree_method = "hist"  # Required when using `device="cuda"`

    max_depth = getattr(args, 'max_depth', None)

    model_mapping = {
        "LogisticRegression": SklearnModelWrapper(LogisticRegression(max_iter=1000)),
        "RandomForest": SklearnModelWrapper(RandomForestRegressor(max_depth=max_depth, random_state=42)) if is_regressor else 
                        SklearnModelWrapper(RandomForestClassifier(max_depth=max_depth, random_state=42)),
        # "XGBoost": SklearnModelWrapper(
        #     XGBRegressor(max_depth=max_depth, random_state=42, tree_method=tree_method, device=device), use_gpu=use_gpu
        # ) if is_regressor else SklearnModelWrapper(
        #     XGBClassifier(max_depth=max_depth, random_state=42, tree_method=tree_method, device=device), use_gpu=use_gpu
        # ),
        "XGBoost": SklearnModelWrapper(
            XGBRegressor(max_depth=max_depth, random_state=42, tree_method="gpu_hist" if use_gpu else "hist"), 
            use_gpu=use_gpu
        ) if is_regressor else SklearnModelWrapper(
            XGBClassifier(max_depth=max_depth, random_state=42, tree_method="gpu_hist" if use_gpu else "hist"), 
            use_gpu=use_gpu
        ),
        "LightGBM": SklearnModelWrapper(
            LGBMRegressor(max_depth=max_depth, random_state=42, verbose=-1)
        ) if is_regressor else SklearnModelWrapper(
            LGBMClassifier(max_depth=max_depth, random_state=42, verbose=-1)
        ),
        "LinearRegression": SklearnModelWrapper(LinearRegression()),
        "DecisionTreeClassifier": SklearnModelWrapper(DecisionTreeClassifier(max_depth=max_depth, random_state=42)),
    }

    if base_name.startswith("NeuralNet_"):
        architecture_index = int(base_name.split("_")[-1]) - 1
        nn_architectures = [[64, 32], [128, 64, 32], [32, 16]]
        input_dim = args.input_dim
        return PytorchModelWrapper(
            input_dim=input_dim,
            layers=nn_architectures[architecture_index],
            learning_rate=args.learning_rate if args.learning_rate else 0.005,
            epochs=args.epochs if args.epochs else 10,
            is_regressor=is_regressor
        )
    
    return model_mapping.get(base_name, None)

