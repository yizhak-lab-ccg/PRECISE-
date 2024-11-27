
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class XGBEstimator(BaseEstimator):
    def __init__(self, max_depth=7, learning_rate=0.2, gamma=0, max_delta_step=0,
                 reg_lambda=1, alpha=1, verbose=False):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_delta_step = max_delta_step
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.verbose = verbose

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        # Convert labels to integers
        classes_, y = np.unique(y, return_inverse=True)

        # Calculate class weights to address imbalances
        number_of_positives = sum(y == 1)
        number_of_negatives = sum(y == 0)
        scale_pos_weight = number_of_negatives / number_of_positives

        classifier = xgb.XGBClassifier(max_depth=self.max_depth,
                                       learning_rate=self.learning_rate,
                                       gamma=self.gamma,
                                       max_delta_step=self.max_delta_step,
                                       reg_lambda=self.reg_lambda,
                                       alpha=self.alpha,
                                       objective='binary:logistic',
                                       scale_pos_weight=scale_pos_weight)

        classifier.fit(X, y, verbose=self.verbose)

        self.feature_importances_ = classifier.feature_importances_
        self._classifier = classifier

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        prediction = self._classifier.predict(X)
        return prediction

    def get_feature_importances(self):
        # Get feature importances from the trained classifier.
        return self.feature_importances_
