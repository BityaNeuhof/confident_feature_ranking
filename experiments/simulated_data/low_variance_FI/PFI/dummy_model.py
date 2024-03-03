from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
import pandas as pd


class DummyRegressor(BaseEstimator):

    def __init__(self, prediction_function):
        self.prediction_function = prediction_function
        self.i_ = 1

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        # Return the regressor
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        pred = self.prediction_function(X, seed=self.i_)
        self.i_ += 1
        return pred