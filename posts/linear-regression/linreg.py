import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

# import numpy.typing as npt
# PSA: PEP 646 – Variadic Generics has been accepted for Python 3.11
# https://stackoverflow.com/a/66657424

class LinearRegression:
    """A class for least-squares linear regression.

    Attributes are only available after calling `self.fit()`.

    Attributes:
        w (np.ndarray): weights of the classifier.
        history (list[int]): accuracy of the classifier on each iteration (index).
    """
    def __init__(self) -> None:
        """Initialize linear regression class with no arguments.
        """
        self.w = None
        self.history = None

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = "analytic", max_iter: int = 1000, alpha: float = 0.1) -> None:
        """Fit a linear regression.

        TODO: detail fitting process
        Populates `self.w` and `self.history`.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.
            method (str): "analytic" (default) for analytic formula, "gradient" for gradient descent.
            max_iter (int): if "gradient", maximum number of iterations.
            alpha (float): if "gradient", learning rate.
        """
        pass

    def __fit_analytic(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using use the analytical formula for the optimal weight vector w.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.
        """
        pass

    def __fit_gradient(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using the gradient of the loss function.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.
        """
        pass


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with fitted linear regression model.

        Args:
            X (np.ndarray): matrix of n observations of p features to predict on.
        
        Returns:
            np.ndarray: values predicted.
        """
        # make sure weights are present
        if self.w is None:
            raise NotFittedError("This linear regression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        return X @ self.w

    def score(self, X: np.ndarray, y: np.ndarray) -> int:
        """Calculate accuracy score on X and y with fitted linear regression model.
        
        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.
        
        Returns:
            int: accuracy score of model predictions, a number between 0 and 1.
        """
        if self.w is None:
            raise NotFittedError("This linear regression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
