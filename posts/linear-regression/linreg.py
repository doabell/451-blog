import numpy as np
from sklearn.exceptions import NotFittedError

# Utility functions


def pad(X: np.ndarray) -> np.ndarray:
    """Turns X into X_ with a constant column of 1's.

    Source: [Least-Squares Linear Regression](https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/regression.html#solution-methods)

    Args:
        X (np.ndarray): Array to transform.

    Returns:
        np.ndarray: transformed array.
    """
    return np.append(X, np.ones((X.shape[0], 1)), 1)


class LinearRegression:
    """A class for least-squares linear regression.

    Attributes are only available after calling `self.fit()`.

    Attributes:
        w (np.ndarray): weights of the regressor.
        score_history (list[float]): score of the classifier on each iteration (index).
    """

    def __init__(self) -> None:
        """Initialize linear regression class with no arguments.
        """
        self.w = None
        self.score_history = []

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = "analytic", max_iter: int = 1000, alpha: float = 0.1) -> None:
        """Fit a linear regression.

        If analytic, uses explicit formula of linear regression weights.
        If gradient, uses the gradient of the loss function.

        Populates `self.w` and `self.score_history`.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): the dependent variable.
            method (str): "analytic" (default) for analytic formula, "gradient" for gradient descent.
            max_iter (int): if "gradient", maximum number of iterations.
            alpha (float): if "gradient", learning rate.
        """
        if method == "analytic":
            X_ = pad(X)
            self.w = np.linalg.inv(X_.T @ X_) @ X_.T @ y
        else:
            self.__fit_gradient(X, y, max_iter, alpha)

    def __fit_gradient(self, X: np.ndarray, y: np.ndarray, max_iter: int = 1000, alpha: float = 0.1) -> None:
        """Linear regression using the gradient of the loss function.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): the dependent variable.
            max_iter (int): maximum number of iterations.
            alpha (float): learning rate.
        """
        # calculate once to save time
        X_ = pad(X)
        P = X_.T @ X_
        q = X_.T @ y

        # initialize
        self.w = np.random.rand(X.shape[1] + 1)
        prev_gradient = np.inf

        # until max_epochs or convergence
        # main loop
        for _ in range(max_iter):
            # Using formula from blog post instructions
            gradient = 2 * (P @ self.w - q)
            # gradient step
            self.w = self.w - alpha * gradient

            # append score
            self.score_history.append(self.score(X, y))
            # checking if gradient did not change much
            if np.allclose(gradient, prev_gradient):
                # stop looping
                return
            prev_gradient = gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with fitted linear regression model.

        Args:
            X (np.ndarray): matrix of n observations of p features to predict on.

        Returns:
            np.ndarray: values predicted.
        """
        # make sure weights are present
        if self.w is None:
            raise NotFittedError(
                "This linear regression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        
        return pad(X) @ self.w

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate loss on X and y with fitted linear regression model.

        Source:[Implementing Linear Regression](https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-linear-regression.html)

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.

        Returns:
            float: loss of model predictions, a number between 0 and 1.
        """
        if self.w is None:
            raise NotFittedError(
                "This linear regression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        return (self.predict(X) - y) ** 2

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score on X and y with fitted linear regression model.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.

        Returns:
            float: accuracy score of model predictions, a number between 0 and 1.
        """
        if self.w is None:
            raise NotFittedError(
                "This linear regression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        return 1 - (
            self.loss(X, y).sum() / ((y.mean() - y) ** 2).sum()
        )
