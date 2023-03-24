import numpy as np
from sklearn.exceptions import NotFittedError

# import numpy.typing as npt
# PSA: PEP 646 – Variadic Generics has been accepted for Python 3.11
# https://stackoverflow.com/a/66657424


class Perceptron:
    """A class for the perceptron classifier.

    Attributes are only available after calling `self.fit()`.

    Attributes:
        w (np.ndarray): weights of the classifier.
        history (list[float]): accuracy of the classifier on each iteration (index).
    """

    def __init__(self) -> None:
        """Initialize perceptron with no arguments.
        """
        self.w = None
        self.history = []

    def fit(self, X: np.ndarray, y: np.ndarray, max_steps: int = 1000) -> None:
        """Fit a perceptron.

        Starting with a random weight, we compute the next weight vector `self.w` with the perceptron algorithm.
        Populates `self.w` and `self.history`.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.
            max_steps (int): maximum steps to run algorithm.
        """

        # X_, y_ for convenience
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        y_ = 2 * y - 1

        # initialize random w
        self.w = np.random.rand(X.shape[1] + 1)

        # until max_steps or score is 1.0
        for _ in range(max_steps):
            # [low, high)
            i = np.random.randint(0, X.shape[0])
            # ((y_[i] * self.w @ X_[i]) < 0):
            #     whether y_[i] and self.w @ X_[i] have different signs
            #     if yes, step `self.w``
            self.w = self.w + ((y_[i] * self.w @ X_[i]) < 0) * y_[i] * X_[i]

            # append score
            score = self.score(X, y)
            self.history.append(score)
            if score == 1.0:
                # stop looping
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with fitted perceptron model.

        Args:
            X (np.ndarray): matrix of n observations of p features to predict on.

        Returns:
            np.ndarray: n predicted binary labels of 0 or 1.
        """
        # make sure weights are present
        if self.w is None:
            raise NotFittedError(
                "This Perceptron instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        # X_ for convenience
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)

        # Should I use @?
        # https://mkang32.github.io/python/2020/08/30/numpy-matmul.html#dot_vs_matmul
        # comparision returns True and False, which can be used like 1 and 0
        return (X_ @ self.w) > 0

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score on X and y with fitted perceptron model.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.

        Returns:
            float: accuracy score of model predictions, a number between 0 and 1.
        """
        if self.w is None:
            raise NotFittedError(
                "This Perceptron instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        # mean of predictions
        # return np.sum(y = self.predict(X)) / X.shape[0]
        return np.mean(y == self.predict(X))
