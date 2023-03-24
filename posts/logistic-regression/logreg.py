import numpy as np
from sklearn.exceptions import NotFittedError

# import numpy.typing as npt
# PSA: PEP 646 – Variadic Generics has been accepted for Python 3.11
# https://stackoverflow.com/a/66657424

# Utility functions


def pad(X: np.ndarray) -> np.ndarray:
    """Turns X into X_ with a constant column of 1's.

    Source: [Optimization for Logistic Regression](https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-optimization.html#tips-and-hints)

    Args:
        X (np.ndarray): Array to transform.

    Returns:
        np.ndarray: transformed array.
    """
    return np.append(X, np.ones((X.shape[0], 1)), 1)


def sigmoid(z: float) -> float:
    """The sigmoid, or the standard logistic function.

    Source: [Optimization with Gradient Descent](https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html#gradient-descent-for-empirical-risk-minimization)

    .. math::
        S(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}=1-S(-x)

    Args:
        z (float): input

    Returns:
        (float): output of the sigmoid.
    """
    return 1 / (1 + np.exp(-z))


def logistic_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Logistic loss of labels, given y_hat and y.

    Source: [Optimization with Gradient Descent](https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html#gradient-descent-for-empirical-risk-minimization)

    Args:
        y_hat (np.ndarray): predicted labels.
        y (np.ndarray): true labels.

    Returns:
        float: loss from y_hat and y.
    """
    return -y * np.log(sigmoid(y_hat)) - (1 - y) * np.log(1 - sigmoid(y_hat))


class LogisticRegression:
    """A class for logistic regression.

    Attributes are only available after calling `self.fit()`.

    Attributes:
        w (np.ndarray): weights of the classifier.
        loss_history (list[float]): loss of the classifier on each iteration (index).
        score_history (list[float]): score of the classifier on each iteration (index).
    """

    def __init__(self) -> None:
        """Initialize logistic regression with no arguments.
        """
        self.w = None
        self.loss_history = []
        self.score_history = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.001,
        max_epochs: int = 1000
    ) -> None:
        """Fit a logistic regression with gradient descent on logistic loss

        Starting with a random weight, we compute the next weight vector `self.w` with gradient descent.
        Populates `self.w`, `self.loss_history`, and `self.score_history`.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.
            alpha (float): learning rate for gradient descent.
            max_epochs (int): maximum steps to run algorithm.
        """

        # X_ for convenience
        X_ = pad(X)

        # initialize
        self.w = np.random.rand(X.shape[1] + 1)
        prev_loss = np.inf

        # until max_epochs or convergence
        # main loop
        for _ in range(max_epochs):
            # gradient of the empirical risk for logistic regression
            # Using formula from lecture notes
            gradient = np.mean(
                (sigmoid(X_ @ self.w) - y) @ X_, axis=0
            )
            # gradient step
            self.w = self.w - alpha * gradient

            # append scores
            new_loss = self.loss(X, y)
            self.loss_history.append(new_loss)
            self.score_history.append(self.score(X, y))
            if np.isclose(new_loss, prev_loss):
                # stop looping
                return
            prev_loss = new_loss

    def fit_stochastic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.001,
        max_epochs: int = 1000,
        batch_size: int = 10,
        momentum: bool = False
    ) -> None:
        """Fit a logistic regression with stochastic gradient descent (optionally, with the momentum method).

        Starting with a random weight, we compute the next weight vector `self.w` with stochastic gradient descent.
        After splitting into random subsets of size `batch_size`, we calculate the gradient on each subset and update.
        Populates `self.w`, `self.loss_history`, and `self.score_history`.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.
            alpha (float): learning rate for gradient descent.
            max_epochs (int): maximum steps to run algorithm.
            batch_size (int): size of "mini-batches" for stochastic gradient descent.
            momentum (bool): whether to use the momentum method.
                If True, uses momentum of 0.8.
                See [Hardt and Recht](https://arxiv.org/pdf/2102.05242.pdf), p. 85.
        """
        # X_ for convenience
        X_ = pad(X)
        n = X_.shape[0]

        # initialize
        self.w = np.random.rand(X.shape[1] + 1)
        prev_w = self.w
        prev_loss = np.inf

        # until max_epochs or convergence
        # main loop
        for _ in range(max_epochs):

            # shuffle
            order = np.arange(n)
            np.random.shuffle(order)

            # loop for each minibatch
            for batch in np.array_split(order, n // batch_size + 1):
                X__batch = X_[batch, :]
                y_batch = y[batch]

                # gradient of the empirical risk for logistic regression
                # Using formula from lecture notes
                gradient = np.mean(
                    (sigmoid(X__batch @ self.w) - y_batch) @ X__batch, axis=0
                )
                # gradient step
                # beta = momentum * 0.8
                self.w = self.w - alpha * gradient + \
                    momentum * 0.8 * (self.w - prev_w)
                # previous w for next batch
                prev_w = self.w

            # end of each epoch
            # append scores
            new_loss = self.loss(X, y)
            self.loss_history.append(new_loss)
            self.score_history.append(self.score(X, y))
            if np.isclose(new_loss, prev_loss):
                # stop looping
                return
            # previous loss for next epoch
            prev_loss = new_loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with fitted logistic regression model.

        Args:
            X (np.ndarray): matrix of n observations of p features to predict on.

        Returns:
            np.ndarray: n predicted binary labels of 0 or 1.
        """
        # make sure weights are present
        if self.w is None:
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        # X_ for convenience
        X_ = pad(X)
        return ((X_ @ self.w) > 0) * 1

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score on X and y with fitted logistic regression model.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.

        Returns:
            float: accuracy score of model predictions, a number between 0 and 1.
        """
        if self.w is None:
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        # mean of predictions
        # return np.sum(y = self.predict(X)) / X.shape[0]
        return np.mean(y == self.predict(X))

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate overall loss (empirical risk) of the current weights.

        Args:
            X (np.ndarray): matrix of predictor variables, with n observations of p features.
            y (np.ndarray): n binary labels of 0 or 1.

        Returns:
            float: accuracy score of model predictions, a number between 0 and 1.
        """
        if self.w is None:
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        X_ = pad(X)
        y_hat = X_ @ self.w
        return np.mean(logistic_loss(y_hat, y))
