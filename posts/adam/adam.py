import numpy as np
from sklearn.exceptions import NotFittedError
from typing import Optional
import numpy.typing as npt

# PSA: PEP 646 – Variadic Generics has been accepted for Python 3.11
# https://stackoverflow.com/a/66657424

# Utility functions

def logsig(x):
    """Compute the log-sigmoid function component-wise.
    
    Source: [How to Evaluate the Logistic Loss and not NaN trying](https://fa.bianp.net/blog/2019/evaluate_logistic/#sec3)
    """
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


def pad(X: npt.NDArray) -> npt.NDArray:
    """Turns X into X_ with a constant column of 1's.

    Source: [Optimization for Logistic Regression](https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-optimization.html#tips-and-hints)

    Args:
        X (npt.NDArray): Array to transform.

    Returns:
        npt.NDArray: transformed array.
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


def logistic_loss(y_hat: npt.NDArray, y: npt.NDArray) -> float:
    """Logistic loss of labels, given y_hat and y.

    Source: [Optimization with Gradient Descent](https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html#gradient-descent-for-empirical-risk-minimization)

    Args:
        y_hat (npt.NDArray): predicted labels.
        y (npt.NDArray): true labels.

    Returns:
        float: loss from y_hat and y.
    """
    return -y * np.log(sigmoid(y_hat)) - (1 - y) * np.log(1 - sigmoid(y_hat))


class LogisticRegression:
    """A class for logistic regression.

    Attributes are only available after calling `self.fit()`.

    Attributes:
        w (npt.NDArray): weights of the classifier.
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
        X: npt.NDArray,
        y: npt.NDArray,
        alpha: float = 0.001,
        max_epochs: int = 1000
    ) -> None:
        """Fit a logistic regression with gradient descent on logistic loss

        Starting with a random weight, we compute the next weight vector `self.w` with gradient descent.
        Populates `self.w`, `self.loss_history`, and `self.score_history`.

        Args:
            X (npt.NDArray): matrix of predictor variables, with n observations of p features.
            y (npt.NDArray): n binary labels of 0 or 1.
            alpha (float): learning rate for gradient descent.
            max_epochs (int): maximum steps to run algorithm.
        """

        # Clear histories
        self.loss_history = []
        self.score_history = []

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
                (sigmoid(X_ @ self.w) - y).reshape(X.shape[0], 1) * X_, axis=0
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
        X: npt.NDArray,
        y: npt.NDArray,
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
            X (npt.NDArray): matrix of predictor variables, with n observations of p features.
            y (npt.NDArray): n binary labels of 0 or 1.
            alpha (float): learning rate for gradient descent.
            max_epochs (int): maximum steps to run algorithm.
            batch_size (int): size of "mini-batches" for stochastic gradient descent.
            momentum (bool): whether to use the momentum method.
                If True, uses momentum of 0.8.
                See [Hardt and Recht](https://arxiv.org/pdf/2102.05242.pdf), p. 85.
        """
        # Clear histories
        self.loss_history = []
        self.score_history = []

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
                    (sigmoid(X__batch @ self.w) - y_batch).reshape(X__batch.shape[0], 1) * X__batch, axis=0
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

    def fit_adam(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        w_0: Optional[npt.NDArray] = None,
        alpha: float = 0.001,
        max_epochs: int = 1000,
        batch_size: int = 10,
    ) -> None:
        """Fit a logistic regression with [the Adam algorithm](https://arxiv.org/pdf/1412.6980.pdf) for stochastic optimization.

        Starting with a given (or random) weight, 
            we calculate exponential moving averages of the gradient (m) and the squared gradient (v).
        Then, we compute the next weight vector `self.w` with m_hat and v_hat.
        After splitting into random subsets of size `batch_size`, we calculate the gradient on each subset and update.
        Populates `self.w`, `self.loss_history`, and `self.score_history`.

        Default values for `alpha`, `beta_1`, `beta_2`, and `epsilon` are taken from Algorithm 1 in the paper.

        Args:
            X (npt.NDArray): matrix of predictor variables, with n observations of p features.
            y (npt.NDArray): n binary labels of 0 or 1.
            beta_1, beta_2 (float): moment estimate decay rates.
            epsilon (float): small constant to avoid dividing by zero.
            w_0 (npt.NDArray): initial guess for the weight vector.
                If `None`, will be randomly generated.
            alpha (float): learning rate for gradient descent.
            max_epochs (int): maximum steps to run algorithm.
            batch_size (int): size of "mini-batches" to go through.
        """
        # Clear histories
        self.loss_history = []
        self.score_history = []

        # X_ for convenience
        X_ = pad(X)
        n = X_.shape[0]

        # initialize
        self.w = w_0 if w_0 is not None else np.random.rand(X.shape[1] + 1)
        prev_loss = np.inf

        m = np.zeros(X_.shape[1]) # Line 1
        v = np.zeros(X_.shape[1]) # Line 2

        # until max_epochs or convergence
        # main loop
        for epoch in range(max_epochs): # Line 3, 5

            # shuffle
            order = np.arange(n)
            np.random.shuffle(order)

            # loop for each minibatch
            for batch in np.array_split(order, n // batch_size + 1):
                X__batch = X_[batch, :]
                y_batch = y[batch]

                # gradient
                # Line 6
                gradient = np.mean(
                    (sigmoid(X__batch @ self.w) - y_batch).reshape(X__batch.shape[0], 1) * X__batch, axis=0
                )
                # Adam step
                # Biased estimates
                m = beta_1 * m + (1 - beta_1) * gradient # Line 7
                v = beta_2 * v + (1 - beta_2) * gradient ** 2 # Line 8
                # Bias-corrected estimates
                m_hat = m / (1 - beta_1 ** (epoch + 1)) # Line 9
                v_hat = v / (1 - beta_2 ** (epoch + 1)) # Line 10
                self.w = self.w - alpha * m_hat / (np.sqrt(v_hat) + epsilon) # Line 11

            # end of each epoch
            # append scores
            new_loss = self.loss(X, y)
            self.loss_history.append(new_loss)
            self.score_history.append(self.score(X, y))
            if np.isclose(new_loss, prev_loss):
                # Line 4
                # stop looping
                return
            # previous loss for next epoch
            prev_loss = new_loss

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """Predict with fitted logistic regression model.

        Args:
            X (npt.NDArray): matrix of n observations of p features to predict on.

        Returns:
            npt.NDArray: n predicted binary labels of 0 or 1.
        """
        # make sure weights are present
        if self.w is None:
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        # X_ for convenience
        X_ = pad(X)
        return ((X_ @ self.w) > 0) * 1

    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        """Calculate accuracy score on X and y with fitted logistic regression model.

        Args:
            X (npt.NDArray): matrix of predictor variables, with n observations of p features.
            y (npt.NDArray): n binary labels of 0 or 1.

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

    def loss(self, X: npt.NDArray, y: npt.NDArray) -> float:
        """Calculate overall loss (empirical risk) of the current weights.

        Args:
            X (npt.NDArray): matrix of predictor variables, with n observations of p features.
            y (npt.NDArray): n binary labels of 0 or 1.

        Returns:
            float: accuracy score of model predictions, a number between 0 and 1.
        """
        if self.w is None:
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        X_ = pad(X)
        y_hat = X_ @ self.w
        return np.mean((1 - y) * y_hat - logsig(y_hat))
