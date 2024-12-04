"""Logistic Regression module for NLP.

This module contains the Logistic Regression class for NLP tasks.

Hints:
- Take a look at [Python's assert statement](https://realpython.com/python-assert-statement/#the-syntax-of-the-assert-statement)
- Take a look at [`np.where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
- Get familiar with the difference between [`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html), [`np.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html), and [`np.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html).
"""

import numpy as np


class LogisticRegression:
    """Logistic Regression class for NLP tasks.

    This class implements a logistic regression model for NLP tasks.
    It can be used for binary classification tasks.

    Attributes:
        theta (np.ndarray): the weights of the model, None before training
        cost (np.ndarray): the cost of the model, None before training
        learning_rate (float): the learning rate of the model
        n_iter (int): the maximum number of iterations for the training
    """

    def __init__(self, learning_rate: float = 1e-9, n_iter: int = 1500) -> None:
        """Initialize the LogisticRegression class.

        The init method accepts two hyperparameters as optional arguments, the learning rate and the number of iterations.

        There are two additional attributes that cannot be passed as arguments, the weights `theta` and the cost `cost`.
        They are of type `np.ndarray` and should be initialized to `None`.

        Args:
            learning_rate (float, optional): the learning rate of the model. Defaults to 1e-9.
            n_iter (int, optional): the number of iterations for the training. Defaults to 1500.
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.theta: np.ndarray = None
        self.cost: np.ndarray = None

    def _initialize_weights(self, n_features: int) -> None:
        """Initialize the weights of the model.

        The weights are initialized to a numpy array of zeros with the shape (n_features, 1).

        Args:
            n_features (int): the number of features of the model
        """
        self.theta = np.zeros((n_features, 1))

    def _sigmoid(self, z: np.ndarray | int | float) -> np.ndarray | float:
        """Compute the sigmoid of z.

        The function accepts a numpy array, a float, or an integer as input.

        Args:
            z (np.ndarray | int | float): the logit as a numpy array, a float, or an integer.

        Returns:
            np.ndarray | float: the sigmoid of the logit, i.e. a probability value
        """
        return 1 / (1 + np.exp(-z))

    def _cost_function(
        self,
        y: np.ndarray,
        h: np.ndarray,
    ) -> None:
        """Compute the cost in the current iteration.

        The number of samples m is obtained from the shape of the labels `y`.

        Args:
            y (np.ndarray): the labels of the training data in shape (m, 1), where m is the number of samples.
            h (np.ndarray): the predictions of the model as obtained from the sigmoid function in shape (m, 1), where m is the number of samples.
        """
        m = y.shape[0]
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        self.cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def _update_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        h: np.ndarray,
    ) -> None:
        """Update the weights of the model.

        This function implements the formula to compute the gradient and update the weights of the model.

        Args:
            X (np.ndarray): the training data in shape (m, n), where m is the number of samples and n is the number of features.
            y (np.ndarray): the labels of the training data in shape (m, 1), where m is the number of samples.
            h (np.ndarray): the predictions of the model as obtained from the sigmoid function in shape (m, 1), where m is the number of samples.
        """
        m = y.shape[0]
        gradient = (1 / m) * np.dot(X.T, (h - y))
        self.theta -= self.learning_rate * gradient

    def _gradient_descent(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform gradient descent with training data X and labels y.

        The process is as follows:
        1. Initialize the weights. The weights are initialized to a numpy array of zeros with the shape (n_features, 1).
        2. For each iteration of the specified number of iterations in the constructor:
            2.1 Compute the predictions using the sigmoid function.
            2.2 Compute the cost.
            2.3 Update the weights.

        For convenience, the function returns the weights and the cost after the specified number of iterations.

        Args:
            X (np.ndarray): the training data in shape (m, n), where m is the number of samples and n is the number of features.
            y (np.ndarray): the labels of the training data in shape (m, 1), where m is the number of samples.

        Returns:
            np.ndarray: the weights `theta` of the model
            np.ndarray: the minimized cost of the model after `n_iter` iterations

        """
        n_features = X.shape[1]
        self._initialize_weights(n_features=n_features)

        for _ in range(self.n_iter):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)

            self._cost_function(y, h)

            self._update_weights(X, y, h)

        return self.theta, self.cost

    def fit(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Performs a gradient descent with training data X and labels y.

        Before executing gradient descent, a couple of assertions are performed to check the validity of the input data:
            - The number of samples in X and y must be equal.
            - X must be a 2-dimensional array.
            - y must be a 2-dimensional array.
            - y must be a column vector.

        If all assertions pass, the gradient descent is executed.

        Args:
            X (np.ndarray): the training data in shape (m, n), where m is the number of samples and n is the number of features.
            y (np.ndarray): the labels of the training data in shape (m, 1), where m is the number of samples.

        Returns:
            np.ndarray: the weights `theta` of the model
            np.ndarray: the minimized cost of the model after `n_iter` iterations
        """
        assert (
            X.shape[0] == y.shape[0]
        ), "The number of samples in X and y must be equal."
        assert X.ndim == 2, "X must be a 2-dimensional array."
        assert y.ndim == 2, "y must be a 2-dimensional array."
        assert y.shape[1] == 1, "y must be a column vector."

        return self._gradient_descent(X, y)

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """Predict the probability for the given samples using the trained weights.

        Args:
            X (np.ndarray): the samples to predict in shape (m, n), where m is the number of samples and n is the number of features.

        Returns:
            np.ndarray: the predicted probabilities as a value between 0 and 1 for the given samples in shape (m, 1), where m is the number of samples.
        """
        if self.theta is None:
            raise ValueError(
                "Model weights are not initialized. Train the model before predicting."
            )

        z = np.dot(X, self.theta)

        probabilities = self._sigmoid(z)

        return probabilities

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict the labels for the given samples using the trained weights.

        The default threshold is 0.5, i.e. if the predicted probability is greater or equal to 0.5, the label is 1, otherwise 0.

        Args:
            X (np.ndarray): the samples to predict in shape (m, n), where m is the number of samples and n is the number of features.
            threshold (float, optional): the threshold to use for the prediction. Defaults to 0.5.
        Returns:
            np.ndarray: the predicted labels for the given samples in shape (m, 1), where m is the number of samples.
        """
        propabilities = self.predict_prob(X)

        predictions = np.where(propabilities >= threshold, 1, 0)

        return predictions
