import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class LinearModel:
    """Simple linear model used to predict the expected absolute error for MS2DeepScore
    predictions. The model uses a pair of embedding evaluations as input.

    This class is a wrapper for scikit-learn's LinearRegression and PolynomialFeatures,
    facilitating polynomial regression analysis.

    This class simplifies the application of polynomial features to input data
    before fitting a linear regression model, and it provides functionality to
    save model parameters to a file.

    Attributes
    ----------
    degree (int):
        The degree of the polynomial features. This determines how complex the model's
        polynomial terms will be.
    model (LinearRegression):
        An instance of scikit-learn's LinearRegression.
        This is the underlying regression model.
    poly (PolynomialFeatures):
        An instance of scikit-learn's PolynomialFeatures.
        This is used to generate polynomial and interaction features from the input variables.

    Parameters
    ----------
    degree (int, optional):
        The degreae of polynomial features to create. Defaults to 2.
    """
    def __init__(self, degree = 2):
        self.degree = degree
        self.model = LinearRegression()
        self.poly = PolynomialFeatures(degree=self.degree)

    def fit(self, X, y):
        """
        Parameters
        ----------
        X:
            Array containing pairs of embedding evaluations.
        y:
            Training targets which should be measures of the prediction uncertainty (e.g. MSE or RMSE).
        """
        X_transformed = self.poly.fit_transform(X)
        self.model.fit(X_transformed, y)

    def predict(self, X: np.ndarray):
        """
        Parameters
        ----------
        X:
            Array containing pairs of embedding evaluations.
        """
        X_transformed = self.poly.transform(X)
        return self.model.predict(X_transformed)

    def save(self, filepath):
        """Save the model to filepath.
        """

        # Extract the model's parameters
        model_params = {
            "coef": self.model.coef_.tolist(),  # Convert numpy array to list for JSON serialization
            "intercept": self.model.intercept_.tolist() if hasattr(self.model.intercept_, "tolist") else self.model.intercept_,
            "degree": self.degree,
            "min_degree": self.poly._min_degree,
            "max_degree": self.poly._max_degree,
            "n_output_features_": self.poly.n_output_features_,
            "_n_out_full": self.poly._n_out_full,
        }

        # Export to JSON
        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(model_params, f)


def compute_error_predictions(
        embedding_evaluations_1: np.ndarray,
        embedding_evaluations_2: np.ndarray,
        linear_model: LinearModel):
    """Compute the error predicted by a linear_model for pairs of embeddings.
    """
    n_samples_1 = embedding_evaluations_1.shape[0]
    n_samples_2 = embedding_evaluations_2.shape[0]
    predictions = np.zeros((n_samples_1, n_samples_2))

    for i in range(n_samples_1):
        X_pair = np.hstack([np.tile(embedding_evaluations_1[i], n_samples_2).reshape(-1, 1), embedding_evaluations_2])

        # Predict using the linear model
        prediction = linear_model.predict(X_pair)

        # Store the prediction
        predictions[i, :] = prediction

    return predictions