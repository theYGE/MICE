import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm


def logreg_impute(original_data, target_column):
    """
    Perform logistic regression-based imputation on a binary target column of the original DataFrame.

    Parameters:
        original_data (pd.DataFrame): The original dataset with missing values.
        target_column (str): The name of the binary column with missing values to be imputed.

    Returns:
        pd.DataFrame: A DataFrame with imputed values for the specified binary target column.
    """

    # Convert the target column to a NumPy array
    y = original_data[target_column].to_numpy()

    # Create a boolean mask for observed values
    ry = ~np.isnan(y)

    # Drop the target column to create the predictor matrix
    x = original_data.drop(columns=[target_column]).to_numpy()

    # Ensure that the target column is binary
    unique_values = np.unique(y[ry])
    if len(unique_values) != 2:
        raise ValueError("The target column must be binary.")

    # Fit a logistic regression model using the observed data
    model = LogisticRegression(solver='liblinear')
    model.fit(x[ry], y[ry])

    # Get the coefficients and intercept from the fitted model
    beta = model.coef_.flatten()  # Coefficients for the features
    intercept = model.intercept_[0]  # Intercept

    # Compute the variance-covariance matrix approximation (inverse of the Fisher Information Matrix)
    X_ry = np.c_[np.ones(x[ry].shape[0]), x[ry]]  # Add intercept to predictor matrix
    p = model.predict_proba(x[ry])[:, 1]
    W = np.diag(p * (1 - p))  # Diagonal matrix of variances

    # Add a small constant to the diagonal for numerical stability (regularization)
    ridge_penalty = 1e-6
    XtWX_inv = np.linalg.pinv(np.dot(X_ry.T, np.dot(W, X_ry)) + ridge_penalty * np.eye(
        X_ry.shape[1]))  # Covariance matrix approximation with regularization

    # Get the diagonal of the covariance matrix
    cov_diag = np.diag(XtWX_inv)

    # Calculate the perturbed coefficients using valid standard deviations
    beta_star = np.hstack([intercept, beta]) + norm.rvs(scale=np.sqrt(np.clip(cov_diag, a_min=0, a_max=None)))

    # Add a column of ones to the missing data matrix to account for the intercept
    x_missing = np.c_[np.ones(np.sum(~ry)), x[~ry]]

    # Calculate probabilities for the missing values using the perturbed coefficients
    p_missing = 1 / (1 + np.exp(-np.dot(x_missing, beta_star)))

    # Impute the missing values by sampling from a Bernoulli distribution
    y_imp = np.random.binomial(1, p_missing)

    # Fill in the missing values with the imputed values
    y[~ry] = y_imp

    # Return the original DataFrame with the imputed target column
    original_data[target_column] = y
    return original_data


# Example usage
if __name__ == "__main__":
    # Example data
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [1, 4, 3, 2, 1],
        'C': [0, 1, np.nan, 0, 1]  # Binary column with missing values
    }
    df = pd.DataFrame(data)

    # Impute missing values in column 'C'
    imputed_df = logreg_impute(df, 'C')

    # Show the DataFrame with imputed values
    print("DataFrame with imputed values:")
    print(imputed_df)
