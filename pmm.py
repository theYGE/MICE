import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal


def pmm(original_data, target_column, donors=5):
    """
    Perform Predictive Mean Matching (PMM) imputation on the specified target column of the original DataFrame.

    Parameters:
        original_data (pd.DataFrame): The original dataset with missing values.
        target_column (str): The name of the column with missing values to be imputed.
        donors (int): The number of closest observed values to draw from during imputation.

    Returns:
        pd.DataFrame: A DataFrame with imputed values for the specified target column.
    """

    # Convert DataFrame to NumPy array
    y = original_data[target_column].to_numpy()
    observed_y = ~np.isnan(y)  # Boolean array for observed values
    x = original_data.drop(columns=[target_column]).to_numpy()  # Predictor variables

    # Identify missing values
    missing_y = np.isnan(y)  # Boolean array for missing values

    # Fit the linear regression model using observed data
    model = LinearRegression().fit(x[observed_y], y[observed_y])

    # Get the coefficients and intercept from the fitted model
    coef = model.coef_
    intercept = model.intercept_

    # Calculate residuals and residual variance
    residuals = y[observed_y] - model.predict(x[observed_y])
    residual_variance = np.var(residuals, ddof=1)

    # Calculate the covariance matrix for the coefficients
    XtX_inv = np.linalg.inv(np.dot(x[observed_y].T, x[observed_y]))
    coef_cov = residual_variance * XtX_inv

    # Draw coefficients from a multivariate normal distribution
    coef_sampled = multivariate_normal.rvs(mean=coef, cov=coef_cov)

    # Predict values for observed data using the original fitted coefficients
    yhatobs = np.dot(x[observed_y], coef) + intercept

    # Predict values for missing data using the sampled coefficients
    yhatmis = np.dot(x[missing_y], coef_sampled) + intercept



    # Calculate distances between predicted values for missing and observed
    distances = cdist(yhatmis.reshape(-1, 1), yhatobs.reshape(-1, 1), metric='euclidean')

    # Find the closest observed values (donors) for each missing value
    idx = np.apply_along_axis(lambda row: np.random.choice(np.argsort(row)[:donors]), 1, distances)

    # Get the actual observed values based on the calculated indices
    observed_values = y[observed_y]
    imputed = observed_values[idx]

    # Create a new DataFrame with imputed values
    imputed_data = original_data.copy()
    imputed_data.loc[missing_y, target_column] = imputed  # Replace missing values with imputed values

    # TODO: Returned before and used in printing method
    # return imputed_data.copy(), pd.Series(imputed, index=original_data.index[missing_y]).copy()
    return imputed_data.loc[:, target_column]
    # 5. Calculatye predicted values for observed and missing Y:
    #     1. Use b hat for observed Y
    #     2. Use b star for missing Y
    # 6. For each case where Y is missing, find 3 closest predicted values where Y is observed (from Y we predicted for observed cases)
    # 7. Draw randomly one of these 3 close cases and Impute missing Yi with the observed value of the close case
    # Suggestions: https://statisticsglobe.com/predictive-mean-matching-imputation-method/


# Example usage
if __name__ == "__main__":
    # Example data
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, 7, 8, 9],
        'C': [np.nan, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)

    # Impute missing values in column 'C' with 2 donors
    imputed_df, imputed_values = pmm(df, 'C', donors=4)

    # Show imputed DataFrame
    print("DataFrame with imputed values:")
    print(imputed_df)

    boys = pd.read_csv("boys.csv", index_col=0, header=0)
    imputed_df, imputed_values = pmm(boys[['bmi', 'age']], 'bmi')

    print("DataFrame with imputed values:")
    print(imputed_df)
    print(imputed_values)
    print("END!")
# import sys
# sys.modules[__name__] = pmm
