import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def sri(original_data, target_column):
    """
    Perform stochastic regression imputation on the target column of the data.

    Parameters:
    - data: numpy array or pandas DataFrame with shape (n_samples, n_features)
    - target_col: integer or string indicating the column index or name of the target column

    Returns:
    - data_imputed: numpy array with the missing values in the target column imputed
    """

    # Automatically identify categorical columns
    categorical_cols = original_data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create dummy variables for categorical columns
    original_data_dummies = pd.get_dummies(original_data, columns=categorical_cols, drop_first=True)
    #
    # # Convert DataFrame to NumPy array
    # y = original_data_dummies[target_column].to_numpy()


    # Convert DataFrame to NumPy array
    data = original_data_dummies.to_numpy()
    target_col_index = original_data_dummies.columns.get_loc(target_column)
    y = data[:, target_col_index]
    ry = ~np.isnan(y)  # Boolean array for observed values

    # Separate predictors (x) and the target (y)
    x = np.delete(data, target_col_index, axis=1)
    y_obs = y[ry]  # Observed values of y
    x_obs = x[ry, :]  # Observed predictor values corresponding to y

    # Fit a linear regression model to the observed data
    reg = LinearRegression().fit(x_obs, y_obs)

    # Predict the missing values
    x_missing = x[~ry, :]  # Predictor values for missing y
    y_pred = reg.predict(x_missing)  # Predicted values for missing y

    # Calculate the residual standard deviation (sigma)
    residuals = y_obs - reg.predict(x_obs)  # Residuals
    # +1 because accounting for intercept
    sigma = np.sqrt(np.sum(residuals ** 2) / (len(y_obs) - (x_obs.shape[1] + 1)))  # Using n - p

    # Add stochastic noise to the predictions
    y_imputed = y_pred + np.random.normal(0, sigma, size=y_pred.shape)

    # Fill in the missing values in the original data
    data_imputed = data.copy()
    data_imputed[~ry, target_col_index] = y_imputed

    # Convert the imputed NumPy array back to a DataFrame
    # imputed_df = pd.DataFrame(data_imputed, columns=original_data.columns)

    return data_imputed[:, target_col_index]


if __name__ == "__main__":
    # Example dataset with missing values
    data = pd.DataFrame({
        'x1': np.random.normal(5, 2, 100),
        'x2': np.random.normal(0, 1, 100),
        'y': np.random.normal(3, 2, 100)
    })

    # Introduce some missing values in y
    data.loc[np.random.choice(data.index, 20, replace=False), 'y'] = np.nan

    # Apply the stochastic regression impute function
    imputed_data = sri(data, target_column="y")

    # Convert back to a DataFrame (optional)
    imputed_data = pd.DataFrame(imputed_data, columns=data.columns)

    # Print the imputed dataset
    print(imputed_data)
