import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


def multinomial_logistic_impute(data, target_col):
    """
    Impute missing values in the target column using multinomial logistic regression.

    Parameters:
    - data: pandas DataFrame containing the dataset with features and target variable.
    - target_col: int, the index of the target column in the DataFrame.

    Returns:
    - data: pandas DataFrame with imputed values in the target column.
    """
    # Copy the original DataFrame to avoid modifying it
    data = data.copy()

    # Separate features (X) and target (y)
    y = data.iloc[:, target_col]  # Target variable (categorical)
    X = data.drop(columns=[data.columns[target_col]])  # Features

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create dummy variables for categorical columns
    original_data_dummies = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # One-hot encode categorical features in X
    # encoder = OneHotEncoder(sparse=False)
    X_encoded = original_data_dummies

    # Identify observed and missing values in the target column
    observed_indices = ~pd.isna(y)

    # Separate observed data
    y_obs = y[observed_indices]
    X_obs = X_encoded[observed_indices]

    # Fit the multinomial logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_obs, y_obs)

    # Predict for missing values
    X_missing = X_encoded[~observed_indices]
    preds = model.predict_proba(X_missing)  # Get predicted probabilities

    # Sample from the predicted probabilities to impute missing values
    y_imputed = []
    for pred in preds:
        # Randomly select a category based on the predicted probabilities and impute
        y_imputed.append(np.random.choice(y_obs.unique(), p=pred))

    # Fill in the missing values in the original data
    data.loc[~observed_indices, data.columns[target_col]] = y_imputed

    return data.iloc[:, target_col]

    # Returned before
    # return data


if __name__ == "__main__":
    # Sample dataset with missing values
    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5, 6, 7],
        'x2': [2, 3, 1, 2, 4, 1, 3],
        'y': ['A', 'B', 'C', np.nan, 'A', np.nan, 'B']
    })

    # Apply the imputation function
    imputed_data = multinomial_logistic_impute(data, target_col=2)

    # Print the imputed dataset
    print(imputed_data)
