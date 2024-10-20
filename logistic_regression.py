import numpy as np
import pandas as pd
import statsmodels.api as sm

def bayesian_logreg_impute(original_data, target_column):
    """
    Perform Bayesian logistic regression-based imputation on a binary target column of the original DataFrame.

    Parameters:
        original_data (pd.DataFrame): The original dataset with missing values.
        target_column (str): The name of the binary column with missing values to be imputed.

    Returns:
        pd.DataFrame: A DataFrame with imputed values for the specified binary target column.
    """

    # Ensure the target column is binary
    category_mapping = original_data[target_column].dropna().unique()
    if len(category_mapping) != 2:
        raise ValueError("The target column must have exactly two categories.")

    # Create binary mapping
    category_to_binary = {category_mapping[0]: 0, category_mapping[1]: 1}

    # Convert the target column to a NumPy array
    y = original_data[target_column].map(category_to_binary)  # Map to binary
    ry = ~pd.isna(y)  # Observed values mask
    y = y.to_numpy()

    # Drop the target column to create the predictor matrix
    x = original_data.drop(columns=[target_column])

    # Automatically identify categorical columns and convert them to 'category' dtype
    categorical_cols = x.select_dtypes(include=['object', 'category']).columns.tolist()

    # Convert string categorical columns to category dtype and create dummy variables
    for col in categorical_cols:
        x[col] = x[col].astype('category')

    x = pd.get_dummies(x, columns=categorical_cols, drop_first=True)

    # Fit a logistic regression model using the observed data
    model = sm.Logit(y[ry], sm.add_constant(x[ry]))  # Add intercept
    result = model.fit(disp=0)

    # Create imputation for missing values
    x_missing = x[~ry]
    x_missing = sm.add_constant(x_missing)  # Add intercept for missing data
    p_missing = result.predict(x_missing)  # Get predicted probabilities

    # Impute missing values by sampling from a Bernoulli distribution
    y_imp = np.random.binomial(1, p_missing)
    y_imp = pd.Series(y_imp).map({0: category_mapping[0], 1: category_mapping[1]})

    y = original_data[target_column]
    # Fill in the missing values with the imputed values
    y[~ry] = y_imp

    # Return the original DataFrame with the imputed target column
    original_data[target_column] = y
    return original_data.loc[:, target_column]

# Example usage
if __name__ == "__main__":
    # Example data
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': ['cat', 'dog', 'cat', 'dog', 'cat'],  # Categorical string values
        'C': ["Female", "Male", np.nan, "Female", "Female"]  # Binary column with missing values
    }
    df = pd.DataFrame(data)

    # Impute missing values in column 'C'
    imputed_df = bayesian_logreg_impute(df, 'C')

    # Show the DataFrame with imputed values
    print("DataFrame with imputed values:")
    print(imputed_df)
