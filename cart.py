import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder


def preprocess_features(X):
    """Preprocess features: handle categorical variables with one-hot encoding."""
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    # One-hot encoding for categorical variables
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(drop='first', sparse=False)
        X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
        X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
        X = pd.concat([X[numerical_cols].reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)

    return X


def fit_cart_model(X_train, y_train):
    """Fit a CART model (regressor or classifier) based on the target variable type."""
    if pd.api.types.is_numeric_dtype(y_train):
        model = DecisionTreeRegressor()
    else:
        model = DecisionTreeClassifier()

    model.fit(X_train, y_train)
    return model


def cart_impute(data, target_col):
    """Perform CART imputation for a specified target column in the dataset."""
    # Identify rows with missing values in the target column
    missing_rows = data[data[target_col].isna()].index

    # Create a training dataset without missing values in the target column
    train_data = data.dropna(subset=[target_col])

    # Define features and target variable
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]

    # Preprocess features (handle categorical variables)
    X_train = preprocess_features(X_train)

    # Fit the CART model
    cart_model = fit_cart_model(X_train, y_train)

    # Prepare data for missing rows
    X_missing = data.loc[missing_rows].drop(columns=[target_col])
    X_missing = preprocess_features(X_missing)

    # Predict the missing values using the fitted CART model
    predicted_values = cart_model.predict(X_missing)

    # Impute the missing values in the original DataFrame
    data.loc[missing_rows, target_col] = predicted_values

    return data


if __name__ == "__main__":
    # Example 1: Numeric target variable
    np.random.seed(123)  # For reproducibility

    # Create a sample dataset for numeric target variable
    data_numeric = pd.DataFrame({
        'feature1': np.random.randn(100),  # Continuous feature
        'feature2': np.random.rand(100),    # Continuous feature
        'target': np.random.randn(100)      # Numeric target
    })

    # Introduce missing values in the target
    data_numeric.loc[np.random.choice(data_numeric.index, size=20, replace=False), 'target'] = np.nan

    # Impute missing values for the numeric target
    imputed_data_numeric = cart_impute(data_numeric.copy(), 'target')

    print("Imputed Data (Numeric Target):")
    print(imputed_data_numeric.head())

    # Example 2: Categorical target variable
    # Create a sample dataset for categorical target variable
    data_categorical = pd.DataFrame({
        'feature1': np.random.randn(100),                       # Continuous feature
        'feature2': np.random.choice(['A', 'B', 'C'], size=100),  # Categorical feature
        'target': np.random.choice(['X', 'Y', 'Z'], size=100)  # Categorical target
    })

    # Introduce missing values in the target
    data_categorical.loc[np.random.choice(data_categorical.index, size=15, replace=False), 'target'] = np.nan

    # Impute missing values for the categorical target
    imputed_data_categorical = cart_impute(data_categorical.copy(), 'target')

    print("\nImputed Data (Categorical Target):")
    print(imputed_data_categorical.head())
