import pandas as pd
import numpy as np
from typing import Dict, Callable, List, Type, Tuple
import statsmodels.api as sm
import numeric_imputations  # Import numeric imputation functions
import categorical_imputations  # Import categorical imputation functions


class MICE:
    def __init__(self, data: pd.DataFrame, column_types: Dict[str, Type] = None, num_imputations: int = 5,
                 num_iterations: int = 10):
        """
        Initialize the MICE class.

        Args:
            data (pd.DataFrame): The dataset with missing values.
            column_types (Dict[str, Type]): Dictionary with column names as keys and column types as values.
            num_imputations (int): Number of multiple imputations to perform.
            num_iterations (int): Number of iterations for each imputation.
        """
        self.data = data
        self.num_imputations = num_imputations
        self.num_iterations = num_iterations
        self.imputed_data = []
        self.inferred_types = {}

        # Infer column types if not provided
        if column_types is None:
            self.column_types = self._infer_column_types(data)
        else:
            self.column_types = column_types

    def _infer_column_types(self, data: pd.DataFrame) -> Dict[str, Type]:
        """
        Infer the types of the columns in the data.

        Args:
            data (pd.DataFrame): The dataset with missing values.

        Returns:
            Dict[str, Type]: A dictionary with column names as keys and inferred types as values.
        """
        inferred_types = {}
        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                inferred_types[column] = np.number
            elif pd.api.types.is_categorical_dtype(data[column]) or data[column].dtype == object:
                inferred_types[column] = 'category'
            else:
                inferred_types[column] = 'unknown'
        self.inferred_types = inferred_types
        return inferred_types

    def modify_column_types(self, new_column_types: Dict[str, Type]):
        """
        Modify the types of columns based on user input.

        Args:
            new_column_types (Dict[str, Type]): Dictionary with column names as keys and new types as values.
        """
        self.column_types.update(new_column_types)

    def impute(self):
        """
        Perform multiple imputations using chained equations.
        """
        # Iterate over the number of imputations
        for i in range(self.num_imputations):
            imputed_data = self.data.copy()

            # Iterate over the specified number of iterations
            for j in range(self.num_iterations):
                # Iterate over each column in the dataframe
                for column in imputed_data.columns:
                    if self.column_types[column] == np.number:
                        # Apply numeric imputation
                        imputed_data[column] = numeric_imputations.impute_numeric(imputed_data, column)
                    elif self.column_types[column] == 'category':
                        # Apply categorical imputation
                        imputed_data[column] = categorical_imputations.impute_categorical(imputed_data, column)
                    else:
                        print(f"Skipping column '{column}' due to unknown type.")

            # Store the imputed dataset
            self.imputed_data.append(imputed_data)

        # After imputation, print the inferred types
        print("Inferred variable types used for imputation:")
        for column, col_type in self.inferred_types.items():
            print(f"Column '{column}': Assumed type '{col_type}'")

        return self.imputed_data

    def fit_models(self, formula: str):
        """
        Fit a regression model on each imputed dataset.

        Args:
            formula (str): The regression formula to use (e.g., 'dependent ~ independent1 + independent2').
        """
        self.models = []
        self.results = []

        for dataset in self.imputed_datasets:
            model = sm.OLS.from_formula(formula, data=dataset).fit()
            self.models.append(model)
            self.results.append(model.params)

    def pool_parameters(self) -> pd.DataFrame:
        """
        Perform parameter pooling using Rubin's rules.

        Returns:
            pd.DataFrame: DataFrame with pooled parameter estimates, standard errors, and other statistics.
        """
        # Combine parameter estimates
        params = pd.concat(self.results, axis=1)
        params.columns = [f'Model_{i}' for i in range(len(self.results))]

        # Calculate means and variances
        mean_params = params.mean(axis=1)
        within_var = params.var(axis=1)
        between_var = params.apply(lambda x: np.var(x, ddof=1), axis=1)

        total_var = within_var + (1 + 1 / self.num_imputations) * between_var

        # Calculate confidence intervals and p-values
        pooled_results = pd.DataFrame({
            'Estimate': mean_params,
            'Std_Error': np.sqrt(total_var),
            'CI_Lower': mean_params - 1.96 * np.sqrt(total_var),
            'CI_Upper': mean_params + 1.96 * np.sqrt(total_var),
            'P_Value': 2 * (1 - sm.stats.norm.cdf(np.abs(mean_params / np.sqrt(total_var))))
        })

        return pooled_results

    def get_imputed_datasets(self) -> List[pd.DataFrame]:
        """
        Get the list of imputed datasets.

        Returns:
            List[pd.DataFrame]: List of imputed datasets.
        """
        return self.imputed_datasets


# Example usage:
if __name__ == "__main__":
    # Example DataFrame with missing values
    df = pd.DataFrame({
        'age': [25, 30, None, 45, None],  # Numeric column
        'income': [50000, 60000, None, None, 75000],  # Numeric column
        'gender': ['M', None, 'F', 'F', None]  # Categorical column
    })

    # Dictionary of column names and their types ('numeric' or 'categorical')
    column_types = {
        'age': 'numeric',
        'income': 'numeric',
        'gender': 'categorical'
    }

    # Create MICE object
    mice = MICE(data=df, column_types=column_types, num_imputations=5, num_iterations=10)

    # Perform imputation
    mice.perform_imputation()

    # Fit models
    formula = 'income ~ age + gender'
    mice.fit_models(formula)

    # Perform parameter pooling
    pooled_results = mice.pool_parameters()

    # Print pooled results
    print(pooled_results)
