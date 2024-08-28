import pandas as pd
import numpy as np
from typing import Dict, Callable, List, Type, Tuple
import statsmodels.api as sm
import numeric_imputations  # Import numeric imputation functions
import categorical_imputations  # Import categorical imputation functions


class MICE:
    def __init__(self, data: pd.DataFrame, column_types: Dict[str, Type], num_imputations: int = 5,
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
        self.column_types = column_types
        self.num_imputations = num_imputations
        self.num_iterations = num_iterations
        self.imputed_datasets = []
        self.imputation_functions = {
            'numeric': numeric_imputations.impute_numeric,
            'categorical': categorical_imputations.impute_categorical
        }
        self.models = []
        self.results = []

    def get_imputation_function(self, column_type: Type) -> Callable[[pd.Series], pd.Series]:
        """
        Get the imputation function based on the column type.

        Args:
            column_type (Type): The type of column ('numeric' or 'categorical').

        Returns:
            Callable[[pd.Series], pd.Series]: The imputation function for the given column type.
        """
        if column_type not in self.imputation_functions:
            raise ValueError(f"No imputation function defined for column type '{column_type}'")
        return self.imputation_functions[column_type]

    def perform_imputation(self):
        """
        Perform multiple imputation by chained equations using custom imputation functions.
        """
        for _ in range(self.num_imputations):
            imputed_data = self.data.copy()

            for column, column_type in self.column_types.items():
                impute_func = self.get_imputation_function(column_type)

                for _ in range(self.num_iterations):
                    # Apply the imputation function
                    imputed_data[column] = impute_func(imputed_data[column])

            self.imputed_datasets.append(imputed_data)

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
