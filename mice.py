import pandas as pd
import numpy as np
from typing import Dict, Callable, List, Type, Tuple
import statsmodels.api as sm
# import numeric_imputations  # Import numeric imputation functions
# import categorical_imputations  # Import categorical imputation functions
from pmm import pmm
from cart import cart_impute
from typing import Optional

# TODO: Implement first imputation which is simply imputing using existing data
# TODO: First imputation should just sample from observed valuers -> what does that mean exactly.
# TODO: Considering now I have all columns imputed in the first iteration, I need to track where the missing values where initially
# TODO: I need a kind of a mask for missing values to now where to perform the imputation
# TODO: After that I'd go over all columns and impute them in one iteration and then I just do it over a number of iterations
# TODO: This way imputing over iteration makes sense becuase in each next iteration I use values from the previous iteration or the column we just imputed


class MICE:
    def __init__(self, data: pd.DataFrame, column_types: Dict[str, Type] = None,
                 num_imputations: int = 5, num_iterations: int = 10,
                 predictor_matrix: Optional[pd.DataFrame] = None):
        """
        Initialize the MICE class.

        Args:
            data (pd.DataFrame): The dataset with missing values.
            column_types (Dict[str, Type]): Dictionary with column names as keys and column types as values.
            num_imputations (int): Number of multiple imputations to perform.
            num_iterations (int): Number of iterations for each imputation.
            predictor_matrix (Optional[pd.DataFrame]): A matrix indicating which variables should be used for
                                                       imputing each variable. If None, use all available variables.
        """
        self.data = data
        self.num_imputations = num_imputations
        self.num_iterations = num_iterations
        self.imputed_data = []
        self.inferred_types = {}
        self.convergence_stats = {col: {'means': [], 'variances': []} for col in data.columns}

        # Infer column types if not provided
        if column_types is None:
            self.column_types = self._infer_column_types(data)
        else:
            self.column_types = column_types

        # Default predictor matrix to use all available variables if not provided
        if predictor_matrix is None:
            self.predictor_matrix = pd.DataFrame(1, index=data.columns, columns=data.columns)
        else:
            self.predictor_matrix = predictor_matrix

        # Compute the missing mask: True where data is missing, False where data is present
        self.missing_mask = self.data.isnull()

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

    def _get_sorted_columns_by_missing(self) -> List[str]:
        """
        Get the list of columns sorted by the number of missing values (ascending).

        Returns:
            List[str]: List of column names sorted by missing value count.
        """
        missing_counts = self.data.isnull().sum()
        return missing_counts[missing_counts > 0].sort_values().index.tolist()

    def impute(self):
        """
        Perform multiple imputations using chained equations.
        """
        # Get columns sorted by missing values count
        sorted_columns = self._get_sorted_columns_by_missing()

        # Iterate over the number of imputations
        for i in range(self.num_imputations):
            imputed_data = self.data.copy()

            # Initial imputation: fill missing values with sampled observed values
            for column in sorted_columns:
                if self.missing_mask[column].any():
                    observed_values = imputed_data[column].dropna()
                    imputed_data.loc[self.missing_mask[column], column] = np.random.choice(observed_values,
                                                                                           size=self.missing_mask[
                                                                                               column].sum(),
                                                                                           replace=True)
            # Iterate over the specified number of iterations
            for j in range(self.num_iterations):
                previous_imputed_data = imputed_data.copy()  # Store the previous iteration's data

                # Iterate over each column in the sorted order
                for column in sorted_columns:
                    if not self.missing_mask[column].any():
                        continue  # Skip if there are no missing values in this column

                    predictors = self.predictor_matrix.loc[column]
                    # Select columns marked as predictors (value == 1) and drop rows with missing values in predictors
                    predictor_columns = predictors[predictors == 1].index

                    # Create a copy of the data with only the missing values in the target column
                    data_for_imputation = previous_imputed_data.copy()
                    data_for_imputation[column] = self.data[column]  # Restore the original missing values

                    # Perform imputation only if there is sufficient data
                    if len(predictor_columns) > 0:
                        if self.column_types[column] == np.number:
                            # Apply numeric imputation
                            imputed_values = pmm(data_for_imputation, column,donors=5)
                            # imputed_values = numeric_imputations.impute_numeric(imputed_data, column)
                        elif self.column_types[column] == 'category':
                            # Apply categorical imputation
                            imputed_values = categorical_imputations.impute_categorical(imputed_data, column)
                        else:
                            print(f"Skipping column '{column}' due to unknown type.")
                            continue

                            # Only update the original missing values
                        imputed_data.loc[self.missing_mask[column], column] = imputed_values[
                                self.missing_mask[column]]

                        # Track convergence statistics for numeric columns
                        if self.column_types[column] == np.number:
                            mean_imputed = imputed_values.mean()
                            var_imputed = imputed_values.var()
                            self.convergence_stats[column]['means'].append(mean_imputed)
                            self.convergence_stats[column]['variances'].append(var_imputed)

            # Store the imputed dataset
            self.imputed_data.append(imputed_data)

        # After imputation, print the inferred types
        print("Inferred variable types used for imputation:")
        for column, col_type in self.inferred_types.items():
            print(f"Column '{column}': Assumed type '{col_type}'")

        return self.imputed_data

    def pool_results(self, analysis_func: callable) -> pd.DataFrame:
        """
        Pool results across all imputed datasets using the provided analysis function.

        Args:
            analysis_func (callable): A function that takes a DataFrame and returns a summary statistic (like mean).

        Returns:
            pd.DataFrame: A DataFrame containing the pooled results.
        """
        results = [analysis_func(df) for df in self.imputed_data]
        pooled_results = pd.concat(results).groupby(level=0).mean()

        return pooled_results

    def plot_convergence(self):
        """
        Plot the convergence of imputations by visualizing the mean and variance of imputed values over iterations.
        """
        for column, stats in self.convergence_stats.items():
            if len(stats['means']) > 0:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(stats['means'], marker='o')
                plt.title(f'Convergence of Mean for {column}')
                plt.xlabel('Iteration')
                plt.ylabel('Mean Imputed Value')

                plt.subplot(1, 2, 2)
                plt.plot(stats['variances'], marker='o')
                plt.title(f'Convergence of Variance for {column}')
                plt.xlabel('Iteration')
                plt.ylabel('Variance of Imputed Value')

                plt.tight_layout()
                plt.show()

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

    # Example usage:
    airquality = pd.read_csv('airquality.csv', index_col=0, header=0)
    # Columns = Ozone, Solar.R. Wind, Temp, Month, Day
    airquality.drop(columns=['Ozone'], inplace=True)

    predictor_matrix = pd.DataFrame({
        'Solar.R': [0, 1, 1, 1, 1],
        'Wind': [1, 0, 1, 1, 1],
        'Temp': [1, 1, 0, 1, 1],
        'Month': [1, 1, 1, 0, 1],
        'Day': [1, 1, 1, 1, 0],
    }, index=['Solar.R', 'Wind', 'Temp', 'Month', 'Day'])
    mice = MICE(airquality, predictor_matrix=predictor_matrix)
    imputed_data = mice.impute()  # Perform the imputation

    pooled_results = mice.pool_results(lambda df: df.mean())  # Pool results using a simple mean function

    # If needed, modify column types and re-run
    mice.modify_column_types({'age': 'category'})
    imputed_data = mice.impute()


    # # Example DataFrame with missing values
    # df = pd.DataFrame({
    #     'age': [25, 30, None, 45, None],  # Numeric column
    #     'income': [50000, 60000, None, None, 75000],  # Numeric column
    #     'gender': ['M', None, 'F', 'F', None]  # Categorical column
    # })
    #
    # # Dictionary of column names and their types ('numeric' or 'categorical')
    # column_types = {
    #     'age': 'numeric',
    #     'income': 'numeric',
    #     'gender': 'categorical'
    # }
    #
    # # Create MICE object
    # mice = MICE(data=df, column_types=column_types, num_imputations=5, num_iterations=10)
    #
    # # Perform imputation
    # mice.perform_imputation()
    #
    # # Fit models
    # formula = 'income ~ age + gender'
    # mice.fit_models(formula)
    #
    # # Perform parameter pooling
    # pooled_results = mice.pool_parameters()
    #
    # # Print pooled results
    # print(pooled_results)
