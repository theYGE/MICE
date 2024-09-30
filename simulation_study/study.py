import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
# from fancyimpute import IterativeImputer  # MICE from fancyimpute
import matplotlib.pyplot as plt

# Load the airquality dataset
from sklearn.datasets import fetch_openml
from mice import MICE


# Function to generate MAR data with desired missingness in one column based on another
def simulate_mar_with_target(data, target_col, condition_col, missing_rate=0.2):
    """
    Introduce MAR missingness in target_col based on values in condition_col.

    Parameters:
    - data: DataFrame
    - target_col: The column where missing values will be introduced
    - condition_col: The column whose values will dictate missingness
    - missing_rate: The proportion of missing values to introduce (e.g., 0.2 for 20%, 0.5 for 50%)

    Returns:
    - Modified DataFrame with MAR missingness introduced.
    """
    data_mar = data.copy()

    # Calculate how many missing values we need based on the missing_rate
    n_missing = int(missing_rate * len(data_mar))

    # Sort by the condition column and choose the first n_missing rows to introduce missingness
    sorted_indices = data_mar[condition_col].sort_values().index

    # Introduce missingness in the target column for the first n_missing sorted rows
    data_mar.loc[sorted_indices[:n_missing], target_col] = np.nan

    # Verify the actual missing percentage
    actual_missing_rate = data_mar[target_col].isna().mean()
    print(f"Actual missing rate in {target_col}: {actual_missing_rate * 100:.2f}%")

    return data_mar


# Function to simulate additional missingness (MCAR/MAR)
def simulate_mcar(data, target_col, missing_rate=0.2):
    """
    Introduce MCAR missingness in target_col (randomly, no dependence on other columns).

    Parameters:
    - data: DataFrame
    - target_col: The column where missing values will be introduced
    - missing_rate: The proportion of missing values to introduce (e.g., 0.2 for 20%, 0.5 for 50%)

    Returns:
    - Modified DataFrame with MCAR missingness introduced.
    """
    data_mcar = data.copy()

    # Calculate how many missing values we need based on the missing_rate
    n_missing = int(missing_rate * len(data_mcar))

    # Randomly select rows to introduce missingness in the target column
    missing_indices = np.random.choice(data_mcar.index, n_missing, replace=False)

    # Introduce missingness in the target column
    data_mcar.loc[missing_indices, target_col] = np.nan

    # Verify the actual missing percentage
    actual_missing_rate = data_mcar[target_col].isna().mean()
    print(f"Actual missing rate in {target_col}: {actual_missing_rate * 100:.2f}%")

    return data_mcar


# Function to calculate CI length and coverage
def calculate_ci(y_true, y_pred, confidence=0.95):
    n = len(y_true)
    se = np.sqrt(np.var(y_pred - y_true) / n)  # Standard error of the prediction
    z_score = 1.96  # For 95% confidence interval (normal distribution approximation)

    ci_lower = y_pred - z_score * se
    ci_upper = y_pred + z_score * se
    ci_length = ci_upper - ci_lower

    # Check coverage: whether true values fall within the CI
    coverage = np.mean((y_true >= ci_lower) & (y_true <= ci_upper))

    return ci_length, coverage


# Function to perform the simulation with MICE imputation
def run_simulation(data, n_sim=500, sample_size = 1000, missing_rate=0.2, mechanism='MCAR', missing_rate=0.2):
    complete_case_mean_value = data["age"].mean()  # Mean calculation
    complete_case_variance_value = data["age"].var()
    complete_case_column = data["age"]


    mean_results, bias_results, mse_results = [], [], []
    ci_lengths, coverages = [], []
    relative_bias_results = []

    for i in range(n_sim):
        sample_data = resample(data, n_samples=sample_size, replace=True)
        if mechanism == "MAR": # Draw a bootstrap sample
            missing_data = simulate_mar_with_target(data, target_col="age", condition_col="height", missing_rate=missing_rate)
        if mechanism == "MCAR":
            missing_data = simulate_mcar(data, target_col="age", missing_rate=missing_rate)

        # Imputation using MICE (fancyimpute)\
        #TODO: Use MICE for Imputation
        nhanes_no_weight = missing_data.drop(columns=['weight'])
        mice = MICE(nhanes_no_weight, num_imputations=5, num_iterations=10)
        imputed_data = mice.impute()

        # Fit regression model for evaluation (using Temp as dependent variable)
        # X = imputed_data[:, :-1]  # independent variables
        # y = imputed_data[:, -1]  # dependent variable (e.g., Temp)
        #
        # model = LinearRegression().fit(X, y)
        # y_pred = model.predict(X)

        # Calculate metrics
        column_after_imputation = imputed_data["age"]
        mean_results.append(np.mean(column_after_imputation))
        bias_results.append(np.mean(column_after_imputation - complete_case_column))
        mse_results.append(mean_squared_error(complete_case_column, column_after_imputation))

        # Calculate relative bias
        bias = np.mean(column_after_imputation - complete_case_column)
        relative_bias = bias / np.mean(complete_case_column) if np.mean(complete_case_column) != 0 else np.nan  # Avoid division by zero
        relative_bias_results.append(relative_bias)

        # Calculate CI length and coverage
        ci_length, coverage = calculate_ci(complete_case_column, column_after_imputation)
        ci_lengths.append(np.mean(ci_length))
        coverages.append(coverage)

    return mean_results, bias_results, mse_results, ci_lengths, coverages, relative_bias_results




# Example usage:
if __name__ == "__main__":
    dataset = pd.read_stata("nhanes2d.dta")
    print(dataset.columns)
    print(dataset.isna().sum())
    dataset.dropna(inplace=True)

    mar_20 = simulate_mar_with_target(data=dataset, target_col="age", condition_col="height", missing_rate=0.2)
    mar_50 = simulate_mar_with_target(data=dataset, target_col="age", condition_col="height", missing_rate=0.5)

    mcar_20 = simulate_mcar(data=dataset, target_col="age", missing_rate=0.2)
    mcar_50 = simulate_mcar(data=dataset, target_col="age", missing_rate=0.5)

    # # Run simulation on airquality data with MCAR and 20% missingness
    # mean_res, bias_res, mse_res, ci_len_res, coverage_res = run_simulation(data.values, n_sim=500, missing_rate=0.2,
    #                                                                        mechanism='MCAR')
    #
    # # Plotting results for CI Length
    # plt.hist(ci_len_res, bins=30, alpha=0.7, label='CI Length')
    # plt.title("Distribution of CI Length (MCAR 20% Missingness - Airquality)")
    # plt.legend()
    # plt.show()
    #
    # # Plotting results for Coverage
    # plt.hist(coverage_res, bins=30, alpha=0.7, label='Coverage')
    # plt.title("Distribution of Coverage (MCAR 20% Missingness - Airquality)")
    # plt.legend()
    # plt.show()



    print("Simulation finished")