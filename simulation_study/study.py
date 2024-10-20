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
from pmm import pmm
from scipy import stats  # Importing stats for t-distribution
from stochastic_regression_imputation import sri
from cart import cart_impute



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
    # z_score = 1.96  # For 95% confidence interval (normal distribution approximation)
    # z_score = 1.645 # for 90% confidence interval

    t_score = stats.t.ppf((1 + confidence) / 2, df=n - 1)  # Two-tailed

    y_pred = np.mean(y_pred)
    y_true = np.mean(y_true)
    ci_lower = y_pred - t_score * se
    ci_upper = y_pred + t_score * se
    ci_length = ci_upper - ci_lower

    # Check coverage: whether true values fall within the CI
    coverage =  1 if ci_lower <= y_true <= ci_upper else 0

    return ci_length, coverage


# def calculate_variance_ci(true_var, imputed_var, n, confidence=0.95):
#     se = np.sqrt((2 * (imputed_var ** 2)) / (n - 1))
#     t_score = stats.t.ppf((1 + confidence) / 2, df=n - 1)
#
#     ci_lower = imputed_var - t_score * se
#     ci_upper = imputed_var + t_score * se
#     ci_length = ci_upper - ci_lower
#
#     # Check coverage: whether the true variance falls within the CI
#     coverage = 1 if ci_lower <= true_var <= ci_upper else 0
#
#     return ci_length, coverage


# def get_true_age_estimate(data):
#     """Fit a linear regression model to estimate age based on other columns."""
#     # Select relevant features
#     features = ['height', 'age', 'sex', 'race']
#
#     # Encode categorical features
#     X = pd.get_dummies(data[features], drop_first=True)
#     y = data['weight']
#
#     # Create and fit the model
#     model = LinearRegression()
#     model.fit(X, y)
#
#     age_index = X.columns.get_loc('age')
#
#     # Get the true estimates of age
#     return model.coef_[age_index]


# def get_mice_age_estimate(data):
#     nhanes_no_weight = data.drop(columns=['weight'])
#
#     mice = MICE(nhanes_no_weight, num_imputations=1, num_iterations=1)
#     imputed_data = mice.impute()
#
#     results = []
#     for i, df in enumerate(imputed_data):
#         df["weight"] = data["weight"]
#
#         # 1. Filter rows where 'Ozone' is not missing
#         # df_filtered["Ozone"] = np.log(df_filtered["Ozone"])
#
#         # 2. Define features (X) and target (y)
#         features = ['height', 'age', 'sex', 'race']
#
#         # Encode categorical features
#         X = pd.get_dummies(df[features], drop_first=True)
#         y = df['weight']  # Target variable
#
#         # 3. Initialize and fit the Linear Regression model
#         model = LinearRegression()
#         model.fit(X, y)
#
#         # 4. Extract the model parameters (coefficients and intercept)
#         coefficients = model.coef_
#         intercept = model.intercept_
#
#         # 5. Store the parameters in a dictionary with feature names
#         age_index = X.columns.get_loc('age')
#
#         result = {
#                   'age': model.coef_[age_index]
#                   }
#
#         # Append the result dictionary to the results list
#         results.append(result)
#
#     # 6. Convert the results list to a DataFrame
#     df_results = pd.DataFrame(results)
#     mice.results = df_results
#     pooled_results = mice.pool_parameters()
#     age_estimate = pooled_results['Estimate'][0]
#     return age_estimate


def calculate_ci_length_and_coverage(simulated_means, true_mean, confidence_level=0.95):
    """
    Calculate the CI length and coverage for the mean estimate across multiple simulations.

    Parameters:
    - simulated_means (array-like): An array of mean estimates from multiple simulations.
    - true_mean (float): The true mean value for comparison.
    - confidence_level (float): The desired confidence level (default is 0.95).

    Returns:
    - ci_length (float): The length of the confidence interval.
    - coverage (float): The proportion of simulations where the true mean is within the confidence interval.
    """
    # Number of simulations
    n_simulations = len(simulated_means)

    # Calculate the standard deviation of the simulated means
    std_dev = np.std(simulated_means, ddof=1)  # ddof=1 for sample standard deviation

    # Calculate the standard error (SE)
    se = std_dev / np.sqrt(n_simulations)

    # Calculate the critical value for the t-distribution
    critical_value = stats.t.ppf((1 + confidence_level) / 2, df=n_simulations - 1)

    # Calculate the CI length
    ci_length = 2 * (critical_value * se)

    # Calculate the lower and upper bounds for each simulation's CI
    lower_bounds = simulated_means - (critical_value * se)
    upper_bounds = simulated_means + (critical_value * se)

    # Calculate coverage: proportion of times the true mean falls within the CIs
    coverage = np.mean((lower_bounds <= true_mean) & (true_mean <= upper_bounds))

    return ci_length, coverage

# def calculate_ci_length_and_coverage(true_value, estimates, confidence_level=0.95):
#     """Calculate average CI length and coverage for regression coefficients over multiple simulations."""
#
#     # Calculate critical value for the given confidence level
#     alpha = 1 - confidence_level
#     z_alpha_over_2 = stats.norm.ppf(1 - alpha / 2)  # Z-value for the two-tailed test
#
#     # Placeholder for storing lengths and coverage results
#     ci_lengths = []
#     coverage_count = 0
#
#     # Calculate CI length and check coverage for each estimate
#     for estimate in estimates:
#         # Assuming the standard error (SE) is provided or calculated previously
#         se = np.std(estimates) / np.sqrt(len(estimates))  # Example calculation of SE
#
#         # Calculate the CI
#         lower_bound = estimate - z_alpha_over_2 * se
#         upper_bound = estimate + z_alpha_over_2 * se
#
#         # Calculate CI length
#         ci_length = upper_bound - lower_bound
#         ci_lengths.append(ci_length)
#
#         # Check if true value is within the CI
#         if lower_bound <= true_value <= upper_bound:
#             coverage_count += 1
#
#     # Calculate average CI length and coverage rate
#     avg_ci_length = np.mean(ci_lengths)
#     coverage_rate = coverage_count / len(estimates)
#
#     return avg_ci_length, coverage_rate


# Function to perform the simulation with MICE imputation
def run_simulation(data, n_sim=500, sample_size = 1000, missing_rate=0.2, mechanism='MCAR'):
    complete_case_mean_value = data["age"].mean()  # Mean calculation
    # complete_case_variance_value = data["age"].var()
    # complete_case_column = data["age"]


    mean_results, bias_results, mse_results = [], [], []
    variance_results, variance_bias_results, variance_mse_results = [], [], []
    age_results, age_bias_results, age_mse_results = [], [], []

    ci_lengths, coverages = [], []
    variance_ci_lengths, variance_coverages = [], []
    age_ci_lengths, age_coverages = [], []

    relative_bias_results = []
    variance_relative_bias_results = []
    age_relative_bias_results = []


    true_mean_results, true_variance_results = [], []
    # true_age_estimate = get_true_age_estimate(data)



    for i in range(n_sim):
        print("Running simulation", i)
        sample_data = resample(data, n_samples=sample_size, replace=True)
        # complete_case_column = sample_data["age"]
        # complete_case_mean_value = complete_case_column.mean()
        # complete_case_variance_value = complete_case_column.var()


        # true_mean_results.append(complete_case_mean_value)
        # true_variance_results.append(complete_case_variance_value)

        # test_age = get_true_age_estimate(sample_data)


        if mechanism == "MAR": # Draw a bootstrap sample
            missing_data = simulate_mar_with_target(sample_data, target_col="age", condition_col="height", missing_rate=missing_rate)
        if mechanism == "MCAR":
            missing_data = simulate_mcar(sample_data, target_col="age", missing_rate=missing_rate)

        # Imputation using MICE (fancyimpute)\
        #TODO: Use MICE for Imputation
        nhanes_no_weight = missing_data.drop(columns=['weight'])
        # mice = MICE(nhanes_no_weight, num_imputations=5, num_iterations=10)
        # imputed_data = mice.impute()

        # Fit regression model for evaluation (using Temp as dependent variable)
        # X = imputed_data[:, :-1]  # independent variables
        # y = imputed_data[:, -1]  # dependent variable (e.g., Temp)
        #
        # model = LinearRegression().fit(X, y)
        # y_pred = model.predict(X)


        # Calculate metrics
        column_after_imputation = pmm(nhanes_no_weight, "age")


        # mice_age_estimate = get_mice_age_estimate(missing_data)
        # age_results.append(mice_age_estimate)


        # column_after_imputation = sri(nhanes_no_weight, "age")
        # column_after_imputation = cart_impute(nhanes_no_weight, "age")
        imputed_mean = np.mean(column_after_imputation)
        mean_results.append(imputed_mean)
        mse_results.append(mean_squared_error([complete_case_mean_value], [imputed_mean]))
        #
        # Calculate the mean of the imputed column
        # Calculate bias for the mean (mean of imputed column - mean of complete case column)
        bias = imputed_mean - complete_case_mean_value
        # Append mean bias to results
        bias_results.append(bias)
        #
        # Calculate relative bias
        relative_bias = bias / complete_case_mean_value if complete_case_mean_value != 0 else np.nan  # Avoid division by zero
        relative_bias_results.append(relative_bias)

        # # Calculate CI length and coverage
        # ci_length, coverage = calculate_ci(complete_case_column, column_after_imputation)
        # ci_lengths.append(np.mean(ci_length))
        # coverages.append(coverage)

        # Calculate variance metrics
        # imputed_variance = np.var(column_after_imputation)
        # variance_results.append(imputed_variance)
        # variance_bias = imputed_variance - complete_case_variance_value
        # variance_bias_results.append(variance_bias)
        # variance_relative_bias = variance_bias / complete_case_variance_value if complete_case_variance_value != 0 else np.nan
        # variance_relative_bias_results.append(variance_relative_bias)
        # variance_mse = mean_squared_error([complete_case_variance_value], [imputed_variance])
        # variance_mse_results.append(variance_mse)

        # variance_ci_length, variance_coverage = calculate_variance_ci(complete_case_variance_value, imputed_variance,
        #                                                               n=sample_size)
        # variance_ci_lengths.append(variance_ci_length)
        # variance_coverages.append(variance_coverage)


    # After the loop, calculate the final bias
    final_bias = np.mean(bias_results)
    print("Final Mean Bias over 500 iterations:", final_bias)
    # After the loop, calculate the final relative bias
    final_relative_bias = np.mean(relative_bias_results)
    print("Final Mean Relative Bias over 500 iterations:", final_relative_bias)
    # After running the simulation loop
    final_mean = np.mean(mean_results)  # Final mean of imputed values across all simulations
    print(f"Final Mean: {final_mean}")
    # final_true_mean = np.mean(true_mean_results)  # Final mean of imputed values across all simulations
    print(f"Real True Mean: {complete_case_mean_value}")
    final_mse = np.mean(mse_results)  # Final MSE across all simulations
    print(f"Final MSE: {final_mse}")
    true_mean = complete_case_mean_value
    population_std_dev = np.std(mean_results)
    standard_error = population_std_dev / np.sqrt(n_sim)
    z_score = 1.96
    ci_lower = true_mean - z_score * standard_error
    ci_upper = true_mean + z_score * standard_error
    coverage = 0
    # for i in range(n_sim):
    #     if ci_lower[i] <= true_mean <= ci_upper[i]:
    #         coverage += 1
    # coverage = coverage / n_sim
    coverage_count = sum(1 for mean in mean_results if ci_lower <= true_mean <= ci_upper)
    coverage_proportion = coverage_count / n_sim

    # ci_length, coverage = calculate_ci_length_and_coverage(mean_results, complete_case_mean_value)
    # print(f"Final Mean CI Length: {ci_length}")
    # print(f"Final Coverage: {coverage}")

    # final_variance_bias = np.mean(variance_bias_results)
    # print(f"Final Variance Bias: {final_variance_bias}")
    # final_variance_relative_bias = np.mean(variance_relative_bias_results)
    # print(f"Final Variance Relative Bias: {final_variance_relative_bias}")
    # final_variance_mean = np.mean(variance_results)
    # print(f"Final Variance Mean: {final_variance_mean}")
    # final_variance_true_mean = np.mean(true_variance_results)
    # print(f"Final True Variance Mean: {final_variance_true_mean}")
    # final_variance_mse = np.mean(variance_mse_results)
    # print(f"Final Variance MSE: {final_variance_mse}")
    # final_variance_ci_length = np.mean(variance_ci_lengths)
    # print(f"Final Variance CI Length: {final_variance_ci_length}")
    # final_variance_coverage = np.mean(variance_coverages)
    # print(f"Final Variance Coverage: {final_variance_coverage}")


    # final_age_mean = np.mean(age_results)
    # print(f"Final Age Mean: {final_age_mean}")
    # print(f"Final True Age Coef: {true_age_estimate}")
    # final_age_bias = final_age_mean - true_age_estimate
    # print(f"Final Age Bias: {final_age_bias}")
    # final_age_relative_bias = final_age_bias / true_age_estimate
    # print(f"Final Age Relative Bias: {final_age_relative_bias}")
    # age_mse = np.mean((age_results - true_age_estimate) ** 2)
    # print(f"Final Age MSE: {age_mse}")
    # age_ci_length, age_coverage = calculate_ci_length_and_coverage(true_age_estimate, age_results)
    # print(f"Final Age CI Length: {age_ci_length}")
    # print(f"Final Age Coverage: {age_coverage}")


    # Calculate the standard error of the mean (SE)
    # standard_error = np.std(age_results, ddof=1) / np.sqrt(len(age_results))
    #
    # # Calculate the 95% confidence interval using the standard error
    # lower_bound = final_age_mean - 1.96 * standard_error
    # upper_bound = final_age_mean + 1.96 * standard_error
    #
    # # Calculate the CI length
    # ci_length = upper_bound - lower_bound
    #
    # # Count the number of estimates within the confidence intervals
    # within_bounds = (age_results >= lower_bound) & (age_results <= upper_bound)
    # coverage_count = np.sum(within_bounds)
    # # Calculate coverage as a proportion
    # coverage = coverage_count / len(age_results)
    # print(f"Final Age CI Length: {ci_length}")
    # print(f"Final Age Coverage: {coverage}")


    return
    # return mean_results, bias_results, mse_results, ci_lengths, coverages, relative_bias_results




# Example usage:
if __name__ == "__main__":
    dataset = pd.read_stata("nhanes2d.dta")
    print(dataset.columns)
    print(dataset.isna().sum())
    columns_to_keep = ['height', 'age', 'weight', 'sex', 'race']
    dataset = dataset.filter(columns_to_keep)
    dataset.dropna(inplace=True)




    result = run_simulation(data=dataset, n_sim=500)

    mcar_20 = simulate_mcar(data=dataset, target_col="age", missing_rate=0.2)
    mcar_50 = simulate_mcar(data=dataset, target_col="age", missing_rate=0.5)

    mar_20 = simulate_mar_with_target(data=dataset, target_col="age", condition_col="height", missing_rate=0.2)
    mar_50 = simulate_mar_with_target(data=dataset, target_col="age", condition_col="height", missing_rate=0.5)

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