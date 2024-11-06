import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
# from fancyimpute import IterativeImputer  # MICE from fancyimpute
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load the airquality dataset
from sklearn.datasets import fetch_openml
from mice import MICE
from pmm import pmm
from scipy import stats  # Importing stats for t-distribution
from stochastic_regression_imputation import sri
from cart import cart_impute
from mice import *



# Function to generate MAR data with desired missingness in one column based on another
def generate_mar_data(data, target_col, condition_cols, weights=None, intercept=-5, missing_rate=0.2):
    """
    Generate MAR missingness in the target column based on specified conditions using logistic regression.

    Parameters:
    - data (pd.DataFrame): The original DataFrame.
    - target_col (str): The column where missing values will be introduced.
    - condition_cols (list): The columns used to determine missingness.
    - weights (list or np.array): Weights for the condition columns in the logistic regression model.
                                  If None, weights will be set to a small value for each condition column.
    - intercept (float): The intercept for the logistic regression model, controlling the overall missingness level.
    - missing_rate (float): Desired proportion of missing values (e.g., 0.2 for 20%).

    Returns:
    - pd.DataFrame: DataFrame with MAR missingness introduced.
    """
    # Copy the data to avoid modifying the original DataFrame
    data_mar = data.copy()

    # Define the predictors (condition columns)
    X = data_mar[condition_cols]
    X = (X - X.mean()) / X.std()

    # Set default weights if none are provided
    if weights is None:
        weights = np.array([0.2, 0.9] )

    # Calculate the probabilities of missingness using the logistic function
    logits = intercept + np.dot(X, weights)
    prob_missing = 1 / (1 + np.exp(-logits))

    # Normalize the probabilities to ensure the correct total number of missing values
    n = len(data_mar)
    target_missing_count = int(n * missing_rate)

    # Select indices for the top probabilities to match the target missing count
    if target_missing_count > 0:
        missing_indices = np.argsort(prob_missing)[-target_missing_count:]  # Get the highest probabilities
    else:
        missing_indices = []

    # Set the target column to NaN for the selected indices
    data_mar.iloc[missing_indices, data_mar.columns.get_loc(target_col)] = np.nan


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

# Function to perform the simulation with MICE imputation
def run_simulation(data, n_sim=500, sample_size = 1000, missing_rate=0.2, mechanism='MCAR'):

    simulation_results = []
    for i in range(n_sim):
        print("Running simulation", i)
        sample_data = resample(data, n_samples=sample_size, replace=True)

        if mechanism == "MAR": # Draw a bootstrap sample
            missing_data = generate_mar_data(sample_data, target_col="age", condition_cols=['height', 'weight'], missing_rate=missing_rate)
        if mechanism == "MCAR":
            missing_data = simulate_mcar(sample_data, target_col="age", missing_rate=missing_rate)

        # Imputation using MICE (fancyimpute)\
        #TODO: Use MICE for Imputation

        # nhanes_no_weight = missing_data.drop(columns=['weight'])
        # features = ['height', 'age', 'sex', 'race']
        #
        #         # Encode categorical features
        # X = pd.get_dummies(df[features], drop_first=True)
        # predictor_matrix = pd.DataFrame({
        #     'height': [0, 1, 1, 1],
        #     'age': [1, 0, 1, 1],
        #     'sex': [1, 1, 0, 1],
        #     'race': [1, 1, 1, 0],
        # }, index=['height', 'age', 'sex', 'race'])
        missing_data.dropna(inplace=True)
        mice = MICE(missing_data, num_imputations=1, num_iterations=0)
        imputed_data = mice.impute()

        # Step 4: Estimate linear regression coefficients on imputed datasets
        results = []
        for df in imputed_data:
            # Add back the original 'Ozone' column and filter rows where 'Ozone' is not missing

            # Define features (X) and target (y)
            features = ['height', 'age', 'sex', 'race']

            X = pd.get_dummies(df[features], drop_first=True)
            y = df['weight']  # Target variable

            # Initialize and fit the Linear Regression model
            model = LinearRegression()
            model.fit(X, y)

            # Extract the model parameters (coefficients and intercept)
            coefficients = model.coef_

            # Store the results in a dictionary
            result = {
                'age': coefficients[1],
                "mean_of_age": np.mean(df["age"]),
                "height": coefficients[0]
            }

            # Append results to the list
            results.append(result)

        # Step 5: Pool results and calculate averages
        mice.results = pd.DataFrame(results)
        pooled_results = mice.pool_parameters()  # Use your pooling function

        # Output pooled results
        # print(pooled_results)
        simulation_results.append(pooled_results)

    sum_age_coefficient = 0
    sum_age_coef__CI_length = 0
    sum_age_coef__CI_coverage = 0
    sum_age_coef_variance = 0
    sum_age_coef_mse = 0

    sum_mean_age = 0
    sum_mean_age_CI_length = 0
    sum_mean_age_CI_coverage = 0
    sum_mean_age_variance = 0
    sum_mean_age_mse = 0

    # Iterate over each DataFrame in the list
    for df in simulation_results:
        # Extract the estimate for age coefficient and mean age
        age_coefficient = df.loc['age', "Estimate"]
        age_mean =df.loc['mean_of_age', "Estimate"]
        height_coefficient = df.loc['height', "Estimate"]

        sum_age_coef_mse += (age_coefficient - 0.1215)**2
        sum_mean_age_mse += (age_mean - 45.5796)**2

        # Add to the running sums
        sum_age_coefficient += age_coefficient
        sum_mean_age += age_mean

        sum_age_coef__CI_length += df.loc['age', "CI_Upper"] - df.loc['age', "CI_Lower"]
        sum_age_coef__CI_coverage += 1 if  df.loc['age', "CI_Lower"]  <= 0.1215 <= df.loc['age', "CI_Upper"] else 0
        sum_age_coef_variance += df.loc['age', "Variance"]

        sum_mean_age_CI_length += df.loc['mean_of_age', "CI_Upper"] - df.loc['mean_of_age', "CI_Lower"]
        sum_mean_age_CI_coverage += 1 if  df.loc['mean_of_age', "CI_Lower"]  <= 47.5796 <= df.loc['mean_of_age', "CI_Upper"] else 0
        sum_mean_age_variance += df.loc['mean_of_age', "Variance"]

    # Calculate the averages
    # After 500 iterations
    # avg_age_coefficient = 0.09, true is 0.1215
    # average_age_mean = 47.5736, true value is 47.5796


    avg_mean_age = sum_mean_age / n_sim
    sum_mean_age_CI_length /= n_sim
    sum_mean_age_CI_coverage /= n_sim
    sum_mean_age_variance /= n_sim
    sum_mean_age_mse /= n_sim

    avg_age_coefficient = sum_age_coefficient / n_sim
    sum_age_coef__CI_length /= n_sim
    sum_age_coef__CI_coverage /= n_sim
    sum_age_coef_variance /= n_sim
    sum_age_coef_mse /= n_sim



    print(avg_age_coefficient, avg_mean_age)


# Example usage:
if __name__ == "__main__":
    dataset = pd.read_stata("nhanes2d.dta")
    print(dataset.columns)
    print(dataset.isna().sum())
    columns_to_keep = ['height', 'age', 'weight', 'sex', 'race']
    dataset = dataset.filter(columns_to_keep)
    dataset.dropna(inplace=True)
    #
    # # Define the true values for comparison
    # true_mean_age = 47.57
    # true_age_parameter = 0.1215
    #
    # # Initialize lists to store the estimates and metrics
    # mean_ages = []
    # age_parameters = []
    # mean_se_age = []
    # se_age_parameters = []
    #
    # # Number of simulations
    # num_simulations = 500
    #
    # for i in range(num_simulations):
    #     print(f"Running simulation {i + 1}")
    #     # Resample your dataset
    #     sample_data = resample(dataset, n_samples=1000, replace=True)
    #
    #     # Generate missing data (assuming your function is defined)
    #     missing_data = generate_mar_data(sample_data, target_col="age", condition_cols=['height', 'weight'],
    #                                      missing_rate=0.2)
    #     # missing_data = simulate_mcar(sample_data, target_col="age", missing_rate=0.5)
    #     missing_data.dropna(inplace=True)
    #
    #     # Define features (X) and target (y)
    #     features = ['height', 'age', 'sex', 'race']
    #     X = pd.get_dummies(missing_data[features], drop_first=True)
    #     y = missing_data['weight']  # Target variable
    #
    #     # Initialize and fit the Linear Regression model
    #     model = LinearRegression()
    #     model.fit(X, y)
    #
    #     # Store the regression coefficients and the mean of age
    #     age_mean = np.mean(X['age'])
    #     age_coef = model.coef_[1]  # Assuming age is the second feature after dummy encoding
    #
    #     # Append the estimates to the lists
    #     mean_ages.append(age_mean)
    #     age_parameters.append(age_coef)
    #
    #     # Calculate standard errors for the current simulation
    #     # Standard error of the mean
    #     se_age = np.std(X['age'], ddof=1) / np.sqrt(len(X))
    #     mean_se_age.append(se_age)
    #
    #     # Standard error of the regression coefficient (assuming homoscedasticity)
    #     # Calculate residuals
    #     residuals = y - model.predict(X)
    #     residual_variance = np.var(residuals, ddof=1)
    #     se_age_parameter = np.sqrt(residual_variance / np.sum((X['age'] - np.mean(X['age'])) ** 2))
    #     se_age_parameters.append(se_age_parameter)
    #
    # # Convert results to DataFrame for easier handling
    # results_df = pd.DataFrame({
    #     'Mean_Age': mean_ages,
    #     'Age_Coefficient': age_parameters,
    #     'SE_Mean_Age': mean_se_age,
    #     'SE_Age_Coefficient': se_age_parameters
    # })
    #
    # # Calculate coverage and confidence intervals
    # z_critical = 1.96  # For a 95% CI
    # results_df['CI_Lower_Mean_Age'] = results_df['Mean_Age'] - z_critical * results_df['SE_Mean_Age']
    # results_df['CI_Upper_Mean_Age'] = results_df['Mean_Age'] + z_critical * results_df['SE_Mean_Age']
    #
    # results_df['CI_Lower_Age_Coefficient'] = results_df['Age_Coefficient'] - z_critical * results_df[
    #     'SE_Age_Coefficient']
    # results_df['CI_Upper_Age_Coefficient'] = results_df['Age_Coefficient'] + z_critical * results_df[
    #     'SE_Age_Coefficient']
    #
    # # Calculate coverage
    # mean_age_coverage = np.mean(
    #     (results_df['CI_Lower_Mean_Age'] <= true_mean_age) & (results_df['CI_Upper_Mean_Age'] >= true_mean_age))
    # age_coef_coverage = np.mean((results_df['CI_Lower_Age_Coefficient'] <= true_age_parameter) & (
    #             results_df['CI_Upper_Age_Coefficient'] >= true_age_parameter))
    #
    # avg_ci_length_mean = np.mean(results_df["CI_Upper_Mean_Age"] - results_df["CI_Lower_Mean_Age"])
    # avg_ci_length_coef = np.mean(results_df["CI_Upper_Age_Coefficient"] - results_df["CI_Lower_Age_Coefficient"])
    #
    # print(f"Mean Age Coverage: {mean_age_coverage * 100:.2f}%")
    # print(f"Age Coefficient Coverage: {age_coef_coverage * 100:.2f}%")




    # # True values for the analysis
    # true_mean_age = 47.57
    # true_age_coefficient = 0.1215
    #
    # # Initialize variables to store results
    # age_means = []
    # age_parameters = []
    #
    # for i in range(500):
    #     print("Running simulation", i)
    #     sample_data = resample(dataset, n_samples=1000, replace=True)
    #     missing_data = generate_mar_data(sample_data, target_col="age", condition_cols=['height', 'weight'],
    #                                      missing_rate=0.5)
    #     # missing_data = simulate_mcar(sample_data, target_col="age", missing_rate=0.5)
    #     missing_data.dropna(inplace=True)
    #
    #     # Define features (X) and target (y)
    #     features = ['height', 'age', 'sex', 'race']
    #     X = pd.get_dummies(missing_data[features], drop_first=True)
    #     y = missing_data['weight']  # Target variable
    #
    #     # Initialize and fit the Linear Regression model
    #     model = LinearRegression()
    #     model.fit(X, y)
    #
    #     # Collect the mean of 'age' and the regression coefficient for 'age'
    #     age_means.append(np.mean(X['age']))
    #     age_parameters.append(model.coef_[1])
    #
    # # Convert collected results to arrays for easier computation
    # age_means = np.array(age_means)
    # age_parameters = np.array(age_parameters)
    #
    # # Calculate metrics for 'age' mean
    # age_mean = np.mean(age_means)
    # age_variance = np.var(age_means, ddof=1)
    # age_std_error = np.sqrt(age_variance / len(age_means))
    # z_critical = 2.576  # For a 95% CI
    # age_ci_lower = age_mean - z_critical * age_std_error
    # age_ci_upper = age_mean + z_critical * age_std_error
    # age_ci_length = age_ci_upper - age_ci_lower
    # age_bias = age_mean - true_mean_age
    # age_relative_bias = age_bias / true_mean_age
    # age_mse = np.mean((age_means - true_mean_age) ** 2)
    # age_coverage = 1 if (true_mean_age >= age_ci_lower and true_mean_age <= age_ci_upper) else 0
    #
    # # Calculate metrics for 'age' regression parameter
    # age_param_mean = np.mean(age_parameters)
    # age_param_variance = np.var(age_parameters, ddof=1)
    # age_param_std_error = np.sqrt(age_param_variance / len(age_parameters))
    # age_param_ci_lower = age_param_mean - z_critical * age_param_std_error
    # age_param_ci_upper = age_param_mean + z_critical * age_param_std_error
    # age_param_ci_length = age_param_ci_upper - age_param_ci_lower
    # age_param_bias = age_param_mean - true_age_coefficient
    # age_param_relative_bias = age_param_bias / true_age_coefficient
    # age_param_mse = np.mean((age_parameters - true_age_coefficient) ** 2)
    # age_param_coverage = 1 if (
    #             true_age_coefficient >= age_param_ci_lower and true_age_coefficient <= age_param_ci_upper) else 0
    #
    # # Output results
    # print(f"Age Mean Estimate: {age_mean}")
    # print(f"Age Mean Variance: {age_variance}")
    # print(f"Age Mean Std Error: {age_std_error}")
    # print(f"Age Mean 95% CI: ({age_ci_lower}, {age_ci_upper})")
    # print(f"Age Mean Bias: {age_bias}")
    # print(f"Age Mean Relative Bias: {age_relative_bias}")
    # print(f"Age Mean MSE: {age_mse}")
    # print(f"Age Mean CI Length: {age_ci_length}")
    # print(f"Age Mean Coverage: {age_coverage}\n")
    #
    # print(f"Age Regression Coefficient Estimate: {age_param_mean}")
    # print(f"Age Regression Coefficient Variance: {age_param_variance}")
    # print(f"Age Regression Coefficient Std Error: {age_param_std_error}")
    # print(f"Age Regression Coefficient 95% CI: ({age_param_ci_lower}, {age_param_ci_upper})")
    # print(f"Age Regression Coefficient Bias: {age_param_bias}")
    # print(f"Age Regression Coefficient Relative Bias: {age_param_relative_bias}")
    # print(f"Age Regression Coefficient MSE: {age_param_mse}")
    # print(f"Age Regression Coefficient CI Length: {age_param_ci_length}")
    # print(f"Age Regression Coefficient Coverage: {age_param_coverage}")





    result = run_simulation(data=dataset, n_sim=500, missing_rate=0.5, mechanism="MAR")





    mcar_20 = simulate_mcar(data=dataset, target_col="age", missing_rate=0.2)
    mcar_50 = simulate_mcar(data=dataset, target_col="age", missing_rate=0.5)

    mar_20 = simulate_mar_with_target(data=dataset, target_col="age", condition_col="height", missing_rate=0.2)
    mar_50 = simulate_mar_with_target(data=dataset, target_col="age", condition_col="height", missing_rate=0.5)


    print("Simulation finished")