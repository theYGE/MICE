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
import matplotlib.pyplot as plt


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
        weights = np.array([0.2, 0.7] )

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
            missing_data = generate_mar_data(sample_data, target_col="sex", condition_cols=['height', 'weight'], missing_rate=missing_rate)
        if mechanism == "MCAR":
            missing_data = simulate_mcar(sample_data, target_col="sex", missing_rate=missing_rate)

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
        # missing_data.dropna(inplace=True)
        mice = MICE(missing_data)
        imputed_data = mice.impute()

        # Step 4: Estimate linear regression coefficients on imputed datasets
        results = []
        for df in imputed_data:
            # Add back the original 'Ozone' column and filter rows where 'Ozone' is not missing
            # df["weight"] = missing_data["weight"]

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
                'sex': coefficients[2],
                "mean_of_sex": X['sex_Female'].mean(),
            }

            # Append results to the list
            results.append(result)

        # Step 5: Pool results and calculate averages
        mice.results = pd.DataFrame(results)
        pooled_results = mice.pool_parameters()  # Use your pooling function

        # Output pooled results
        # print(pooled_results)
        simulation_results.append(pooled_results)

    sum_sex_coefficient = 0
    sum_sex_coef__CI_length = 0
    sum_sex_coef__CI_coverage = 0
    sum_sex_coef_variance = 0
    sum_sex_coef_mse = 0

    sum_mean_sex = 0
    sum_mean_sex_CI_length = 0
    sum_mean_sex_CI_coverage = 0
    sum_mean_sex_variance = 0
    sum_mean_sex_mse = 0

    # Iterate over each DataFrame in the list
    for df in simulation_results:
        # Extract the estimate for age coefficient and mean age
        sex_coefficient = df.loc['sex', "Estimate"]
        sex_mean =df.loc['mean_of_sex', "Estimate"]

        sum_sex_coef_mse += (sex_coefficient - (-1.706))**2
        sum_mean_sex_mse += (sex_mean - 0.5251)**2

        # Add to the running sums
        sum_sex_coefficient += sex_coefficient
        sum_mean_sex += sex_mean

        sum_sex_coef__CI_length += df.loc['sex', "CI_Upper"] - df.loc['sex', "CI_Lower"]
        sum_sex_coef__CI_coverage += 1 if  df.loc['sex', "CI_Lower"]  <= -1.706 <= df.loc['sex', "CI_Upper"] else 0
        sum_sex_coef_variance += df.loc['sex', "Variance"]

        sum_mean_sex_CI_length += df.loc['mean_of_sex', "CI_Upper"] - df.loc['mean_of_sex', "CI_Lower"]
        sum_mean_sex_CI_coverage += 1 if  df.loc['mean_of_sex', "CI_Lower"]  <= 0.5251 <= df.loc['mean_of_sex', "CI_Upper"] else 0
        sum_mean_sex_variance += df.loc['mean_of_sex', "Variance"]

    # Calculate the averages
    # After 500 iterations
    # True Proportion Female = 52.51
    # True female coef = -1.706


    avg_mean_sex = sum_mean_sex / n_sim
    sum_mean_sex_CI_length /= n_sim
    sum_mean_sex_CI_coverage /= n_sim
    sum_mean_sex_variance /= n_sim
    sum_mean_sex_mse /= n_sim

    # 2% bias
    avg_sex_coefficient = sum_sex_coefficient / n_sim
    sum_sex_coef__CI_length /= n_sim
    sum_sex_coef__CI_coverage /= n_sim
    sum_sex_coef_variance /= n_sim
    sum_sex_coef_mse /= n_sim



    print(avg_sex_coefficient, avg_mean_sex)


# Example usage:
if __name__ == "__main__":
    dataset = pd.read_stata("nhanes2d.dta")
    print(dataset.columns)
    print(dataset.isna().sum())
    columns_to_keep = ['height', 'age', 'weight', 'sex', 'race']
    dataset = dataset.filter(columns_to_keep)
    dataset.dropna(inplace=True)

    # Define the true values for comparison
    true_mean_female = 0.5251
    true_female_parameter = -1.706

    # Initialize lists to store the estimates and metrics
    mean_females = []
    female_parameters = []
    mean_se_female = []
    se_female_parameters = []

    # Number of simulations
    num_simulations = 500

    for i in range(num_simulations):
        print(f"Running simulation {i + 1}")
        # Resample your dataset
        sample_data = resample(dataset, n_samples=1000, replace=True)

        # Generate missing data (assuming your function is defined)
        missing_data = generate_mar_data(sample_data, target_col="sex", condition_cols=['height', 'weight'],
                                         missing_rate=0.5)
        # missing_data = simulate_mcar(sample_data, target_col="sex", missing_rate=0.5)
        missing_data.dropna(inplace=True)

        # Define features (X) and target (y)
        features = ['height', 'age', 'sex', 'race']
        X = pd.get_dummies(missing_data[features], drop_first=True)
        y = missing_data['weight']  # Target variable

        # Initialize and fit the Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        # Store the regression coefficients and the mean of age
        female_mean = np.mean(X['sex_Female'])
        female_coef = model.coef_[2]  # Assuming age is the second feature after dummy encoding

        # Append the estimates to the lists
        mean_females.append(female_mean)
        female_parameters.append(female_coef)

        # Calculate standard errors for the current simulation
        # Standard error of the mean
        se_female = np.std(X['sex_Female'], ddof=1) / np.sqrt(len(X))
        mean_se_female.append(se_female)

        # Standard error of the regression coefficient (assuming homoscedasticity)
        # Calculate residuals
        residuals = y - model.predict(X)
        residual_variance = np.var(residuals, ddof=1)
        se_female_parameter = np.sqrt(residual_variance / np.sum((X['sex_Female'] - np.mean(X['sex_Female'])) ** 2))
        se_female_parameters.append(se_female_parameter)

    # Convert results to DataFrame for easier handling
    results_df = pd.DataFrame({
        'Mean_Female': mean_females,
        'Female_Coefficient': female_parameters,
        'SE_Mean_Female': mean_se_female,
        'SE_Female_Coefficient': se_female_parameters
    })

    # Calculate coverage and confidence intervals
    z_critical = 1.96  # For a 95% CI
    results_df['CI_Lower_Mean_Female'] = results_df['Mean_Female'] - z_critical * results_df['SE_Mean_Female']
    results_df['CI_Upper_Mean_Female'] = results_df['Mean_Female'] + z_critical * results_df['SE_Mean_Female']

    results_df['CI_Lower_Female_Coefficient'] = results_df['Female_Coefficient'] - z_critical * results_df[
        'SE_Female_Coefficient']
    results_df['CI_Upper_Female_Coefficient'] = results_df['Female_Coefficient'] + z_critical * results_df[
        'SE_Female_Coefficient']

    # Calculate coverage
    mean_female_coverage = np.mean(
        (results_df['CI_Lower_Mean_Female'] <= true_mean_female) & (results_df['CI_Upper_Mean_Female'] >= true_mean_female))
    female_coef_coverage = np.mean((results_df['CI_Lower_Female_Coefficient'] <= true_female_parameter) & (
                results_df['CI_Upper_Female_Coefficient'] >= true_female_parameter))

    avg_ci_length_mean = np.mean(results_df["CI_Upper_Mean_Female"] - results_df["CI_Lower_Mean_Female"])
    avg_ci_length_coef = np.mean(results_df["CI_Upper_Female_Coefficient"] - results_df["CI_Lower_Female_Coefficient"])

    print(f"Mean Age Coverage: {mean_age_coverage * 100:.2f}%")
    print(f"Age Coefficient Coverage: {age_coef_coverage * 100:.2f}%")




    # # True values for the analysis
    # true_mean_female = 0.5251
    # true_female_coefficient = -1.706
    #
    # # Initialize variables to store results
    # female_means = []
    # female_parameters = []
    #
    # for i in range(500):
    #     print("Running simulation", i)
    #     sample_data = resample(dataset, n_samples=1000, replace=True)
    #     missing_data = generate_mar_data(sample_data, target_col="sex", condition_cols=['height', 'weight'],
    #                                      missing_rate=0.5)
    #     # missing_data = simulate_mcar(sample_data, target_col="sex", missing_rate=0.5)
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
    #     female_means.append(np.mean(X['sex_Female']))
    #     female_parameters.append(model.coef_[2])
    #
    # # Convert collected results to arrays for easier computation
    # female_means = np.array(female_means)
    # female_parameters = np.array(female_parameters)
    #
    # # Calculate metrics for 'age' mean
    # female_mean = np.mean(female_means)
    # female_variance = np.var(female_means, ddof=1)
    # female_bias = female_mean - true_mean_female
    # female_relative_bias = female_bias / true_mean_female
    # female_mse = np.mean((female_means - true_mean_female) ** 2)
    #
    # # Calculate metrics for 'age' regression parameter
    # female_param_mean = np.mean(female_parameters)
    # female_param_variance = np.var(female_parameters, ddof=1)
    # female_param_bias = female_param_mean - true_female_coefficient
    # female_param_relative_bias = female_param_bias / true_female_coefficient
    # female_param_mse = np.mean((female_parameters - true_female_coefficient) ** 2)
    #
    # # Output results
    # print("done")


    # True Proportion Female = 52.51
    # True female coef = -1.706

    # female_proportion = 0
    # female_parameter = model.coef_[2]
    # for i in range(500):
    #     print("Running simulation", i)
    #     sample_data = resample(dataset, n_samples=1000, replace=True)
    #     # missing_data = generate_mar_data(sample_data, target_col="sex", condition_cols=['height', 'weight'], missing_rate=0.5)
    #     missing_data = simulate_mcar(sample_data, target_col="sex", missing_rate=0.5)
    #     missing_data.dropna(inplace=True)
    #
    #     features = ['height', 'age', 'sex', 'race']
    #     X = pd.get_dummies(missing_data[features], drop_first=True)
    #     female_proportion += X['sex_Female'].mean()
    #
    # female_proportion /= 500
    # print(age_mean)


        # missing_data = simulate_mcar(sample_data, target_col="age", missing_rate=0.2)
        # for _ in range(20):
        #     sample_data['age'] = sri(missing_data, 'age')

        # Define features (X) and target (y)




    # proportion_female = (dataset['sex'] == 'Female').mean()
    # print(f"Proportion of Female in the whole dataset: {proportion_female}")
    #
    # features = ['height', 'age', 'sex', 'race']
    #
    # X = pd.get_dummies(dataset[features], drop_first=True)
    # y = dataset['weight']  # Target variable
    #
    # # Initialize and fit the Linear Regression model
    # model = LinearRegression()
    # model.fit(X, y)
    # # True female coef = -1.706
    # female_parameter = model.coef_[2]

    # proportion_female = 0
    # female_parameter = 0
    # # #
    # for i in range(1000):
    #     print("Running simulation", i)
    #     sample_data = resample(dataset, n_samples=1000, replace=True)
    #     # missing_data = simulate_mcar(sample_data, target_col="sex", missing_rate=0.5)
    #     # for _ in range(20):
    #     #     sample_data['age'] = sri(missing_data, 'age')
    #     missing_data = generate_mar_data(sample_data, target_col="sex", condition_cols=['height', 'weight'], missing_rate=0.2)
    #
    #     # Define features (X) and target (y)
    #     features = ['height', 'age', 'sex', 'race']
    #
    #     X = pd.get_dummies(missing_data[features], drop_first=True)
    #     y = missing_data['weight']  # Target variable
    #
    #     p_female = X['sex_Female'].mean()
    #     proportion_female += p_female
    #
    #     # Initialize and fit the Linear Regression model
    #     model = LinearRegression()
    #     model.fit(X, y)
    #     female_parameter += model.coef_[2]
    # female_parameter = female_parameter / 1000 # -1.71
    # average_proportion_female = proportion_female / 1000 # 0.5249




    result = run_simulation(data=dataset, n_sim=500, missing_rate=0.5, mechanism="MAR")





    mcar_20 = simulate_mcar(data=dataset, target_col="age", missing_rate=0.2)
    mcar_50 = simulate_mcar(data=dataset, target_col="age", missing_rate=0.5)

    mar_20 = simulate_mar_with_target(data=dataset, target_col="age", condition_col="height", missing_rate=0.2)
    mar_50 = simulate_mar_with_target(data=dataset, target_col="age", condition_col="height", missing_rate=0.5)


    print("Simulation finished")