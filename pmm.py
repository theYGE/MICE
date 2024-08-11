import numpy as np
import pandas as pd

# TODO: Decide on parameters taken by the function
def pmm():

    # Algorithm:
    # 1. Estimate a linear regression where:
    #     1. Y is variable to imput
    #     2. X is the rest of dataset
    # 2. We have b hat. Now we need b star
    # 3. Draw randomly from posterior predictive distribution and produce b star
    # 4. Draw randomly (using function) from Multivariate Normal Distribution where:
    #     1. Mean is b hat
    #     2. Variance is sigma squared (variance of residuals) multiplied by variance-covariance matrix of b hat
    # 5. Calculatye predicted values for observed and missing Y:
    #     1. Use b hat for observed Y
    #     2. Use b star for missing Y
    # 6. For each case where Y is missing, find 3 closest predicted values where Y is observed (from Y we predicted for observed cases)
    # 7. Draw randomly one of these 3 close cases and Impute missing Yi with the observed value of the close case
    # Suggestions: https://statisticsglobe.com/predictive-mean-matching-imputation-method/


    # TODO: Decide on return value. Imputed Dataset?
    return None
