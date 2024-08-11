import pandas as pd
import numpy as np

# TODO: Define set of arguments
def sri():

    # Stochastic regression imputation is basically regression imputation where noise is added

    # Algorithm:
    #     1. Fit a regresstion model on Y observed
    #     2. Calculate residuals: ei = yi observed - yi hat (predicted)
    #     3. Assume residuals are distributed with Normal distribution with mean 0 and sigma = residual.std() (standard deviation)
    #     4. For each missing Y predict Y hat from regression
    #     5. Y imputed = Y predicted + error
    #     6. Y imputed = Y predicted + normal(0, sigma, size = Y.pred.shape)
    #     7. error is randomly drawn from the normal distribution

    #TODO: Should I do Bayesian multiple imputation or bootstrap multiple imputation on top?


    # TODO: Define return value. Imputed dataset?
    return None