import numpy as np
import scipy.stats as sp
from scipy.special import expit
from scipy.optimize import minimize

SMALL = np.finfo(float).eps


__all__ = ['logistic_regression']


def _logr_statistics(independent_vars, regression_coefficients):
    """Computes the significance tests for regression"""
    # Compute the probability for the current parameters
    kernel = regression_coefficients[None, :] @ independent_vars
    eval_sigmoid = expit(kernel)
    eval_sigmoid *= (1. - eval_sigmoid)

    full_matrix = independent_vars * np.sqrt(eval_sigmoid)
    hessian = full_matrix @ full_matrix.T

    standard_error = np.sqrt(np.diag(np.linalg.pinv(hessian)))
    
    # Two-Tailed test
    z_values = regression_coefficients / standard_error
    p_values = sp.norm.sf(np.abs(z_values)) * 2
    coefficient_tests = list(zip(z_values, p_values))
    
    # Odds Ratio
    odds_ratio = np.exp(regression_coefficients[1:])
    
    # Confidence Intervals
    scalar = sp.norm.isf(0.025)
    confidence_intervals = np.array([regression_coefficients - scalar * standard_error,
                                     regression_coefficients + scalar * standard_error])
    confidence_intervals = np.exp(confidence_intervals[:, 1:])

    output = {'Standard Errors': standard_error,
              'Coefficient Z-Tests': coefficient_tests,
              'Odds Ratio': odds_ratio,
              'Odds Ratio 95th CI': confidence_intervals}

    return output


def _min_func(params, independent_var, dependent_var, true_mask):
    """Minimum function for logistic regression."""
    intercept, slopes = params[0], params[1:]
    kernel = slopes[None, :] @ independent_var + intercept
    
    probability_one = expit(kernel).squeeze()
    probability_zero = 1. - probability_one
    
    # Return negative since this is going into a minimization function
    return (np.sum(np.log(probability_one[true_mask] + SMALL)) +
            np.sum(np.log(probability_zero[~true_mask] + SMALL))) * -1


def logistic_regression(independent_vars, dependent_var):
    """Computes a logistic regression.

    Args:
        independent_vars: [n_vars x n_observations], array of independent variables
        dependent_var:  Binary output variable (Coded as 0 and 1)

    Returns:
        results_dictionary: Dictionary with parameters and statistics

    Note:
        Missing data (marked by nan) is removed from all data.
    """
    independent_vars = np.atleast_2d(independent_vars)
    dependent_var = dependent_var.squeeze()

    valid_mask_dependent = ~np.isnan(dependent_var)
    valid_mask_independent = ~np.isnan(independent_vars)
    valid_mask = valid_mask_dependent & np.all(valid_mask_independent, axis=0)

    independent_vars = independent_vars[:, valid_mask]
    dependent_var = dependent_var[valid_mask]

    # Make sure dependent_y is coded as 0, 1
    if((dependent_var.min() != 0) or dependent_var.max() != 1 or
       (np.unique(dependent_var).size != 2)):
        raise AssertionError("Dependent Variable must be binary, (0 or 1)!")
    
    # Normalize inputs
    independent_means = independent_vars.mean(axis=1)
    independent_stds = independent_vars.std(axis=1, ddof=1)
    normalized_xs = ((independent_vars - independent_means[: , None]) 
                     / independent_stds[:, None])

    # Perform the minimization
    x0 = np.ones(independent_vars.shape[0] + 1)
    true_mask = dependent_var == 1
    results = minimize(_min_func, x0, args=(normalized_xs, dependent_var, true_mask),
                       method='SLSQP')

    # Convert back to un-normalized valuess
    regression_coefficients = np.zeros((independent_vars.shape[0] + 1))
    regression_coefficients[1:] = results.x[1:] / independent_stds
    regression_coefficients[0] = results.x[0] - (regression_coefficients[1:] 
                                                 * independent_means).sum()
    
    # Account for intercept
    independent_vars = np.vstack((np.ones_like(independent_vars[0]), 
                                  independent_vars))
    
    # Compute Standard Errors
    output = {'Regression Coefficients': regression_coefficients}
    statistics = _logr_statistics(independent_vars, regression_coefficients)
    output.update(statistics)
    
    # Deviance
    output['Residual Deviance'] = (dependent_var.size - independent_vars.shape[0], 
                                   2 * results.fun)
    
    # Deviance with no parameters
    n0 = np.count_nonzero(dependent_var==0)
    n1 = np.count_nonzero(dependent_var==1)
    ratio = n0 / n1
    output['Null Deviance'] = (dependent_var.size - 1,
                               2 * (n1 * np.log1p(ratio) + n0 * np.log1p(1. / ratio)))

    return output