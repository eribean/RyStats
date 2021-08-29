import numpy as np
import scipy.stats as sp


__all__ = ["linear_regression"]


def _lr_statistics(independent_vars, dependent_var, coefficients):
    """Computes t-test and confidence intervals for linear regression."""
    n_groups = independent_vars.shape[0] - 1
    deg_of_freedom = dependent_var.size - n_groups - 1

    sum_squared_total = np.var(dependent_var) * dependent_var.size
    sum_squared_error = np.square(dependent_var 
                                  - independent_vars.T @ coefficients).sum()
    sum_squared_residual = sum_squared_total - sum_squared_error
    
    mean_squared_residual = sum_squared_residual / n_groups
    mean_squared_error = sum_squared_error / deg_of_freedom
    
    f_statistic = mean_squared_residual / mean_squared_error
    f_p_value = sp.f.sf(f_statistic, n_groups, deg_of_freedom)

    # R-Squared
    r_squared = sum_squared_residual / sum_squared_total
    r_squared_adjust = (1 - mean_squared_error 
                        * (dependent_var.size - 1) / sum_squared_total)

    # T-Tests
    estimator = np.linalg.inv(independent_vars @ independent_vars.T)
    estimator *= mean_squared_error
    standard_errors = np.sqrt(np.diag(estimator))

    # Use 2-tailed tests
    t_value = coefficients / standard_errors
    t_p_value = sp.t.sf(np.abs(t_value), deg_of_freedom) * 2
    coefficient_statistics = list(zip(t_value, t_p_value))

    # 95th Percentile confidence intervals
    scalar = sp.t.isf(0.025, deg_of_freedom)
    confidence_interval = np.array([coefficients - scalar * standard_errors,
                                    coefficients + scalar * standard_errors])

    statistics = {
        'DF': (n_groups, deg_of_freedom),
        'F_value': f_statistic, 'P_value': f_p_value,
        'Coefficient T-Tests': coefficient_statistics, 
        'RSq': r_squared, 'RSq_adj': r_squared_adjust, 
        '95th CI': confidence_interval}

    return statistics


def linear_regression(independent_vars, dependent_var):
    """Performs a least-squares linear regression.
    
    Args:
        independent_vars: [n_vars x n_observations] array of independent variables
        dependent_var:  dependent variable [n_observations]

    Returns:
        regression_dict: Dictionary with regression parameters and statistics

    Example:
        result = linear_regression(np.vstack((independent1, independent2, ...)),
                                    dependent_y)
                                
    Note:
        Missing data (marked by nan) is removed from all data.
    """
    # Clean the data if necessary
    valid_mask_dependent = ~np.isnan(dependent_var)
    valid_mask_independent = ~np.isnan(independent_vars)
    valid_mask = valid_mask_dependent & np.all(valid_mask_independent, axis=0)

    independent_vars = independent_vars[:, valid_mask]
    dependent_var = dependent_var[valid_mask]
    
    # Remove Mean and Standard Deviation
    dep_mean = dependent_var.mean()
    dep_std = dependent_var.std(ddof=1)
    
    idep_mean = independent_vars.mean(axis=1)
    idep_std = independent_vars.std(axis=1, ddof=1)
    
    # Centered - Scaled Variables
    new_dependent = (dependent_var - dep_mean) / dep_std
    new_independent = (independent_vars - idep_mean[:, None]) / idep_std[:, None]
    
    # "Beta" (Standardized) Coefficients    
    coeffs = (np.linalg.pinv(new_independent.T) @ new_dependent[:, None]).squeeze()
    coeffs = np.atleast_1d(coeffs)

    # Regression Coefficients
    ratio = dep_std / idep_std
    regression_coefficients = np.ones((coeffs.size + 1))
    regression_coefficients[1:] = ratio * coeffs
    regression_coefficients[0] = dep_mean - (regression_coefficients[1:] * idep_mean).sum()
 
    output = {'Regression Coefficients': regression_coefficients, 
              'Standard Coefficients': np.concatenate(([0], coeffs))}
    
    # Account for the intercept for statistics
    independent_vars = np.vstack((np.ones_like(dependent_var), independent_vars))   
    statistics = _lr_statistics(independent_vars, dependent_var, 
                                regression_coefficients)
    output.update(statistics)

    return output    
    