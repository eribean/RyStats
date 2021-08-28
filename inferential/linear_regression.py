import numpy as np
import scipy.stats as sp


__all__ = ["linear_regression"]


def _linear_statistics(independent_vars, dependent_var, beta_coefficients):
    """Helper function to comput the significance values for the regression."""
    n_groups = independent_vars.shape[0]
    deg_of_freedom = dependent_var.size - n_groups - 1

    # F-value
    sum_squared_total = np.power(dependent_var, 2).sum()
    sum_squared_residual = dependent_var.T @ (independent_vars.T @ beta_coefficients)
    sum_squared_error = sum_squared_total - sum_squared_residual
    mean_squared_residual = sum_squared_residual / independent_vars.shape[0]
    mean_squared_error = sum_squared_error / (dependent_var.size - independent_vars.shape[0] - 1)
    f_statistic = mean_squared_residual / mean_squared_error
    f_p_value = sp.f.sf(f_statistic, n_groups, deg_of_freedom)

    # R-squared
    r_squared = sum_squared_residual / sum_squared_total
    r_squared_adjust = 1 - mean_squared_error * (dependent_var.size - 1) / sum_squared_total

    # T-tests
    estimator = np.linalg.inv(independent_vars @ independent_vars.T)
    estimator *= mean_squared_error
    output_pstats = list()

    # Compute the standard errors

    for coeff, standard_error in zip(beta_coefficients, np.diag(estimator)):
        t_value = coeff / np.sqrt(standard_error)
        p_value = sp.t.sf(np.abs(t_value), deg_of_freedom) * 2
        output_pstats.append((t_value, p_value))


    # Pack Output
    output = {'DF': (n_groups, deg_of_freedom),
              'F_Value': f_statistic, 'FPvalue': f_p_value,
              'Coeff': output_pstats, 'RSq': r_squared,
              'RSq_adj': r_squared_adjust, 'Scalar': sp.t.isf(0.025, deg_of_freedom),
              'estimator': estimator}

    return output


def _standard_errors(output_dict, indep_mean, indep_stds, dep_std):
    """Computes the standard errors and t/p_values for intercept

    Args:
        output_dict: output dictionary from linear regression
        indep_mean:  means of independent variables
        indep_stds:  standard deviations of independent variables
        dep_std:  standard deviation of dependent variable

    Returns:
        updated dictionary with standard errors and statistics for intercept

    Notes:
        This is a piecemeal approach, refactor this to go into the above function.
    """
    total_size = np.sum(output_dict['DF']) + 1
    estimator = output_dict['estimator']
    temp_array = np.zeros((estimator.shape[0] + 1, estimator.shape[1] + 1))
    temp_array[0, 0] = 1.0 / total_size
    temp_array[1:, 1:] = estimator

    # Correction matrices
    scale = np.diag(dep_std / np.concatenate(([1], indep_stds)))
    shift = np.eye(temp_array.shape[0])
    shift[1:, 0] = -indep_mean

    estimator = scale @ temp_array @ scale
    standard_error = np.sqrt(np.diag(shift.T @ estimator @ shift))

    # Compute tolerance of intercept
    t_value = output_dict['RegIntercept'] / standard_error[0]
    p_value = sp.t.sf(np.abs(t_value), output_dict['DF'][1]) * 2

    output_dict['SE'] = standard_error.tolist()
    output_dict['RegInterceptErrors'] = (t_value, p_value)

    # Use this to plot the confidence intervals of the output
    output_dict['estimator'] = np.diag(estimator).tolist()
    return output_dict


def linear_regression(independent_vars, dependent_var):
    """Performs the linear regression via least squares.

    Args:
        independent_vars: [n_vars x n_observations] array of independent variables
        dependent_y:  dependent variable

    Returns:
        Slopes, intercept of linear equation

    Example:
        result = linear_regression((independent1, independent2, ...),
                                    dependent_y)
    """
    # Center the Dependent Variable
    dep_mean = dependent_var.mean()
    dep_std = dependent_var.std(ddof=1)
    dependnt_var = dependent_var - dep_mean
    dependnt_var *= (1. / dep_std)

    # Center the Independent Variable
    indep_mean = independent_vars.mean(axis=1)
    indep_stds = independent_vars.std(axis=1, ddof=1)
    independnt_vars = independent_vars - indep_mean[:, None]
    independnt_vars /= indep_stds[:, None]

    # Compute the coefficients
    beta_coefficients = (np.linalg.pinv(independnt_vars.T) @
                         dependnt_var[:, None]).squeeze()
    beta_coefficients = np.atleast_1d(beta_coefficients)

    regression_coefficients = beta_coefficients * dep_std / indep_stds
    regression_intercept = dep_mean - (regression_coefficients * indep_mean).sum()

    # TODO: Refactor estimator in statistics
    output = _linear_statistics(independnt_vars, dependnt_var, beta_coefficients)
    output['Beta'] = beta_coefficients.tolist()
    output['RegCoeff'] = regression_coefficients.tolist()
    output['RegIntercept'] = regression_intercept.tolist()
    output['Means'] = [0,] + indep_mean.tolist()

    # Get the standard error to account for the issue when the coeff is zero
    output = _standard_errors(output, indep_mean, indep_stds, dep_std)

    return output
    
