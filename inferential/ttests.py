import numpy as np
import scipy.stats as sp


__all__ = ['one_sample_ttest', 'equal_variance_ttest', 'unequal_variance_ttest',
           'repeated_ttest']


def _mean_std_size_for_group(dataset):
    """Returns the mean, std, and size for a dataset."""
    the_mean = np.nanmean(dataset)
    the_std = np.nanstd(dataset, ddof=1)
    the_size = np.count_nonzero(~np.isnan(dataset))
    
    return the_mean, the_std, the_size


def _p_value_and_confidence_intervals(t_value, deg_of_freedom, tailed):
    """Returns the p_value and confidence intervals scalars."""

    if tailed.lower() == 'left':
        p_value = sp.t.sf(t_value, deg_of_freedom)
        confidence_interval_scalar = sp.t.isf(0.05, deg_of_freedom)
        confidence_interval = np.array([-np.inf, confidence_interval_scalar])

    elif tailed.lower() == 'right':
        p_value = 1 - sp.t.sf(t_value, deg_of_freedom)
        confidence_interval_scalar = sp.t.isf(0.05, deg_of_freedom)
        confidence_interval = np.array([-confidence_interval_scalar, np.inf])

    elif tailed.lower() == 'two':
        p_value = sp.t.sf(np.abs(t_value), deg_of_freedom) * 2
        confidence_interval_scalar = sp.t.isf(0.025, deg_of_freedom)
        confidence_interval = np.array([-confidence_interval_scalar, 
                                        confidence_interval_scalar])

    else:
        raise ValueError(f"Invalid tailed option {tailed}, choose "
                          "between 'left', 'right' or 'two")

    return p_value, confidence_interval


def one_sample_ttest(group1, mean, tailed="Two"):
    """Computes a ttest hypothesis for one sample.

    Args:
        group1:  Sample group
        mean: the hypothetical mean of the population
        tailed: (String) [('Two') | 'Left' | 'Right] Defines two-tailed or 
                left/right one tailed t-test                    

    Returns:
        tests_structure: structure with ttest 

    Note:
        Doesn't include values marked as np.nan
    """
    mean1, std1, valid_size = _mean_std_size_for_group(group1)
    deg_of_freedom = valid_size - 1

    cohen_d = (mean1 - mean) / std1
    t_metric = cohen_d * np.sqrt(valid_size)

    # Inference
    p_value, ci_scalar = _p_value_and_confidence_intervals(t_metric, deg_of_freedom, tailed)
    ci_scalar *= std1 / np.sqrt(valid_size)

    cohen_d = (mean1 - mean) / std1
    confidence_interval = mean1 + ci_scalar

    output = {'Mean1': mean1, 'Std1': std1, 'n1': valid_size,
              'Mean2': mean, 'Std2': 0, 'n2': 1,
              'T_value': t_metric, 'P_value': p_value, '95th CI': confidence_interval,
              'Cohen_d': cohen_d,
              'df': deg_of_freedom}

    return output


def equal_variance_ttest(group1, group2, tailed="Two"):
    """Computes the t-test for two groups.

    assumes equal_variance between the groups

    Args:
        group1:  First group for comparison
        group2:  Second group for comparison
        tailed: (String) [('Two') | 'Left' | 'Right] Defines two-tailed or 
                left/right one tailed t-test

    Returns:
        structure of ttest values

    """
    mean1, std1, valid_size1 = _mean_std_size_for_group(group1)
    mean2, std2, valid_size2 = _mean_std_size_for_group(group2)

    deg_of_freedom = valid_size1 + valid_size2 - 2

    pooled_var = std1**2 * (valid_size1 - 1) + std2**2 * (valid_size2 - 1)
    pooled_var /= deg_of_freedom

    standard_error = np.sqrt(pooled_var * (1. / valid_size1 + 1. / valid_size2))
    t_metric = (mean1 - mean2) / standard_error

    # Inference
    p_value, ci_scalar = _p_value_and_confidence_intervals(t_metric, deg_of_freedom, tailed)
    ci_scalar *= standard_error
    cohen_d = (mean1 - mean2) / np.sqrt(pooled_var)
    confidence_interval = (mean1 - mean2) + ci_scalar

    output = {'Mean1': mean1, 'Std1': std1, 'n1': valid_size1,
              'Mean2': mean2, 'Std2': std2, 'n2': valid_size2,
              'T_value': t_metric, 'P_value': p_value, '95th CI': confidence_interval,
              'Cohen_d': cohen_d, 'df': deg_of_freedom}

    return output


def unequal_variance_ttest(group1, group2, tailed="Two"):
    """Computes the t-test for two groups

    Doesn't assume two groups have equal variance. Uses
    Welch's method

    Args:
        group1:  First group for comparison
        group2:  Second group for comparison
        tailed: (String) [('Two') | 'Left' | 'Right] Defines two-tailed or 
                left/right one tailed t-test

    Returns:
        structure of ttest values

    """
    mean1, std1, valid_size1 = _mean_std_size_for_group(group1)
    mean2, std2, valid_size2 = _mean_std_size_for_group(group2)

    variance1 = std1**2 / valid_size1
    variance2 = std2**2 / valid_size2
    weighted_variance = variance1 + variance2

    deg_of_freedom = np.square(weighted_variance)
    deg_of_freedom /= (np.square(variance1) / (valid_size1 - 1) +
                       np.square(variance2) / (valid_size2 - 1))

    t_metric = (mean1 - mean2) / np.sqrt(weighted_variance)

    # Inference
    p_value, ci_scalar = _p_value_and_confidence_intervals(t_metric, deg_of_freedom, tailed)
    ci_scalar *= np.sqrt(weighted_variance)
    cohen_d = (mean1 - mean2) * np.sqrt(2) / np.sqrt(std1**2 + std2**2)
    confidence_interval = (mean1 - mean2) + ci_scalar

    output = {'Mean1': mean1, 'Std1': std1, 'n1': valid_size1,
              'Mean2': mean2, 'Std2': std2, 'n2': valid_size2,
              'T_value': t_metric, 'P_value': p_value, '95th CI': confidence_interval,
              'Cohen_d': cohen_d, 'df': deg_of_freedom}

    return output


def repeated_ttest(group1, group2, tailed="Two"):
    """Computes the t-test for two groups

    assumes equal_variance between the groups and repeated measures

    Args:
        group1:  First group for comparison
        group2:  Second group for comparison
        tailed: (String) [('Two') | 'Left' | 'Right] Defines two-tailed or 
                left/right one tailed t-test

    Returns:
        structure of ttest values
    """
    if group1.size != group2.size:
        raise AssertionError("Repteated TTest groups must be the same size.")

    mean1, std1, _ = _mean_std_size_for_group(group1)
    mean2, std2, _ = _mean_std_size_for_group(group2)

    pairs = group1 - group2
    mean, std, valid_size = _mean_std_size_for_group(pairs)
    deg_of_freedom = valid_size - 1

    standard_error = std / np.sqrt(valid_size)
    t_metric = mean / standard_error

    # Inference
    p_value, ci_scalar = _p_value_and_confidence_intervals(t_metric, deg_of_freedom, tailed)
    ci_scalar *= standard_error
    cohen_d = mean / std
    confidence_interval = mean + ci_scalar

    output = {'Mean1': mean1, 'Std1': std1, 'n1': valid_size,
              'Mean2': mean2, 'Std2': std2, 'n2': valid_size,
              'T_value': t_metric, 'P_value': p_value, '95th CI': confidence_interval,
              'Cohen_d': cohen_d, 'df': deg_of_freedom}

    return output