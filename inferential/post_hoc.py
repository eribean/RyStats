import numpy as np
from scipy.stats import studentized_range

from RyStats.inferential import (equal_variance_oneway_anova,
                                 equal_variance_ttest,
                                 unequal_variance_ttest,
                                 repeated_ttest)


__all__ = ['tukey_posthoc', 'games_howell_posthoc', 
           'bonferonni_posthoc']


def tukey_posthoc(*args):
    """Post-Hoc test to compute valid differences

    Args:
        group1:  Input group 1 as a numpy array
        group2:  Input group 2 as a numpy array
        groupN:  Input group n as a numpy array

    Returns:
        P_values: p_value associated with mean difference
        Delta Means: difference between the means
        Lower CI : Lower 95th Confidence Interval
        Upper CI : Upper 95th Confidence Interval
    """
    k_groups = len(args)
    p_values = np.full((k_groups, k_groups), 0.0)
    delta_x = p_values.copy()
    lower_ci = p_values.copy()
    upper_ci = p_values.copy()

    anova_output = equal_variance_oneway_anova(*args)
    ms_error = anova_output['Within']['MS']
    deg_of_freedom = anova_output['Within']['DF']

    sigma_value = studentized_range.isf(0.05, 
                                        k_groups, deg_of_freedom)    

    for ndx2 in range(k_groups):
        for ndx1 in range(ndx2+1, k_groups):
            group1 = args[ndx1]
            group2 = args[ndx2]

            std_err1 = ms_error / np.count_nonzero(~np.isnan(group1))
            std_err2 = ms_error / np.count_nonzero(~np.isnan(group2))
            total_std_error = np.sqrt(0.5 * (std_err1 + std_err2))

            deltaX = np.nanmean(group1) - np.nanmean(group2)
            t_value = deltaX / total_std_error

            p_value = studentized_range.sf(np.abs(t_value), 
                                           k_groups, deg_of_freedom)

            p_values[ndx1, ndx2] = p_value
            delta_x[ndx1, ndx2] = deltaX
            lower_ci[ndx1, ndx2] = deltaX - sigma_value * total_std_error
            upper_ci[ndx1, ndx2] = deltaX + sigma_value * total_std_error

    return {'P_value': p_values,
            'Delta Means': delta_x,
            'Lower CI': lower_ci,
            'Upper CI': upper_ci}


def games_howell_posthoc(*args):
    """Post-Hoc test to compute valid differences

    Args:
        group1:  Input group 1 as a numpy array
        group2:  Input group 2 as a numpy array
        groupN:  Input group n as a numpy array

    Returns:
        P_values: p_value associated with mean difference
        Delta Means: difference between the means
        Lower CI : Lower 95th Confidence Interval
        Upper CI : Upper 95th Confidence Interval
    """
    k_groups = len(args)
    p_values = np.full((k_groups, k_groups), 0.0)
    delta_x = p_values.copy()
    lower_ci = p_values.copy()
    upper_ci = p_values.copy()    

    for ndx2 in range(k_groups):
        for ndx1 in range(ndx2+1, k_groups):
            group1 = args[ndx1]
            group2 = args[ndx2]

            group1_size = np.count_nonzero(~np.isnan(group1))
            group2_size = np.count_nonzero(~np.isnan(group2))

            std_err1 = np.nanvar(group1, ddof=1) / group1_size
            std_err2 = np.nanvar(group2, ddof=1) / group2_size

            total_std_error = np.sqrt(0.5 * (std_err1 + std_err2))
            deg_of_freedom = (np.square(std_err1 + std_err2) 
                              / (std_err1**2 / (group1_size - 1) 
                              + std_err2**2 / (group2_size - 1)))

            deltaX = np.nanmean(group1) - np.nanmean(group2)
            t_value = deltaX / total_std_error

            sigma_value = studentized_range.isf(0.05, 
                                                k_groups, deg_of_freedom)
            p_value = studentized_range.sf(np.abs(t_value), 
                                           k_groups, deg_of_freedom)
            p_values[ndx1, ndx2] = p_value
            delta_x[ndx1, ndx2] = deltaX
            lower_ci[ndx1, ndx2] = deltaX - sigma_value * total_std_error
            upper_ci[ndx1, ndx2] = deltaX + sigma_value * total_std_error

    return {'P_value': p_values,
            'Delta Means': delta_x,
            'Lower CI': lower_ci,
            'Upper CI': upper_ci}


def bonferonni_posthoc(*args, ttest_type='equal'):
    """Computes T-Tests between groups, should adjust your
       significance value, a.k.a. bonferonni correction to
       minimize family wide error rates

    Args:
        group1:  Input group 1 as a numpy array
        group2:  Input group 2 as a numpy array
        groupN:  Input group n as a numpy array
        ttest_type: [('equal') | 'unequal' | 'repeated'] string
                    of which type of ttest to perform

    Returns:
        P_values: p_value associated with mean difference
        Delta Means: difference between the means
        Lower CI : Lower 95th Confidence Interval
        Upper CI : Upper 95th Confidence Interval
    """
    k_groups = len(args)
    ttest = {'equal': equal_variance_ttest,
             'unequal': unequal_variance_ttest, 
             'repeated': repeated_ttest}[ttest_type.lower()]

    p_values = np.full((k_groups, k_groups), 0.0)
    delta_x = p_values.copy()
    lower_ci = p_values.copy()
    upper_ci = p_values.copy()

    for ndx2 in range(k_groups):
        for ndx1 in range(ndx2+1, k_groups):
            result = ttest(args[ndx1], args[ndx2])
            deltaX = result['Mean1'] - result['Mean2']

            delta_x[ndx1, ndx2] = deltaX
            p_values[ndx1, ndx2] = result['P_value']
            lower_ci[ndx1, ndx2] = result['95th CI'][0]
            upper_ci[ndx1, ndx2] = result['95th CI'][1]

    return {'P_value': p_values,
            'Delta Means': delta_x,
            'Lower CI': lower_ci,
            'Upper CI': upper_ci}