import numpy as np
from scipy.stats import studentized_range

from RyStats.inferential import (equal_variance_oneway_anova, 
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
        output matrix with:
            output[:, :, 0] = pvalues
            output[:, :, 1] = delta means
            output[:, :, 2] = lower confidence interval
            output[:, :, 3] = upper confidence interval
    """
    k_groups = len(args)
    output = np.zeros((k_groups, k_groups, 4))

    anova_output = equal_variance_oneway_anova(*args)
    ms_error = anova_output['Within']['MS']
    deg_of_freedom = anova_output['Within']['DF']

    for ndx1 in range(k_groups):
        for ndx2 in range(ndx1+1, k_groups):
            group1 = args[ndx1]
            group2 = args[ndx2]

            std_err1 = ms_error / group1.size
            std_err2 = ms_error / group2.size
            total_std_error = np.sqrt(0.5 * (std_err1 + std_err2))

            deltaX = group1.mean() - group2.mean()
            t_value = deltaX / total_std_error

            sigma_value = studentized_range.isf(0.05, 
                                                k_groups, deg_of_freedom)
            p_value = studentized_range.sf(np.abs(t_value), 
                                           k_groups, deg_of_freedom)
            output[ndx1, ndx2, 0] = p_value
            output[ndx1, ndx2, 1] = deltaX
            output[ndx1, ndx2, 2] = deltaX - sigma_value * total_std_error
            output[ndx1, ndx2, 3] = deltaX + sigma_value * total_std_error

    return output


def games_howell_posthoc(*args):
    """Post-Hoc test to compute valid differences

    Args:
        group1:  Input group 1 as a numpy array
        group2:  Input group 2 as a numpy array
        groupN:  Input group n as a numpy array

    Returns:
        output matrix with:
            output[:, :, 0] = pvalues
            output[:, :, 1] = delta means
            output[:, :, 2] = lower confidence interval
            output[:, :, 3] = upper confidence interval
    """
    k_groups = len(args)
    output = np.zeros((k_groups, k_groups, 4))

    for ndx1 in range(k_groups):
        for ndx2 in range(ndx1+1, k_groups):
            group1 = args[ndx1]
            group2 = args[ndx2]

            std_err1 = group1.var(ddof=1) / group1.size
            std_err2 = group2.var(ddof=1) / group2.size
            total_std_error = np.sqrt(0.5 * (std_err1 + std_err2))
            deg_of_freedom = (np.square(std_err1 + std_err2) /
                              (std_err1**2 / (group1.size - 1) + std_err2**2 / (group2.size - 1)))
            deltaX = group1.mean() - group2.mean()
            t_value = deltaX / total_std_error

            sigma_value = studentized_range.isf(0.05, 
                                                k_groups, deg_of_freedom)
            p_value = studentized_range.sf(np.abs(t_value), 
                                           k_groups, deg_of_freedom)
            output[ndx1, ndx2, 0] = p_value
            output[ndx1, ndx2, 1] = deltaX
            output[ndx1, ndx2, 2] = deltaX - sigma_value * total_std_error
            output[ndx1, ndx2, 3] = deltaX + sigma_value * total_std_error

    return output


def bonferonni_posthoc(*args, repeated=True):
    """Computes T-Tests between groups, should adjust your
       significance value, a.k.a. bonferonni correction to
       minimize family wide error rates

    Args:
        group1:  Input group 1 as a numpy array
        group2:  Input group 2 as a numpy array
        groupN:  Input group n as a numpy array
        repeated: Use the repeated measures T-Test

    Returns:
        output matrix with:
            output[:, :, 0] = pvalues
            output[:, :, 1] = delta means
            output[:, :, 2] = lower confidence interval
            output[:, :, 3] = upper confidence interval
    """
    k_groups = len(args)
    output = np.zeros((k_groups, k_groups, 4))
    ttest = [unequal_variance_ttest, 
             repeated_ttest][repeated]

    for ndx1 in range(k_groups):
        for ndx2 in range(ndx1+1, k_groups):
            result = ttest(args[ndx1], args[ndx2])
            deltaX = result['Mean1'] - result['Mean2']
            output[ndx1, ndx2, 0] = result['P_value']
            output[ndx1, ndx2, 1] = deltaX
            output[ndx1, ndx2, 2] = deltaX - result['scalar']
            output[ndx1, ndx2, 3] = deltaX + result['scalar']

    return output