import numpy as np
import scipy.stats as sp


__all__ = ["equal_variance_oneway_anova", "unequal_variance_oneway_anova",
           "repeated_oneway_anova"]


def equal_variance_oneway_anova(*args):
    """Determines if the means in each group are different.

    Assumes equal variances for compulation of ANOVA

    Args:
        group1:  Input group 1 as a numpy array
        group2:  Input group 2 as a numpy array
        ...
        groupN:  Input group n as a numpy array

    Returns:
        Structure with anova table values
    """
    n_groups = len(args)
    means = np.array([np.nanmean(group) for group in args])
    n_each = np.array([np.count_nonzero(~np.isnan(group)) for group in args])

    all_data = np.concatenate(args)
    all_data_size = n_each.sum()

    grand_mean = np.nanmean(all_data)
    sum_squared_total = np.nanvar(all_data) * all_data_size
    sum_squared_residual = (np.power(means - grand_mean, 2) * n_each).sum()
    sum_squared_error = sum_squared_total - sum_squared_residual

    mean_squared_residual = sum_squared_residual / (n_groups - 1)
    mean_squared_error = sum_squared_error / (all_data_size - n_groups)

    f_value = mean_squared_residual / mean_squared_error
    p_value = sp.f.sf(f_value, n_groups - 1, all_data_size - n_groups)

    output = {'Between': 
                {'DF': n_groups - 1, 
                 'SS': sum_squared_residual, 
                 'MS': mean_squared_residual},
              'Within': 
                {'DF': all_data_size - n_groups, 
                 'SS': sum_squared_error, 
                 'MS': mean_squared_error},
              'Total': 
                {'DF': all_data_size - 1, 
                 'SS': sum_squared_total},
              'F_Value': f_value, 'P_value': p_value}

    return output


def unequal_variance_oneway_anova(*args):
    """Determines if the means in each group are different.

    Alternative to typical one-way anova since it assumes
    different variances among the groups.

    Args:
        group1:  Input group 1 as a numpy array
        group2:  Input group 2 as a numpy array
        groupN:  Input group n as a numpy array

    Returns:
        Structure with anova table values
    """
    n_groups = len(args)
    means = np.array([np.nanmean(group) for group in args])
    n_each = np.array([np.count_nonzero(~np.isnan(group)) for group in args])
    
    weights = np.array([group_size / np.nanvar(group, ddof=1) 
                        for (group_size, group) in zip(n_each, args)])
    weight_sum = weights.sum()
    grand_mean = (weights * means).sum() / weight_sum

    sum_squared_between = (weights * np.square(means - grand_mean)).sum()
    mean_squared_between = sum_squared_between / (n_groups - 1)

    # The below is analagous to the error term only by
    # equating terms
    deg_of_freedom = (n_groups**2 - 1)
    deg_of_freedom /= (3. * (np.square(1. - weights / weight_sum) 
                             / (n_each - 1)).sum())

    sum_squared_error = deg_of_freedom + 2 * (n_groups - 2) / 3
    mean_squared_error = sum_squared_error / deg_of_freedom

    f_value = mean_squared_between / mean_squared_error
    p_value = sp.f.sf(f_value, n_groups - 1, deg_of_freedom)

    # Total SS
    total_size = n_each.sum() - 1
    sum_square_total = np.nanvar(np.concatenate(args), ddof=1) * total_size

    output = {'Between': 
                {'DF': n_groups - 1, 
                 'SS': sum_squared_between, 
                 'MS': mean_squared_between},
              'Within': 
                {'DF': deg_of_freedom, 
                 'SS': sum_squared_error, 
                 'MS': mean_squared_error},
              'Total': 
                {'DF': total_size, 
                 'SS': sum_square_total},
              'F_Value': f_value, 'P_value': p_value}

    return output


def repeated_oneway_anova(*args):
    """Repeated Measurements one-way anova

    Treatments cannot have missing data.

    Args:
        group1:  Condition 1
        group2:  Condition 2
        groupN:  Condition N

    Returns:
        Structure with anova table values
    """
    n_conditions = len(args)
    means = np.array([group.mean() for group in args])
    n_each = np.array([group.size for group in args])

    if np.unique(n_each).size != 1:
        raise AssertionError("Number of Subjects must be Equal in each Condition.")

    subject_means = np.array([np.mean(group) for group in zip(*args)])

    # Sum of Squares Total
    all_data = np.concatenate(args)
    grand_mean = all_data.mean()
    ss_total = all_data.var() * all_data.size
    df_total = all_data.size - 1

    # Sum of Squares of Condition
    ss_condition = (n_each * np.square(means - grand_mean)).sum()
    df_condition = (n_conditions - 1)
    ms_condition = ss_condition / df_condition

    # Sum of Squares Subjects
    ss_subjects = (n_conditions * np.square(subject_means - grand_mean).sum())
    df_subjects = int(n_each[0] - 1)

    # Sum of Squares within
    ss_within = ss_total - ss_condition
    ss_error = ss_within - ss_subjects
    df_error = df_subjects * df_condition
    ms_error = ss_error / df_error

    # F values
    f_value = ms_condition / ms_error
    p_value = sp.f.sf(f_value, df_condition, df_error)

    # Package outputs
    output = {"Between": 
                {"DF": df_condition, 
                 "SS": ss_condition, 
                 "MS": ms_condition},
              "Within": 
                {"DF": df_total - df_condition, 
                 'SS': ss_within},
              "BetweenS": 
                {"DF": df_subjects, 
                 "SS": ss_subjects},
              "Error": 
                {"DF": df_error, 
                 "SS": ss_error, 
                 "MS": ms_error},
              "Total": 
                {"DF": df_total, 
                 "SS": ss_total},
              "F_Value": f_value, "P_value": p_value}

    return output