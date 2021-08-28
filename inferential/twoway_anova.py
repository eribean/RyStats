import numpy as np
import scipy.stats as sp
from itertools import product

__all__ = ['twoway_anova']


def _effect_coding(levels):
    """Effect Coding for linear model"""
    values = np.unique(levels)
    compare = (levels == values[-1]).astype('float')
    
    coding_matrix = np.zeros((levels.shape[0], values.size - 1))
    for ndx, value in enumerate(values[:-1]):
        coding_matrix[:, ndx] = (levels == value) - compare
        
    return coding_matrix


def _interaction_coding(level1, level2):
    interaction_matrix = np.zeros((level1.shape[0], 
                                   level1.shape[1] * level2.shape[1]), 
                                  dtype='float')
    
    for ndx, (ndx1, ndx2) in enumerate(product(range(level1.shape[1]), 
                                               range(level2.shape[1]))):
        interaction_matrix[:, ndx] = level1[:, ndx1] * level2[:, ndx2]

    return interaction_matrix


def _sum_of_squared_difference(design, dependent):
    """Computes the sum of squares difference."""
    coefficients = np.linalg.pinv(design) @ dependent
    values = design @ coefficients
    return np.square(values - dependent).sum()
    

def _compute_statistics(the_dict):
    """Computes MS / F / P values for input"""
  
    # Mean Squared Error and F Values
    the_dict['MS1'] = the_dict['SS1'] / the_dict['df1']
    the_dict['MS2'] = the_dict['SS2'] / the_dict['df2']
    the_dict['MS12'] = the_dict['SS12'] / the_dict['df12']
    the_dict['MSE'] = the_dict['SSE'] / the_dict['dfe']
    the_dict['MST'] = the_dict['SST'] / the_dict['dft']
    the_dict['F1'] = the_dict['MS1'] / the_dict['MSE']
    the_dict['F2'] = the_dict['MS2'] / the_dict['MSE']
    the_dict['F12'] = the_dict['MS12'] / the_dict['MSE']

    # Oh yeah, hypothesis testing...
    the_dict['p1'] = sp.f.sf(the_dict['F1'], the_dict['df1'], the_dict['dfe'])
    the_dict['p2'] = sp.f.sf(the_dict['F2'], the_dict['df2'], the_dict['dfe'])
    the_dict['p12'] = sp.f.sf(the_dict['F12'], the_dict['df12'], the_dict['dfe'])
    
    return the_dict


def twoway_anova(level1, level2, dependent):
    """Computes a two way anova.

    Computes Type1, Type2, Type3 anova models

    Args:
        independent1: (int) input vector defining IV 1 group association
        independent2: (int) input vector defining IV 2 group association
        dependent: input vector containing measurements

    Returns:
        Type1_a: Type 1 anova with all level 1 variance
        Type1_b: Type 1 anova with all level 2 variance
        Type2: Type 2 anova when no interaction assumed
        Type3: Type 3 anova for no overlapping variance

    Notes:
        Missing data (denoted by np.nan) is removed from consideration,
        to change this behaviour, impute the data before running this 
        function

    """
    # Truncate missing data
    valid_data_mask = ~np.isnan(dependent)
    dependent = dependent[valid_data_mask]
    level1 = level1[valid_data_mask]
    level2 = level2[valid_data_mask]

    # Get the Individual matrics
    level1_matrix = _effect_coding(level1)
    level2_matrix = _effect_coding(level2)
    
    # Degress of freedom
    level1_dof = level1_matrix.shape[1] 
    level2_dof = level2_matrix.shape[1] 
    inter_dof = level1_dof * level2_dof
    total_dof = dependent.size - 1
    error_dof = total_dof - level1_dof - level2_dof - inter_dof
    dof_dict = {'df1': level1_dof, 'df2': level2_dof, 'df12': inter_dof,
                'dfe': error_dof, 'dft': total_dof}

    interaction_matrix = _interaction_coding(level1_matrix, level2_matrix)
    mean_matrix = np.ones((level1.shape[0], 1))
    
    # Total Error
    ss_total = np.var(dependent) * dependent.size
    
    # Residual Error
    full_model = np.c_[mean_matrix, level1_matrix, level2_matrix, 
                       interaction_matrix]
    ss_error = _sum_of_squared_difference(full_model, dependent)
    
    # Type 1 / Type 2
    levelA = np.c_[mean_matrix, level1_matrix]
    ss_levelA = _sum_of_squared_difference(levelA, dependent)
    ss_a1 = ss_total - ss_levelA

    levelB = np.c_[mean_matrix, level1_matrix, level2_matrix]
    ss_levelB = _sum_of_squared_difference(levelB, dependent)
    ss_b2 = ss_levelA - ss_levelB

    levelA = np.c_[mean_matrix, level2_matrix]
    ss_levelA = _sum_of_squared_difference(levelA, dependent)
    ss_b1 = ss_total - ss_levelA
    
    levelB = np.c_[mean_matrix, level2_matrix, level1_matrix]
    ss_a2 = ss_levelA - ss_levelB

    # Interaction Term
    ss_int = ss_levelB - ss_error
    
    # Type 3
    levelA = np.c_[mean_matrix, level2_matrix, interaction_matrix]
    ss_a3 = (_sum_of_squared_difference(levelA, dependent) - 
             ss_error)
    
    levelA = np.c_[mean_matrix, level1_matrix, interaction_matrix]
    ss_b3 = (_sum_of_squared_difference(levelA, dependent) - 
             ss_error)
    
    # Populate the outputs
    type1_a = {'SS1': ss_a1, 'SS2': ss_b2, 'SS12': ss_int, 
               'SSE': ss_error, 'SST': ss_total}
    type1_a.update(dof_dict)
    type1_a = _compute_statistics(type1_a)

    type1_b = {'SS1': ss_a2, 'SS2': ss_b1, 'SS12': ss_int, 
               'SSE': ss_error, 'SST': ss_total}
    type1_b.update(dof_dict)    
    type1_b = _compute_statistics(type1_b)

    type2 = {'SS1': ss_a2, 'SS2': ss_b2, 'SS12': ss_int, 
             'SSE': ss_error, 'SST': ss_total}
    type2.update(dof_dict)    
    type2 = _compute_statistics(type2)

    type3 = {'SS1': ss_a3, 'SS2': ss_b3, 'SS12': ss_int, 
             'SSE': ss_error, 'SST': ss_total}
    type3.update(dof_dict)
    type3 = _compute_statistics(type3)
    
    # UNWEIGHTED cell and marginal means
    coefficients = np.linalg.pinv(full_model) @ dependent
    codings, ndx = np.unique(full_model, axis=0, return_index=True)
    cell_means = (codings[np.argsort(ndx), :] @ coefficients).reshape(level1_dof + 1,
                                                                      level2_dof + 1)
    weighted_grand_mean = np.mean(dependent)
    weighted_marginals = [[dependent[level1 == value].mean() for value in np.unique(level1)],
                          [dependent[level2 == value].mean() for value in np.unique(level2)],]
    
    # Variance contribution
    r_squared = 1 - ss_error / ss_total
    r_squared_adj = 1 - (1 - r_squared) * total_dof / error_dof

    return {'Type1_a': type1_a, 'Type1_b': type1_b, 
            'Type2': type2, 'Type3': type3,
            'cell_means': cell_means.tolist(),
            'marginals': [cell_means.mean(1).tolist(), cell_means.mean(0).tolist()],
            'grand': coefficients[0],
            'weighted_marginals': weighted_marginals,
            'weighted_mean': weighted_grand_mean,
            'R2': r_squared,
            'R2_adj': r_squared_adj}