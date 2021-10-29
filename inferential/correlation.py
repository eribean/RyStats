import numpy as np

from scipy import stats as sp
from scipy.optimize import fminbound


__all__ =  ['pearsons_correlation', 'polyserial_correlation']


def pearsons_correlation(raw_data):
    """Computes the correlation and statistics for a dataset.

    Args:
        raw_data:  Data matrix [n_items, n_observations]

    Returns:
        dict: Dictionary of correlation, and critical rho values

    Notes:
        The integration is over the n_observations such that the output is
        of size [n_items, n_items]
    """
    correlation = np.corrcoef(raw_data)

    # Compute the critical values for the 3 significance tests
    deg_of_freedom = raw_data.shape[1] - 2
    t_critical = sp.t.isf([.025, .005, 0.0005] , deg_of_freedom)
    r_critical = np.sqrt(t_critical**2 / (t_critical**2 + deg_of_freedom))

    return {
        'Correlation': correlation, 
        'R critical': {'.05': r_critical[0],
                       '.01': r_critical[1],
                       '.001': r_critical[2]},
    }   


def polyserial_correlation(continuous, ordinal):
    """Computes the polyserial correlation.
    
    Estimates the correlation value based on a bivariate
    normal distribution. If the ordinal input is dichotomous, 
    then the biserial correlation is returned.
    
    Args:
        continuous: Continuous Measurement
        ordinal: Ordinal Measurement
        
    Returns:
        polyserial_correlation: correlation value
        
    Notes:
        User must handle missing data
    """
    # Get the number of ordinal values
    values, counts = np.unique(ordinal, return_counts=True)
    
    # Compute the thresholds (tau's)
    thresholds = sp.norm.isf(1 - counts.cumsum() / counts.sum())[:-1]
    
    # Standardize the continuous variable
    standardized_continuous = ((continuous - continuous.mean())
                               / continuous.std(ddof=1))

    def _min_func(correlation):
        denominator = np.sqrt(1 - correlation * correlation)
        k = standardized_continuous * correlation
        log_likelihood = 0
        
        for ndx, value in enumerate(values):
            mask = ordinal == value
            
            if ndx == 0:
                numerator = thresholds[ndx] - k[mask]
                probabilty = sp.norm.cdf(numerator / denominator)
                
            elif ndx == (values.size -1):
                numerator = thresholds[ndx-1] - k[mask]
                probabilty = (1 - sp.norm.cdf(numerator / denominator))
                
            else:
                numerator1 = thresholds[ndx] - k[mask]
                numerator2 = thresholds[ndx-1] - k[mask]
                probabilty = (sp.norm.cdf(numerator1 / denominator)
                              - sp.norm.cdf(numerator2 / denominator))
        
            log_likelihood -= np.log(probabilty).sum()
        
        return log_likelihood
        
    rho = fminbound(_min_func, -.99, .99)

    # Likelihood ratio test
    log_likelihood_rho = _min_func(rho)
    log_likelihood_zero = _min_func(0.0)
    likelihood_ratio = -2 * (log_likelihood_rho - log_likelihood_zero)
    p_value = sp.chi2.sf(likelihood_ratio, 1)
    
    return {
        'Correlation': rho,
        'Likelihood Ratio': likelihood_ratio,
        'p-value': p_value
    }