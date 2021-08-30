import numpy as np


__all__ = ['simple_mediation']


def _get_mediated_coeffs(dependent, independent, mediator):
    corr_matrix = np.corrcoef(np.c_[dependent, independent, mediator].T)
    total_effect = corr_matrix[0, 1] # iv -> dv
    indy_effect = corr_matrix[2, 1] # iv -> m
    
    dep_std = dependent.std(ddof=1)
    idep_std = independent.std(ddof=1)
    med_std = mediator.std(ddof=1)

    # Compute the mediated effect m -> dv
    mediated_effect = ((corr_matrix[2, 0] - corr_matrix[2, 1] * corr_matrix[1, 0]) / 
                       (1 - corr_matrix[2, 1]**2))

    # iv -> dv removing m
    direct_effect = ((corr_matrix[1, 0] - corr_matrix[2, 1] * corr_matrix[2, 0]) / 
                     (1 - corr_matrix[2, 1]**2))

    return (total_effect * dep_std / idep_std, 
            direct_effect * dep_std / idep_std, 
            mediated_effect * dep_std / med_std, 
            indy_effect * med_std / idep_std)
    

def simple_mediation(dependent, independent, mediator, n_bootstrap=2000, seed=None):
    """Computes a simple mediation between three variables.
    
    Args:
        dependent: (1d array) dependent variable 
        independent: (1d array) independent variable
        mediator: (1d array) mediating variable
        n_bootstrap: (int) number of boostrap samples to run
        seed: (int) seed for random number generator
        
    Returns:
        mediation_dict: dictonary of values and statistics
    """
    rng = np.random.default_rng(seed)

    # Normalization terms
    dep_std = dependent.std(ddof=1)
    idep_std = independent.std(ddof=1)
    med_std = mediator.std(ddof=1)
        
    # run bootstrap to determine confidence intervals
    bootstrap_array = np.zeros((6, n_bootstrap))
    for ndx in range(n_bootstrap):
        # Resample data with replacement
        resample_ndx = rng.choice(independent.size, independent.size, replace=True)

        dep_resample = dependent[resample_ndx]
        idep_resample = independent[resample_ndx]
        med_resample = mediator[resample_ndx]
        
        otpt_effects = _get_mediated_coeffs(dep_resample, idep_resample, 
                                            med_resample)
        
        bootstrap_array[:-2, ndx] = otpt_effects
        bootstrap_array[-2, ndx] = np.prod(otpt_effects[2:])
        bootstrap_array[-1, ndx] = otpt_effects[2] * otpt_effects[3] / otpt_effects[0] * 100
    
    # get 95th and 99th confidence interval
    ci_95 = np.percentile(bootstrap_array, axis=1, q=[2.5, 97.5])
    ci_99 = np.percentile(bootstrap_array, axis=1, q=[0.5, 99.5])
    
    # compute regression coefficents
    otpt_effects = _get_mediated_coeffs(dependent, independent, 
                                        mediator)

    # Package the output
    return {'Total Effect': {'Coefficient': otpt_effects[0],
                             'Beta': otpt_effects[0] * idep_std / dep_std,
                             '95th CI': ci_95[:, 0],
                             '99th CI': ci_99[:, 0]},
            'Direct Effect': {'Coefficient': otpt_effects[1],
                              'Beta': otpt_effects[1] * idep_std / dep_std,
                              '95th CI': ci_95[:, 1],
                              '99th CI': ci_99[:, 1]},
            'Mediated Effect': {'Coefficient': otpt_effects[2],
                                'Beta': otpt_effects[2] * med_std / dep_std,
                                '95th CI': ci_95[:, 2],
                                '99th CI': ci_99[:, 2]},
            'Second Effect': {'Coefficient': otpt_effects[3],
                              'Beta': otpt_effects[3] * idep_std / med_std,
                              '95th CI': ci_95[:, 3],
                              '99th CI': ci_99[:, 3]},
            'Indirect Effect': {'Coefficient': otpt_effects[2] * otpt_effects[3],
                                'Beta': otpt_effects[2] * otpt_effects[3] * idep_std / dep_std,
                                '95th CI': ci_95[:, 4],
                                '99th CI': ci_99[:, 4]},
            'Percent Mediated': {'Coefficient': (otpt_effects[2] * otpt_effects[3] 
                                                 / otpt_effects[0] * 100),
                                 '95th CI': ci_95[:, 5], '99th CI': ci_99[:, 5]}
           }        
