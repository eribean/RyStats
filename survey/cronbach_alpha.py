import numpy as np


__all__ = ["cronbach_alpha"]


def cronbach_alpha(data, invalid_response=None):
    """Computes reliability coefficient based on data.

    Args:
        data: [items x observation] Input collected
        invalid_response: (numeric) number to specify bad values

    Returns:
        alpha: measure of internal consistency

    Notes:
        Items need to be corrected for reverese scoring
    """
    n_items = data.shape[0]

    valid_mask = True
    if invalid_response is not None:
        valid_mask = data != invalid_response
    
    item_variance = np.var(data, axis=1, ddof=1, where=valid_mask).sum()
    people_variance = (n_items * n_items
                       * np.mean(data, axis=0, where=valid_mask).var(ddof=1))
    
    return (n_items / (n_items - 1) 
            * (1 - item_variance / people_variance))