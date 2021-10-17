import numpy as np


__all__ = ['reverse_score']


def reverse_score(dataset, mask, max_val=5, invalid_response=None):
    """Reverse scores a set of ordinal data.

    Args:
        data: Matrix [items x observations] of measured responses
        mask: Boolean Vector with True indicating a reverse scored item
        max_val:  (int) maximum value in the Likert (-like) scale
        invalid_response: (numeric) invalid response in data

    Returns:
        data_reversed: Dataset with responses reverse scored
    """
    if(dataset.shape[0] != mask.shape[0]):
        raise AssertionError("First dimension of data and mask must be equal")

    new_data = dataset.copy()

    new_data[mask] = max_val + 1 - new_data[mask]
    
    # If there is an invalid response reset it
    if invalid_response is not None:
        new_data[dataset == invalid_response] = invalid_response

    return new_data