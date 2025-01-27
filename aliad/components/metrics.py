from typing import Optional, Union, Callable

import numpy as np
from quickstats.maths.interpolation import get_roots

def sic(
    tpr: np.ndarray,
    fpr: np.ndarray,
    epsilon: Optional[float] = None
) -> np.ndarray:
    """Calculate the Significance Improvement Characteristic.

    Parameters
    ----------
    tpr : array-like
        True positive rate values
    fpr : array-like
        False positive rate values
    epsilon : float, optional
        Small constant to add to fpr to avoid division by zero

    Returns
    -------
    np.ndarray
        Significance improvement values for each (tpr, fpr) pair

    Notes
    -----
    If epsilon is None, points with fpr=0 are removed.
    Otherwise, epsilon is added to all fpr values.
    """
    tpr, fpr = np.asarray(tpr), np.asarray(fpr)
    if tpr.shape != fpr.shape:
        raise ValueError('`tpr` and `fpr` must have the same shape')
    if epsilon is None:
        mask = (fpr > 0)
        tpr, fpr = tpr[mask], fpr[mask]
    else:
        fpr = fpr + epsilon
    return tpr / np.sqrt(fpr)

def max_sic(
    tpr: np.ndarray,
    fpr: np.ndarray,
    epsilon: Optional[float] = None
) -> float:
    """Calculate the maximum Significance Improvement Characteristic.

    Parameters
    ----------
    tpr : array-like
        True positive rate values
    fpr : array-like
        False positive rate values
    epsilon : float, optional
        Small constant to add to fpr to avoid division by zero

    Returns
    -------
    float
        Maximum significance improvement value
    """
    sic_values = sic(tpr, fpr, epsilon=epsilon)
    return float(np.max(sic_values))

def threshold_sic(
    tpr: np.ndarray,
    fpr: np.ndarray,
    fpr_thres: float,
    delta: float = 0.0001,
    reduction: Optional[str] = 'mean',
    default: Optional[float] = 0.
) -> float:
    """Calculate the threshold-based Significance Improvement Characteristic.

    Parameters
    ----------
    tpr : array-like
        True positive rate values
    fpr : array-like
        False positive rate values
    fpr_thres : float
        False positive rate threshold
    delta : float, default=0.0001
        Tolerance for root finding
    reduction : {'mean', 'median', None}, default='mean'
        Method to reduce multiple threshold values
    default : float, optional, default=0.
        Default value when no roots are found

    Returns
    -------
    float
        Threshold-based significance improvement value

    Raises
    ------
    ValueError
        If reduction method is not recognized
    """
    tprs_thres = get_roots(tpr, fpr, y_ref=fpr_thres, delta=delta)
    
    if len(tprs_thres) == 0:
        return default
        
    if reduction is None:
        reduce: Callable = lambda x: x
    elif reduction == 'mean':
        reduce = np.mean
    elif reduction == 'median':
        reduce = np.median
    else:
        raise ValueError(f'Unknown reduction method: "{reduction}". '
                       f'Expected one of: None, "mean", "median"')
    
    reduced_tpr_thres = reduce(tprs_thres)
    return reduced_tpr_thres / np.sqrt(fpr_thres)

def negative_log_likelihood(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """Calculate the negative log likelihood.

    Parameters
    ----------
    y_true : array-like
        Truth labels
    y_pred : array-like
        Predicted likelihoods
    sample_weight : array-like, optional
        Sample weights

    Returns
    -------
    float
        Negative log likelihood value
    """
    if sample_weight is None:
        sample_weight = 1
    log_likelihoods = y_true * sample_weight * np.log(y_pred)
    axis = np.ndim(log_likelihoods) - 1
    return - np.sum(log_likelihoods, axis=axis)

# Alias for negative_log_likelihood
nll = negative_log_likelihood