from typing import Optional

import numpy as np

from quickstats.maths.interpolation import get_roots

def significance(tpr:np.ndarray, fpr:np.ndarray, epsilon:Optional[float]=None):
    tpr, fpr = np.asarray(tpr), np.asarray(fpr)
    if epsilon is None:
        mask = (fpr > 0)
        tpr, fpr = tpr[mask], fpr[mask]
    else:
        fpr = fpr + epsilon
    sig = tpr / (fpr ** 0.5)
    return sig

def max_significance(tpr:np.ndarray, fpr:np.ndarray, epsilon:Optional[float]=None):
    sig = significance(tpr, fpr, epsilon=epsilon)
    max_sig = np.max(sig)
    return max_sig

def threshold_significance(tpr, fpr, fpr_thres:float, delta:float=0.0001,
                           reduction:Optional[str]='mean',
                           default:Optional[float]=0.):
    tprs_thres = get_roots(tpr, fpr, y_ref=fpr_thres, delta=delta)
    if len(tprs_thres) == 0:
        return default
    if reduction is None:
        reduce = lambda x : x
    if reduction == 'mean':
        reduce = np.mean
    elif reduction == 'median':
        reduce = np.median
    else:
        raise ValueError(f'unknown reduction method: "{reduction}"')
    reduced_tpr_thres = reduce(tprs_thres)
    sig = reduced_tpr_thres / (fpr_thres ** 0.5)
    return sig

def negative_log_likelihood(y_true, y_pred, sample_weight):
    if sample_weight is None:
        sample_weight = 1
    # use sum not mean to get the correct scaling
    return - np.sum(y_true * sample_weight * np.log(y_pred))

nll = negative_log_likelihood

def prior_ratio(y_true, y_pred, sample_weight=None):
    return 1 / np.mean(y_pred / (1 - y_pred))