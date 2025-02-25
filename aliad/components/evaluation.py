import numpy as np

from scipy.interpolate import interp1d

def get_bootstrap_prediction(y_prob:np.ndarray, y_true:np.ndarray,
                             sample_weight:np.ndarray=None,
                             num_bootstrap_samples:int=20,
                             seed:int=2023):
    np.random.seed(seed)
    results = {
        'y_prob': [],
        'y_true': []
    }
    if sample_weight is not None:
        results['sample_weight'] = []
    sample_size = y_prob.shape[0]
    for _ in range(num_bootstrap_samples):
        # Sampling with replacement
        indices = np.random.choice(sample_size, sample_size, replace=True)
        for label, values in [("y_prob", y_prob),
                              ("y_true", y_true),
                              ("sample_weight", sample_weight)]:
            if values is None:
                continue
            results[label].append(values[indices])
    for label in results:
        results[label] = np.array(results[label])
    return results

def get_significance(fpr, tpr, epsilon:float=1e-4):
    fpr, tpr = np.array(fpr), np.array(tpr)
    significance = tpr/((fpr + epsilon) ** (0.5))
    return significance

def get_max_significance(fpr, tpr, epsilon:float=1e-4):
    significance = get_significance(fpr, tpr, epsilon=epsilon)
    return np.max(significance)
    
def compute_median_and_variance_roc_sic(fprs_list, tprs_list,
                                        resolution:int=1000,
                                        mode:str='median',
                                        epsilon:float=1e-4,
                                        exclude_zero:bool=True):
    
    # interpolation
    max_min_tpr = 0.
    min_max_tpr = 1.
    for tpr in tprs_list:
        if min(tpr) > max_min_tpr:
            max_min_tpr = min(tpr)
        if max(tpr) < min_max_tpr:
            min_max_tpr = max(tpr)
    tpr_manual = np.linspace(max_min_tpr, min_max_tpr, resolution)
    if exclude_zero:
        tpr_manual = tpr_manual[tpr_manual > 0]
        
    roc_interpol = []
    sic_interpol = []
    fpr_interpol = []
    for tpr, fpr in zip(tprs_list, fprs_list):
        roc_function = interp1d(tpr, 1 /(fpr + epsilon))
        sic_function = interp1d(tpr, tpr/((fpr + epsilon) ** (0.5)))
        fpr_function = interp1d(tpr, fpr)
        roc_interpol.append(roc_function(tpr_manual))
        sic_interpol.append(sic_function(tpr_manual))
        fpr_interpol.append(fpr_function(tpr_manual))
    # mean + std
    if mode == 'mean':
        roc_median = np.mean(np.stack(roc_interpol), axis=0)
        sic_median = np.mean(np.stack(sic_interpol), axis=0)
        fpr_median = np.mean(np.stack(fpr_interpol), axis=0)
        roc_std = np.std(np.stack(roc_interpol), axis=0)
        sic_std = np.std(np.stack(sic_interpol), axis=0)
        fpr_std = np.std(np.stack(fpr_interpol), axis=0)
        roc_std = (roc_std, roc_std)
        sic_std = (sic_std, sic_std)
        fpr_std = (fpr_std, fpr_std)
    # median + quantiles
    elif mode == 'median':
        roc_median = np.median(np.stack(roc_interpol), axis=0)
        sic_median = np.median(np.stack(sic_interpol), axis=0)
        fpr_median = np.median(np.stack(fpr_interpol), axis=0)
        roc_std = (roc_median - np.quantile(np.stack(roc_interpol), 0.16, axis=0), np.quantile(np.stack(roc_interpol), 0.84, axis=0) - roc_median)
        sic_std = (sic_median - np.quantile(np.stack(sic_interpol), 0.16, axis=0), np.quantile(np.stack(sic_interpol), 0.84, axis=0) - sic_median)
        fpr_std = (fpr_median - np.quantile(np.stack(fpr_interpol), 0.16, axis=0), np.quantile(np.stack(fpr_interpol), 0.84, axis=0) - fpr_median)
    else:
        raise ValueError(f'unsupported mode: "{mode}"')
        
    results = {
        'tpr': tpr_manual,
        'fpr': fpr_median,
        'roc': roc_median,
        'sic': sic_median,
        'roc_errlo': roc_std[0],
        'roc_errhi': roc_std[1],
        'sic_errlo': sic_std[0],
        'sic_errhi': sic_std[1],
        'fpr_errlo': fpr_std[0],
        'fpr_errhi': fpr_std[1],
    }
    return results

