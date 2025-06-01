from typing import List, Dict

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

def compute_roc_sic_statistics(
    fprs: List[np.ndarray],
    tprs: List[np.ndarray],
    resolution: int = 1000,
    mode: str = 'median',
    epsilon: float = 1e-4,
    exclude_zero: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute statistics for ROC (Receiver Operating Characteristic) and SIC (Signal-to-Interference plus noise Ratio Curve).
    
    This function calculates mean/median and confidence intervals for ROC and SIC curves across multiple runs.
    It first interpolates the curves to a common TPR axis, then computes statistics.
    
    Args:
        fprs: List of false positive rate arrays
        tprs: List of true positive rate arrays (must have same length as fprs)
        resolution: Number of points for interpolation
        mode: Statistical mode, either 'mean' or 'median'
        epsilon: Small value to prevent division by zero
        exclude_zero: Whether to exclude zero values from the TPR axis
    
    Returns:
        Dictionary containing:
            - 'tpr': Common TPR axis
            - 'fpr': Median/mean FPR values
            - 'roc': Median/mean ROC values (1/FPR)
            - 'sic': Median/mean SIC values (TPR/sqrt(FPR))
            - Error bounds for each metric ('*_errlo', '*_errhi')
    
    Raises:
        ValueError: If an unsupported mode is provided
    """
    # Find the common TPR range across all curves
    max_min_tpr = max(min(tpr) for tpr in tprs)
    min_max_tpr = min(max(tpr) for tpr in tprs)
    
    # Create common TPR axis for interpolation
    tpr_axis = np.linspace(max_min_tpr, min_max_tpr, resolution)
    if exclude_zero:
        tpr_axis = tpr_axis[tpr_axis > 0]
    
    # Interpolate metrics to common TPR axis
    interpolated = {
        'roc': [],
        'sic': [],
        'fpr': []
    }
    
    for tpr, fpr in zip(tprs, fprs):
        interp_funcs = {
            'roc': interp1d(tpr, 1 / (fpr + epsilon)),
            'sic': interp1d(tpr, tpr / np.sqrt(fpr + epsilon)),
            'fpr': interp1d(tpr, fpr)
        }
        
        for metric, func in interp_funcs.items():
            interpolated[metric].append(func(tpr_axis))
    
    for metric in interpolated:
        interpolated[metric] = np.stack(interpolated[metric])
    
    if mode == 'mean':
        central_values = {
            metric: np.mean(values, axis=0) 
            for metric, values in interpolated.items()
        }
        
        error_bounds = {
            metric: (np.std(values, axis=0), np.std(values, axis=0))
            for metric, values in interpolated.items()
        }
    
    elif mode == 'median':
        central_values = {
            metric: np.median(values, axis=0) 
            for metric, values in interpolated.items()
        }
        
        error_bounds = {
            metric: (
                central_values[metric] - np.quantile(values, 0.16, axis=0),
                np.quantile(values, 0.84, axis=0) - central_values[metric]
            )
            for metric, values in interpolated.items()
        }
    
    else:
        raise ValueError(f'Unsupported mode: "{mode}". Choose "mean" or "median".')
    
    results = {
        'tpr': tpr_axis,
        'fpr': central_values['fpr'],
        'roc': central_values['roc'],
        'sic': central_values['sic'],
    }

    for metric in ['roc', 'sic', 'fpr']:
        results[f'{metric}_errlo'] = error_bounds[metric][0]
        results[f'{metric}_errhi'] = error_bounds[metric][1]
    
    return results