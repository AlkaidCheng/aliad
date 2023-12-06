import math
from numbers import Integral
from typing import Union, Optional, List, Dict

import numpy as np

def split_array(array, num_samples):
    if (num_samples is None) or (num_samples == 0):
        return None, array
    return array[:num_samples], array[num_samples:]

def get_optimal_stratified_split(split_sizes:Dict, cls_sizes:Dict):
    def is_all_int(sizes_):
        return all(isinstance(size, Integral) for size in sizes_)
    split_sizes_ = list(split_sizes.values())
    if not is_all_int(split_sizes_):
        raise ValueError('split_sizes must be all integers')
    cls_sizes_ = list(cls_sizes.values())
    if not is_all_int(cls_sizes_):
        raise ValueError('cls_sizes must be all integers')
    total_split_size = np.sum(split_sizes_)
    total_cls_size = np.sum(cls_sizes_)
    if total_split_size > total_cls_size:
        raise ValueError('sum of split sizes cannot be larger than sum of class sizes')
    cls_sizes_ = np.array(cls_sizes_)
    classes = list(cls_sizes)
    cls_fractions = cls_sizes_ / total_cls_size
    remain_cls_sizes = cls_sizes_.copy()
    strat_split_sizes = {}
    for label, size in split_sizes.items():
        opt_cls_sizes = optimize_fraction_partition(size, cls_fractions,
                                                    upper_bounds=remain_cls_sizes)
        strat_split_sizes[label] = dict(zip(classes, opt_cls_sizes))
        remain_cls_sizes -= opt_cls_sizes
    return strat_split_sizes

def optimize_fraction_partition(total, fractions:np.ndarray,
                                upper_bounds:Optional[np.ndarray]=None,
                                backfill:bool=False):
    fractions = np.array(fractions)
    sum_frt = np.sum(fractions)
    if sum_frt > 1:
        raise ValueError('sum of fractions must not exceed 1')
    if not (fractions > 0).all():
        raise ValueError('all fractions must be positibe')
    exp_total = min(round(sum_frt * total), total)
    trial_sizes = np.round(total * fractions).astype(int)
    trial_total = np.sum(trial_sizes)
    if (trial_total == exp_total):
        return trial_sizes
    diff = exp_total - trial_total
    if upper_bounds is None:
        upper_bounds = np.full(fractions.shape, total)
    if np.sum(upper_bounds) < total:
        raise ValueError('sum of upper bounds can not be smaller than total size')
    while diff != 0:
        direction = np.sign(diff)
        total_ = np.sum(trial_sizes) + direction
        new_fractions = (trial_sizes + direction) / total_
        derivative = np.abs(new_fractions/fractions - 1)
        bounded_indices = np.where((trial_sizes + direction) > upper_bounds)
        derivative[bounded_indices] = np.max(derivative) + 1e-5
        indices = np.where(derivative == derivative.min())[0]
        indices = indices[~np.in1d(indices, bounded_indices[0])]
        if len(indices) == 0:
            raise RuntimeError('can not find partition that satisfies the upper bound condition')
        if (indices.shape[0] == 1) or (not backfill):
            idx = indices[0]
        else:
            idx = indices[-1]
        trial_sizes[idx] += direction
        diff -= direction
    return trial_sizes

def optimize_split_sizes(total_count:int,
                         split_sizes:Dict, 
                         backfill:bool=False):
    sizes = list(split_sizes.values())
    all_int = all(isinstance(size, Integral) for size in sizes)
    if all_int:
        if np.sum(sizes) > total_count:
            raise ValueError('sum of split sizes must not exceed the total size')
        return {**split_sizes}
    sizes = np.array(sizes)
    all_frt = ((sizes > 0) & (sizes < 1)).all()
    if not all_frt:
        raise ValueError('Sizes must be all integers or all fractions')
    sum_frt = np.sum(sizes)
    if sum_frt > 1:
        raise ValueError('sum of split fractions must not exceed 1')
    opt_sizes = optimize_fraction_partition(total_count, sizes, backfill=backfill)
    return dict(zip(list(split_sizes), opt_sizes))

def get_train_val_test_split_sizes(total_size:int, test_size=None, val_size=None, train_size=None):
    split_sizes = {
        'train' : train_size,
        'val'   : val_size,
        'test'  : test_size
    }
    split_sizes = {k: v for k, v in split_sizes.items() if v is not None}
    sizes = np.array(list(split_sizes.values()))
    all_frt = ((sizes > 0) & (sizes < 1)).all()
    # make train size the residual of val and/or test size
    if all_frt and (train_size is None):
        # order is important here
        split_sizes = {'train': 1 - np.sum(sizes), **split_sizes}
    opt_split_sizes = optimize_split_sizes(total_size, split_sizes, backfill=True)
    return opt_split_sizes
    
def get_split_indices(total_size:int, split_sizes:Union[int, Dict], stratify=None, shuffle=True, seed=None):
    if isinstance(split_sizes, int):
        split_sizes = [split_sizes] * (total_size // split_sizes) + [total_size % split_sizes]
    if isinstance(split_sizes, (list, tuple)):
        split_sizes = {i: size for i, size in enumerate(split_sizes)}
    split_sizes_ = np.array(list(split_sizes.values()))
    if not np.issubdtype(split_sizes_.dtype, np.integer):
        raise ValueError('split sizes must be all integers')
    total_split_size = np.sum(split_sizes_)
    if total_split_size > total_size:
        raise ValueError('sum of split sizes must not exceed the total size')
    if stratify is not None:
        assert len(stratify) == total_size
    rng = np.random.default_rng(seed)
    if shuffle:
        indices = rng.permutation(total_size)
    else:
        if stratify is not None:
            raise ValueError('stratify can not be used with shuffle=False')
        indices = np.arange(total_size)
    split_indices = {}
    if stratify is not None:
        y = stratify
        classes, y_indices = np.unique(y[indices], return_inverse=True)
        class_indices = {cls: indices[y_indices == i] for i, cls in enumerate(classes)}
        class_sizes = {cls: class_indices[cls].shape[0] for cls in classes}
        sample_sizes = get_optimal_stratified_split(split_sizes, class_sizes)
        for sample, size in sample_sizes.items():
            split_indices[sample] = []
            for cls in classes:
                indices_, class_indices[cls] = split_array(class_indices[cls], size[cls])
                split_indices[sample].append(indices_)
            split_indices[sample] = np.concatenate(split_indices[sample])
            # should always be true
            if shuffle:
                split_indices[sample] = rng.permutation(split_indices[sample])
        del class_indices
    else:
        for label, size in split_sizes.items():
            split_indices[label], indices = split_array(indices, size)
        del indices
    return split_indices

def split_dataset(X, y=None, weight=None, test_size=None, val_size=None, train_size=None,
                  stratify=None, shuffle=True, seed=None):
    """
    Split dataset into training, validation, and test sets.

    Parameters:
    - X (array-like, tuple of array-like or dict of array-like): Features to be split. If X is a tuple/dictionary, 
      the values must be of equal length.
    - y (array-like): Labels corresponding to X. The length must be equal to the 
      length of X.
    - test_size (float or int, optional): If float, should be between 0.0 and 1.0 and 
      represent the proportion of the dataset to include in the test split. If int, 
      represents the absolute number of test samples.
    - val_size (float or int, optional): If float, should be between 0.0 and 1.0 and 
      represent the proportion of the dataset to include in the validation split. 
      If int, represents the absolute number of validation samples.
    - train_size (float or int, optional): If float, should be between 0.0 and 1.0 and 
      represent the proportion of the dataset to include in the train split. If int, 
      represents the absolute number of train samples.
    - shuffle (bool, optional, default=True): Whether or not to shuffle the data 
      before splitting.
    - seed (int or RandomState instance, optional): Pseudo-random number 
      generator state used for random sampling.

    Returns:
    - dict: Dictionary containing split data. Possible keys are 'X_train', 'X_val', 
      'X_test', 'y_train', 'y_val', 'y_test'. The values are the corresponding splits.

    Behavior:
    - If any of test_size, val_size, or train_size are fractions, they must all be 
      fractions or None. The fractions represent proportions of the dataset.
    - If test_size, val_size, and train_size are all specified as fractions, their 
      sum must be equal to 1.0.
    - If only one of the sizes is specified as a fraction, the other(s) will be 
      inferred to make the fractions sum to 1.0.
    - If the sizes are specified as integers, they represent the absolute number of 
      samples from the dataset.
    - If the sum of the integers specified for the sizes is greater than the total 
      number of samples available, a ValueError will be raised.
    - If both fractions and integers are used to specify the sizes, a ValueError 
      will be raised.

    Example:
    ```python
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 1, 0, 1, 0, 1])
    data_splits = split_dataset(X, y, test_size=0.3, val_size=0.2, seed=42)
    ```
    """
    def get_total_size():
        sizes = []
        if isinstance(X, tuple):
            sizes.extend([len(x) for x in X])
        elif isinstance(X, dict):
            sizes.extend([len(x) for x in X.values()])
        else:
            sizes.append(len(X))
        if y is not None:
            sizes.append(len(y))
        if weight is not None:
            sizes.append(len(weight))
        if len(np.unique(sizes)) != 1:
            raise ValueError('input arrays have inconsistent sizes')
        return sizes[0]
        
    total_size = get_total_size()

    split_sizes = get_train_val_test_split_sizes(total_size, train_size=train_size,
                                                 val_size=val_size, test_size=test_size)
    split_indices = get_split_indices(total_size,
                                      split_sizes=split_sizes,
                                      stratify=y if stratify else None,
                                      shuffle=shuffle,
                                      seed=seed)
    def select_data(data, index):
        if index is None:
            return None
        return data[index]

    def split_data(X_data, y_data, weight_data, sample_indices):
        if not isinstance(X_data, (tuple, dict)):
            X_data = (X_data, )
        data_splits = {}
        keys = range(len(X_data)) if isinstance(X_data, tuple) else X_data.keys()
        for sample in sample_indices:
            data_splits[f'X_{sample}'] = {}
            for key in keys:
                data_splits[f'X_{sample}'][key] = select_data(X_data[key], sample_indices[sample])
            if y is not None:
                data_splits[f'y_{sample}'] = select_data(y_data, sample_indices[sample])
            if weight_data is not None:
                data_splits[f'weight_{sample}'] = select_data(weight_data, sample_indices[sample])
        # unwrap data
        if isinstance(X_data, tuple):
            if len(X_data) == 1:
                for sample in sample_indices:
                    data_splits[f'X_{sample}'] = data_splits[f'X_{sample}'][0]
            else:
                for sample in sample_indices:
                    data_splits[f'X_{sample}'] = tuple(data_splits[f'X_{sample}'].values())
        return data_splits

    data_splits = split_data(X, y, weight, split_indices)
    
    return data_splits


"""
def _validate(result, total_size, split_sizes, cls_sizes):
    cls_fractions = {k: v / total_size for k, v in cls_sizes.items()}
    split_sizes_ = {}
    for label in result:
        split_sizes_[label] = np.sum(list(result[label].values()))
    for label in split_sizes:
        if split_sizes_[label] != split_sizes[label]:
            print(f'FAILED SPLIT SIZE: "{label}" ({split_sizes_[label]} vs {split_size[label]})')
            return
    print("SPLIT SIZE: ")
    print(split_sizes_)
    cls_sizes_ = {cls: 0 for cls in cls_counts}
    for label in result:
        for cls in result[label]:
            cls_sizes_[cls] += result[label][cls]
    print("CLASS SIZE: ")
    print(cls_sizes_)
    for cls in cls_sizes:
        if cls_sizes_[cls] != cls_sizes[cls]:
            print(f'FAILED CLASS SIZE: "{cls}" ({cls_sizes_[cls]} vs {cls_sizes[cls]})')
            return
    print("CLASS SIZE: PASSED")
    print("Class Fraction (Reference):")
    print(cls_fractions)
    for label in result:
        label_total_size = np.sum(list(result[label].values()))
        label_cls_fractions = {}
        for cls in result[label]:
            label_cls_fractions[cls] = result[label][cls] / label_total_size
        print(f"Class Fraction ({label}):")
        print(label_cls_fractions)
"""