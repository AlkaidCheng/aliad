import numpy as np

def split_dataset(X, y, weight=None, test_size=None, val_size=None, train_size=None, shuffle=True, random_state=None):
    """
    Split dataset into training, validation, and test sets.

    Parameters:
    - X (array-like or dict of array-like): Features to be split. If X is a dictionary, 
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
    - random_state (int or RandomState instance, optional): Pseudo-random number 
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
    data_splits = split_dataset(X, y, test_size=0.3, val_size=0.2, random_state=42)
    ```
    """
    total_samples = y.shape[0]
    rng = np.random.default_rng(random_state)

    def calculate_size(size):
        if size is None:
            return None
        elif 0 < size < 1:
            return int(size * total_samples)
        elif size >= 1:
            return int(size)
        else:
            raise ValueError("Size must be a fraction between 0 and 1 or an integer greater than or equal to 1")

    test_samples = calculate_size(test_size)
    val_samples = calculate_size(val_size)
    train_samples = calculate_size(train_size)

    
    fractions = [isinstance(size, float) and 0 < size < 1 for size in [test_size, val_size, train_size]]
    numbers   = [isinstance(size, int) for size in [test_size, val_size, train_size]]
    if any(fractions) and any(numbers):
        raise ValueError('Sizes must be all integers or all fractions')
    if any(fractions) and (train_size is None):
        if (test_size is not None) and (val_size is not None):
            train_size = 1 - test_size - val_size
        elif (val_size is not None):
            train_size = 1 - val_size
        elif (test_size is not None):
            train_size = 1 - test_size
        train_samples = calculate_size(train_size)

    if all(fractions):
        if train_size + val_size + test_size != 1:
            raise ValueError("Fractions of train, val and test do not sum to 1")
    
    if train_samples is not None and val_samples is not None and test_samples is not None:
        if train_samples + val_samples + test_samples > total_samples:
            raise ValueError("Requested number of samples is larger than the number of samples available")

    def split_array(array, num_samples):
        if (num_samples is None) or (num_samples == 0):
            return None, array
        return array[:num_samples], array[num_samples:]
    if shuffle:
        indices = rng.permutation(total_samples)
    else:
        indices = np.arange(total_samples)
    train_index, indices = split_array(indices, train_samples)
    val_index, indices = split_array(indices, val_samples)
    test_index, indices = split_array(indices, test_samples)
    del indices

    def select_data(data, index):
        if index is None:
            return data
        return data[index]

    def split_data(X_data, y_data, weight_data, train_samples, val_samples, test_samples):
        if not isinstance(X_data, (tuple, dict)):
            X_data = (X_data, )
        X_train, X_val, X_test = {}, {}, {}
        keys = range(len(X_data)) if isinstance(X_data, tuple) else X_data.keys()
        for key in keys:
            X_key_train = select_data(X_data[key], train_index)
            X_key_val = select_data(X_data[key], val_index)
            X_key_test = select_data(X_data[key], test_index)
            X_train[key], X_val[key], X_test[key] = X_key_train, X_key_val, X_key_test
        # unwrap data
        if isinstance(X_data, tuple):
            if len(X_data) == 1:
                X_train, X_val, X_test = X_train[0]. X_val[0], X_test[0]
            else:
                X_train, X_val, X_test = tuple(X_train.values()), tuple(X_val.values()), tuple(X_test.values())
        y_train = select_data(y_data, train_index)
        y_val = select_data(y_data, val_index)
        y_test = select_data(y_data, test_index)
        if weight_data is not None:
            weight_train = select_data(weight_data, train_index)
            weight_val = select_data(weight_data, val_index)
            weight_test = select_data(weight_data, test_index)
        else:
            weight_train = None
            weight_val = None
            weight_test = None
        return X_train, X_val, X_test, y_train, y_val, y_test, weight_train, weight_val, weight_test

    X_train, X_val, X_test, y_train, y_val, y_test, weight_train, weight_val, weight_test = split_data(X, y, weight, train_samples, val_samples, test_samples)
    data_splits = {'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'y_train': y_train, 'y_val': y_val, 'y_test': y_test}
    if weight is not None:
        data_splits['weight_train'] = weight_train
        data_splits['weight_val'] = weight_val
        data_splits['weight_test'] = weight_test
    return data_splits