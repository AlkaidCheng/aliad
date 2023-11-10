from typing import Optional, Tuple, Union, Iterable, Callable
import numpy as np
import tensorflow as tf

def prepare_dataset(*X: Union[np.ndarray, tf.Tensor], 
                    y: Union[np.ndarray, tf.Tensor],
                    weight: Optional[Union[np.ndarray, tf.Tensor]]=None,
                    batch_size: int = 32,
                    shuffle: bool = True,
                    seed: Optional[int] = None,
                    drop_remainder: bool = True,
                    buffer_size: Optional[int] = None,
                    cache:bool = False,
                    prefetch:bool = True,
                    repeat:bool = False,
                    preprocess_function: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
                    device: str = "/cpu:0") -> tf.data.Dataset:
    """
    Prepare a TensorFlow dataset from numpy arrays or TensorFlow tensors with options for shuffling, caching, batching, and prefetching.
    
    The function creates a TensorFlow Dataset from the provided feature tensors (or arrays) and a label tensor (or array).
    The dataset can be optionally shuffled, cached, batched, and prefetched to improve training performance.
    All operations are performed in the TensorFlow device context specified by the 'device' parameter.

    Parameters:
    *X (Iterable[Union[np.ndarray, tf.Tensor]]): An iterable of numpy arrays or TensorFlow tensors representing features.
        Each array or tensor should have the same first dimension size (number of samples).
    y (Union[np.ndarray, tf.Tensor]): A numpy array or TensorFlow tensor representing labels. 
        Should have the same first dimension size (number of samples) as the elements in *X.
    batch_size (int, optional): Number of consecutive elements of the dataset to combine in a single batch.
        Defaults to 32.
    shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
    seed (Optional[int], optional): Random seed used for shuffling the dataset. Defaults to None.
    drop_remainder (bool, optional): Whether the last batch should be dropped in case it has fewer than batch_size elements.
        Defaults to True.
    buffer_size (Optional[int], optional): Buffer size to use for shuffling the dataset. 
        If None, it defaults to the number of samples in the dataset. Defaults to None.
    cache (bool, optional): Whether to cache the dataset in memory. Defaults to False.
    prefetch (bool, optional): Whether to prefetch batches of the dataset. Defaults to True.
    preprocess_function (Optional[Callable[[tf.Tensor], tf.Tensor]], optional): A function to preprocess the input feature tensors.
        It should take in a tuple of tensors and return a tuple of tensors with the same length.
    device (str, optional): TensorFlow device to use for creating the dataset. Defaults to "/cpu:0".

    Returns:
    tf.data.Dataset: A tf.data.Dataset instance representing the prepared dataset.

    Notes:
    - Caching is useful when the dataset is small enough to fit in memory, as it can significantly speed up training
      by avoiding repeated data loading and preprocessing. However, it should be used cautiously with large datasets
      to avoid out-of-memory errors.
    - Prefetching allows the data loading to be performed asynchronously, improving GPU utilization during training.
    - Shuffling is performed before batching, and the buffer size for shuffling should be sufficiently large to ensure
      good randomness.
    - If a `preprocess_function` is provided, it will be applied to the dataset after loading and before any other
      transformations. The function should expect a tuple of feature tensors and a label tensor, and return a tuple
      of preprocessed feature tensors and a label tensor.
    """
    with tf.device(device):
        if buffer_size is None:
            buffer_size = X[0].shape[0]

        if len(X) == 1:
            if weight is None:
                ds = tf.data.Dataset.from_tensor_slices((X[0], y))
            else:
                ds = tf.data.Dataset.from_tensor_slices((X[0], y, weight))

            if preprocess_function is not None:
                ds = ds.map(lambda x, y: (preprocess_function(x), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
        elif len(X) > 1:
            if weight is None:
                ds = tf.data.Dataset.from_tensor_slices((tuple(X), y))
            else:
                ds = tf.data.Dataset.from_tensor_slices((tuple(X), y, weight))

            if preprocess_function is not None:
                ds = ds.map(lambda x, y: (preprocess_function(*x), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
        else:
            raise ValueError('no feature arrays specified')
        
        if cache:
            ds = ds.cache()
            
        if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)

        if batch_size is not None:
            ds = ds.batch(batch_size, drop_remainder=drop_remainder)

        if repeat:
            ds = ds.repeat()
        
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        
    return ds
