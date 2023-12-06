from functools import partial
from typing import Optional, Tuple, Union, Iterable, Callable, Dict, List
import numpy as np
import tensorflow as tf

from tensorflow.keras import Input

from aliad.data.partition import get_split_indices, get_train_val_test_split_sizes

def apply_pipelines(ds:tf.data.Dataset,
                    batch_size: int = 32,
                    shuffle: bool = True,
                    seed: Optional[int] = None,
                    drop_remainder: bool = True,
                    buffer_size: Optional[int] = None,
                    cache:bool = False,
                    prefetch:bool = True,
                    repeat:bool = False) -> tf.data.Dataset:
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
                    map_funcs: Optional[Iterable[Callable]] = None,
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
                
        else:
            raise ValueError('no feature arrays specified')
            
        if map_funcs is not None:
            for map_func in map_funcs:
                ds = ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
            
        ds = apply_pipelines(ds,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             seed=seed,
                             buffer_size=buffer_size, 
                             prefetch=prefetch,
                             drop_remainder=drop_remainder,
                             cache=cache, 
                             repeat=repeat)
    return ds

def get_tf_inputs(metadata:Dict):
    Inputs = {}
    for label, feature_metadata  in metadata.items():
        Inputs[label] = Input(name=label, **feature_metadata)
    return Inputs

def get_feature_method(array:np.ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    dtype_ = array.dtype.name
    if dtype_ in ['float32']:
        # note FloatList converts float/double to float
        return lambda array_: tf.train.Feature(float_list=tf.train.FloatList(value=array_))
    elif dtype_ in ['float64', 'bool']:
        return lambda array_: tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_.tobytes()]))
    elif dtype_ in ['int64']:
        return lambda array_: tf.train.Feature(int64_list=tf.train.Int64List(value=array_))
    else:  
        raise ValueError('array must have dtype of float32, float64, or int64')

def get_feature_description(array:tf.Tensor):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        dtype_ = array.dtype.name
        if dtype_ in ['float32', 'int64']:
            return tf.io.FixedLenFeature([np.prod(array.shape[1:])], dtype=dtype_)
        elif dtype_ in ['float64', 'bool']:
            return tf.io.FixedLenFeature([], dtype=tf.string)
        else:
            raise ValueError('array must have dtype of float32, float64, or int64')
        
def get_ndarray_tfrecord_example_parser(metadata:Dict):
    Inputs = get_tf_inputs(metadata)
    feature_description = {}
    feature_parser = {}
    for label, input_ in Inputs.items():
        feature_description[label] = get_feature_description(input_)
        dtype = input_.dtype
        shape = input_.shape[1:]
        if dtype.name in ['bool', 'float64']:
            feature_parser[label] = (lambda example, out_type=dtype, shape=shape:
                                     tf.reshape(tf.io.decode_raw(example, out_type=out_type), shape))
        else:
            feature_parser[label] = (lambda example, shape=shape:
                                     tf.reshape(example, shape))
    
    def get_parsed_example(example):
        parsed_example = tf.io.parse_single_example(example, feature_description)
        for label, input_ in Inputs.items():
            parsed_example[label] = feature_parser[label](parsed_example[label])
        return parsed_example
        
    return get_parsed_example

def tfds_to_tfrecords(ds, writer:"tf.io.TFRecordWriter"):
    if isinstance(writer, str):
        writer = tf.io.TFRecordWriter(writer)
    ds_first = list(ds.take(1))[0]
    if not isinstance(ds_first, dict):
        raise RuntimeError('tfds must be sequence of dictionaries for conversion to tfrecord format')
    def _validate_arrays(**X_):
        metadata = {}
        for label, tensor in X_.items():
            if not isinstance(tensor, tf.Tensor):
                raise RuntimeError(f'input with label "{label}" is not a tensor')
            metadata[label] = {'shape': tensor.shape.as_list(), 'dtype': tensor.dtype.name}
        return metadata
    feature_metadata = _validate_arrays(**ds_first)
    feature_methods = {}
    for label, tensor in ds_first.items():
        feature_methods[label] = get_feature_method(tensor.numpy())
    size = 0
    for i, data in ds.enumerate():
        feature = {}
        for label, method in feature_methods.items():
            feature[label] = method(data[label].numpy())
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
        size += 1
    metadata = {
        "features": feature_metadata,
        "size": size
    }
    return metadata
        

def ndarray_to_tfrecords(writer:"tf.io.TFRecordWriter", **X):
    if isinstance(writer, str):
        writer = tf.io.TFRecordWriter(writer)
    def _make_proper_array(label, x_):
        if x_.ndim == 1:
            print(f'Warning: array "{label}" of dimension = 1 will be reshaped to dimension 2')
            x_ = x_.reshape((x_.shape[0], 1))
        shape = x_.shape[1:]
        dtype_ = x_.dtype.name
        if x_.ndim > 2:
            print(f'Warning: array "{label}" of dimension > 2 will be reshaped to dimension 2')
            x_ = x_.reshape((x_.shape[0], np.prod(x_.shape[1:])))
        if dtype_ not in ['float32', 'float64', 'int64', 'bool']:
            raise ValueError(f'invalid input "{label}": array must have dtype of float32, float64, bool or int64')
        if dtype_ in ['bool', 'float64']:
            print(f'Warning: {dtype_} array "{label}" will be converted into bytes')
        metadata = {"shape": shape, "dtype": dtype_}
        return x_, metadata
    def _validate_arrays(**X_):
        valid_X_ = {}
        metadata = {}
        sizes = []
        invalid_X_ = {}
        for i, (label, x_) in enumerate(X_.items()):
            if not isinstance(x_, np.ndarray):
                valid_X_[label] = None
                metadata[label] = None
                invalid_X_[label] = x_
                continue
            valid_X_[label], metadata_ = _make_proper_array(label, x_)
            metadata[label] = metadata_
            sizes.append(x_.shape[0])
        if np.unique(sizes).shape[0] != 1:
            raise ValueError('input arrays have inconsistent batch sizes')
        size = sizes[0]
        for label, x_ in invalid_X_.items():
            np_x_ = np.array(x_)
            if np_x_.shape == ():
                np_x_ = np_x_.reshape(1)
            new_shape = (size,) + tuple(np.ones(np_x_.ndim, dtype='int64'))
            print(f'Warning: input "{label}" is not a np array, it will be broadcasted into np.ndarray with shape {new_shape}')
            np_x_ = np.tile(np_x_, new_shape)
            valid_X_[label], metadata_ = _make_proper_array(label, np_x_)
            metadata[label] = metadata_
        metadata_ = {"features": metadata, "size": size}
        return valid_X_, metadata_
    valid_X, metadata = _validate_arrays(**X)
    feature_methods = {}
    for label, array in valid_X.items():
        feature_methods[label] = get_feature_method(array)
    for i in range(metadata['size']):
        feature = {}
        for label, method in feature_methods.items():
            feature[label] = method(valid_X[label][i])
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
    return metadata

def select_dataset_by_index(ds, indices):
    # Make a tensor of type tf.int64 to match the one by Dataset.enumerate(). 
    indices_ts = tf.constant(indices, dtype='int64')
    def is_index_in(index, rest):
        return tf.math.reduce_any(tf.math.equal(index, indices_ts))
    def drop_index(index, rest):
        return rest
    selected_ds = (ds
                   .enumerate()
                   .filter(is_index_in)
                   .map(drop_index))
    return selected_ds

def partition_dataset(ds, partition_sizes:Union[int, List[int]], total_size:Optional[int]=None,
                      stratify_map=None, shuffle:bool=True, seed:Optional[int]=None,
                      buffer_size:Optional[int]=None, partition_indices=None):
    est_size = ds.cardinality().numpy()
    if est_size < 0:
        if total_size is None:
            raise ValueError('total size must be given for TFRecordDataset')
    elif (total_size is None) and (est_size != total_size):
        raise ValueError('total size does not match cardinality of dataset')
    else:
        total_size = est_size
        
    if (isinstance(partition_sizes, Iterable) and 
        not isinstance(partition_sizes, dict)):
        split_sizes = {i: size for i, size in enumerate(partition_sizes)}
    else:
        split_sizes = partition_sizes

    if stratify_map is None:
        stratify = None
    else:
        stratify = np.array(list(ds.map(stratify_map)))

    split_indices = get_split_indices(total_size, split_sizes=split_sizes,
                                      stratify=stratify, shuffle=shuffle, seed=seed)
    ds_parts = {}
    for label, indices in split_indices.items():
        ds_part = select_dataset_by_index(ds, indices)
        if shuffle and (buffer_size is not None):
            if buffer_size < 0:
                buffer_size = len(indices)
            ds_part = ds_part.shuffle(buffer_size=buffer_size, seed=seed,
                                      reshuffle_each_iteration=False)
        ds_parts[label] = ds_part
    if (isinstance(partition_sizes, (Iterable, int)) and 
        not isinstance(partition_sizes, dict)):
        return tuple(ds_parts.values())
    return ds_parts
    

def split_dataset(ds, test_size=None, val_size=None, train_size=None,
                  total_size=None, stratify_map=None, shuffle_index:bool=True,
                  seed:int=None):
    est_size = ds.cardinality().numpy()
    if est_size < 0:
        if total_size is None:
            raise ValueError('total size must be given for TFRecordDataset')
    elif (total_size is None) and (est_size != total_size):
        raise ValueError('total size does not match cardinality of dataset')
    else:
        total_size = est_size
    split_sizes = get_train_val_test_split_sizes(total_size, train_size=train_size,
                                                 val_size=val_size, test_size=test_size)
    ds_splits = partition_dataset(ds, partition_sizes=split_sizes,
                                  total_size=total_size,
                                  stratify_map=stratify_map,
                                  shuffle=shuffle_index, seed=seed)   
    return ds_splits

"""
tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
tf.data.Dataset.list_files(pattern).interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE)

"""