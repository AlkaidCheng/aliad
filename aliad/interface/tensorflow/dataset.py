from functools import partial
import json
from typing import Optional, Tuple, Union, Iterable, Callable, Dict, List, Any

import numpy as np
import tensorflow as tf

from tensorflow.keras import Input

from quickstats import VerbosePrint
from quickstats.utils.string_utils import get_field_names
from aliad.data.partition import (
    get_split_indices, get_train_val_test_split_sizes,
    get_partition_ranges
)

def apply_pipelines(ds:tf.data.Dataset,
                    batch_size: Optional[int] = None,
                    shuffle: bool = True,
                    seed: Optional[int] = None,
                    drop_remainder: bool = False,
                    buffer_size: Optional[int] = None,
                    cache:bool = False,
                    prefetch:bool = True,
                    repeat:bool = False,
                    reshuffle_each_iteration:bool=False,
                    distribute_strategy=None) -> tf.data.Dataset:
    if cache:
        ds = ds.cache()

    if shuffle:
        if buffer_size is None:
            sample_size = ds.cardinality().numpy()
            if sample_size < 0:
                raise RuntimeError('buffer_size must be given when shuffle is True while'
                                   'cardinality of dataset can not be determined')
            buffer_size = sample_size            
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed,
                        reshuffle_each_iteration=reshuffle_each_iteration)

    if batch_size is not None:
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if repeat or (distribute_strategy is not None):
        ds = ds.repeat()

    if prefetch and (distribute_strategy is None):
        ds = ds.prefetch(tf.data.AUTOTUNE)

    if (distribute_strategy is not None):
        ds = distribute_strategy.experimental_distribute_dataset(ds)
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

def feature_selector(inner_slices:List[str], outer_slices:List[str]):
    def map_fn(X):
        return tuple(X[ins] for ins in inner_slices), *(X[outs] for outs in outer_slices)
    return map_fn

def get_dtype_str(dtype):
    if isinstance(dtype, str):
        return dtype
    return dtype.name

def get_symbolic_inputs(metadata_map:Dict, downcast:bool=False,
                        keys:Optional[List[str]]=None):
    """symbolic tensor or placeholders"""
    inputs = {}
    if keys is None:
        keys = list(metadata_map.keys())
    for key in keys:
        metadata = metadata_map[key]
        if downcast and (metadata['dtype'] == 'float64'):
            metadata = dict(metadata)
            metadata['dtype'] = 'float32'
        inputs[key] = Input(name=key, **metadata)
    return inputs
        
def get_tfrecord_array_parser(feature_metadata:Dict, downcast:bool=False,
                              keys:Optional[List[str]]=None):
    keys = keys or list(feature_metadata)
    inputs = get_symbolic_inputs(feature_metadata, keys=keys)
    feature_description = {}
    feature_parser = {}
    for label, input_ in inputs.items():
        feature_description[label] = get_feature_description(input_)
        dtype = input_.dtype
        dtype_str = get_dtype_str(dtype)
        shape = input_.shape[1:]
        if dtype_str in ['bool', 'float64']:
            if downcast and dtype_str == 'float64':
                feature_parser[label] = (lambda example, shape=shape:
                                         tf.cast(tf.reshape(tf.io.decode_raw(example, out_type='float64'), shape), dtype='float32'))
            else:
                feature_parser[label] = (lambda example, out_type=dtype, shape=shape:
                                         tf.reshape(tf.io.decode_raw(example, out_type=out_type), shape))
        else:
            feature_parser[label] = (lambda example, shape=shape:
                                     tf.reshape(example, shape))
    
    def get_parsed_example(example):
        parsed_example = tf.io.parse_single_example(example, feature_description)
        for label, input_ in inputs.items():
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
            metadata[label] = {'shape': tensor.shape.as_list(), 'dtype': get_dtype_str(tensor.dtype)}
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

def get_feature_description(array: tf.Tensor):
    dtype_str = get_dtype_str(array.dtype)
    if dtype_str in ['float32', 'int64']:
        return tf.io.FixedLenFeature([np.prod(array.shape[1:])], dtype=array.dtype)
    elif dtype_str in ['float64', 'bool']:
        return tf.io.FixedLenFeature([], dtype=tf.string)
    else:
        raise ValueError('array must have dtype of float32, float64, or int64')

def float_feature(array: np.ndarray) -> tf.train.Feature:
    """
    Converts a numpy array of floats to a TensorFlow float_list Feature.

    Parameters
    ----------
    array : np.ndarray
        Input array with dtype float32.

    Returns
    -------
    tf.train.Feature
        A TensorFlow Feature containing the float list.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=array))

def bytes_feature(array: np.ndarray) -> tf.train.Feature:
    """
    Converts a numpy array to a TensorFlow bytes_list Feature.

    Parameters
    ----------
    array : np.ndarray
        Input array with any dtype.

    Returns
    -------
    tf.train.Feature
        A TensorFlow Feature containing the bytes list.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[array.tobytes()]))

def int64_feature(array: np.ndarray) -> tf.train.Feature:
    """
    Converts a numpy array of int64s to a TensorFlow int64_list Feature.

    Parameters
    ----------
    array : np.ndarray
        Input array with dtype int64.

    Returns
    -------
    tf.train.Feature
        A TensorFlow Feature containing the int64 list.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=array))

def get_feature_method(array:np.ndarray):
    """Match appropriate tf.train.Feature class with dtype of an array. """
    dtype_str = get_dtype_str(array.dtype)
    if dtype in ['float32']:
        # note FloatList converts float/double to float
        return float_feature
    elif dtype in ['float64', 'bool']:
        return bytes_feature
    elif dtype in ['int64']:
        return int64_feature
    else:  
        raise ValueError('array must have dtype of float32, float64, or int64')

def write_tfrecord(writer: tf.io.TFRecordWriter, **X: Any) -> Dict[str, Any]:
    """
    Write array data to a TFRecord file.

    Parameters
    ----------
    writer : tf.io.TFRecordWriter
        A TFRecordWriter instance.
    **X : Any
        Key-value pairs where the key is the label and the value is the data.

    Returns
    -------
    dict
        A dictionary containing metadata about the features.
    """
    
    def _regularize_array(x: np.ndarray) -> (np.ndarray, Dict[str, Any]):
        if x.ndim == 1:
            x = x.reshape((x.shape[0], 1))
        shape = x.shape[1:]
        # contract to 2D, will be reshaped after decoding
        if x.ndim > 2:
            x = x.reshape((x.shape[0], np.prod(x.shape[1:])))
        dtype_str = get_dtype_str(x.dtype)
        if dtype not in ['float32', 'float64', 'bool', 'int64']:
            raise ValueError('Array must have dtype of float32, float64, bool or int64')
        feature_metadata = {"shape": shape, "dtype": dtype}
        return x, feature_metadata

    valid_X = {}
    invalid_X = {}
    metadata = {"features": {}}
    sizes = []

    for label, x in X.items():
        if not isinstance(x, np.ndarray):
            invalid_X[label] = x
            valid_X[label] = None
            metadata["features"][label] = None
            continue
        valid_X[label], feature_metadata = _regularize_array(x)
        metadata["features"][label] = feature_metadata
        sizes.append(x.shape[0])

    if np.unique(sizes).shape[0] != 1:
        raise ValueError('Arrays have inconsistent batch sizes')
    size = sizes[0]
    metadata['size'] = size

    # forcing all arrays to have the same batchsize
    for label, x in invalid_X.items():
        x = np.array(x)
        if x.shape == ():
            x = x.reshape(1)
        new_shape = (size,) + tuple(np.ones(x.ndim, dtype='int64'))
        tiled_x = np.tile(x, new_shape)
        valid_X[label], feature_metadata = _regularize_array(tiled_x)
        metadata["features"][label] = feature_metadata

    feature_methods = {label: get_feature_method(x) for label, x in valid_X.items()}

    # bottleneck is here
    for i in range(size):
        feature = {label: method(valid_X[label][i]) for label, method in feature_methods.items()}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()
        writer.write(serialized)

    return metadata

def arrays_to_sharded_tfrecords(filename_fmt:str, num_shards:int, parallel:int=0, **X):
    if "shard_index" not in get_field_names(filename_fmt):
        raise ValueError('filename_fmt must contain the field name "shard_index"')
    array_sizes = [len(x) for x in X.values()]
    if np.unique(array_sizes).shape[0] != 1:
        raise ValueError('input arrays have inconsistent batch sizes')
    array_size = array_sizes[0]
    partition_ranges = get_partition_ranges(array_size, num_shards)
    part_X = []
    for i, prange in enumerate(partition_ranges):
        part_x = {key:x[prange[0]:prange[1]] for key, x in X.items()}
        part_X.append(part_x)

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

def count_elements(dataset: tf.data.Dataset):
    return dataset.reduce(0, lambda x, _ : x + 1).numpy()

def concatenate_datasets(datasets: List[tf.data.Dataset]):
    ds = datasets[0]
    for ds_i in datasets[1:]:
        ds = ds.concatenate(ds_i)
    return ds

def get_tfrecord_dataset(filenames:List[str], parse_fn,
                         dataset_options:Optional[Dict]=None,
                         map_options:Optional[Dict]=None):
    if dataset_options is None:
        dataset_options = {}
    if map_options is None:
        map_options = {}
    if 'num_parallel_reads' not in dataset_options:
        dataset_options['num_parallel_reads'] = tf.data.AUTOTUNE
    if 'num_parallel_calls' not in map_options:
        map_options['num_parallel_calls'] = tf.data.AUTOTUNE
    ds = tf.data.TFRecordDataset(filenames, **dataset_options)
    ds = ds.map(parse_fn, **map_options)
    return ds