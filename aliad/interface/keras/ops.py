try:
    from keras import ops
    from keras.ops import convert_to_tensor
    from keras.ops import slice as slice_op
except ImportError:
    from keras import backend as ops
    from tensorflow import convert_to_tensor
    from tensorflow import slice as slice_op

def transpose_last_n_dimensions(tensor, n: int):
    """
    Transposes the last N dimensions of a given tensor.

    Parameters
    ----------
    tensor : Tensor (Keras tensor)
        The input tensor to be transposed.
    n : int
        The number of last dimensions to transpose.

    Returns
    -------
    Tensor (Keras tensor)
        A tensor with the last N dimensions transposed.

    Examples
    --------
    >>> import tensorflow as tf
    >>> tensor = tf.constant([[[1, 2, 3], [4, 5, 6]],
    ...                       [[7, 8, 9], [10, 11, 12]],
    ...                       [[13, 14, 15], [16, 17, 18]]], dtype=tf.float32)
    >>> transposed_tensor = transpose_last_n_dimensions(tensor, 2)
    >>> print(transposed_tensor)
    <tf.Tensor: shape=(3, 3, 2), dtype=float32, numpy=
    array([[[ 1.,  4.],
            [ 2.,  5.],
            [ 3.,  6.]],
           [[ 7., 10.],
            [ 8., 11.],
            [ 9., 12.]],
           [[13., 16.],
            [14., 17.],
            [15., 18.]]], dtype=float32)>
    """
    if n <= 1:
        return tensor

    rank = ops.ndim(tensor)
    indices = ops.arange(rank)
    lead_indices = indices[: -n]
    trail_indices = indices[rank - n : rank]
    trail_indices = ops.reverse(trail_indices, 0)
    perm = ops.concatenate([lead_indices, trail_indices])
    return ops.permute_dimensions(tensor, perm)

def trim_elements(tensor, n: int, axis: int):
    """
    Removes elements from the beginning or the end along a specified axis of a tensor.

    Parameters
    ----------
    tensor : Tensor (Keras tensor)
        The input tensor from which elements are removed.
    n : int
        The number of elements to remove. If positive, removes the first `n` elements; 
        if negative, removes the last `n` elements.
    axis : int
        The axis along which elements will be removed. Supports negative values for axis indexing.

    Returns
    -------
    Tensor (Keras tensor)
        A tensor with the specified elements removed along the specified axis.

    Examples
    --------
    Remove elements along axis 2 (the last dimension):
    
    >>> import tensorflow as tf
    >>> tensor = tf.constant([[[1, 2, 3], [4, 5, 6]],
    ...                       [[7, 8, 9], [10, 11, 12]],
    ...                       [[13, 14, 15], [16, 17, 18]]], dtype=tf.float32)
    >>> modified_tensor = trim_elements(tensor, n=1, axis=2)
    >>> print(modified_tensor)
    <tf.Tensor: shape=(3, 2, 2), dtype=float32, numpy=
    array([[[ 2.,  3.],
            [ 5.,  6.]],
           [[ 8.,  9.],
            [11., 12.]],
           [[14., 15.],
            [17., 18.]]], dtype=float32)>

    Remove the first element along axis 0:

    >>> tensor = tf.constant([[[1, 2, 3], [4, 5, 6]],
    ...                       [[7, 8, 9], [10, 11, 12]],
    ...                       [[13, 14, 15], [16, 17, 18]]], dtype=tf.float32)
    >>> modified_tensor = trim_elements(tensor, n=1, axis=0)
    >>> print(modified_tensor)
    <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
    array([[[ 7.,  8.,  9.],
            [10., 11., 12.]],
           [[13., 14., 15.],
            [16., 17., 18.]]], dtype=float32)>

    Remove the last element along axis 1:

    >>> tensor = tf.constant([[[1, 2, 3], [4, 5, 6]],
    ...                       [[7, 8, 9], [10, 11, 12]],
    ...                       [[13, 14, 15], [16, 17, 18]]], dtype=tf.float32)
    >>> modified_tensor = trim_elements(tensor, n=-1, axis=1)
    >>> print(modified_tensor)
    <tf.Tensor: shape=(3, 1, 3), dtype=float32, numpy=
    array([[[ 1.,  2.,  3.]],
           [[ 7.,  8.,  9.]],
           [[13., 14., 15.]]], dtype=float32)>

    Remove elements along a negative axis:

    >>> tensor = tf.constant([[[1, 2, 3], [4, 5, 6]],
    ...                       [[7, 8, 9], [10, 11, 12]],
    ...                       [[13, 14, 15], [16, 17, 18]]], dtype=tf.float32)
    >>> modified_tensor = trim_elements(tensor, n=1, axis=-2)
    >>> print(modified_tensor)
    <tf.Tensor: shape=(3, 1, 3), dtype=float32, numpy=
    array([[[ 4.,  5.,  6.]],
           [[10., 11., 12.]],
           [[16., 17., 18.]]], dtype=float32)>
    """
    rank = ops.ndim(tensor)
    
    # Adjust for negative axis values
    if axis < 0:
        axis_ = rank + axis
    else:
        axis_ = axis

    tensor_shape = ops.shape(tensor)

    begin = [0] * rank
    
    if n >= 0:
        begin[axis_] = n

    begin = convert_to_tensor(begin, dtype_hint="int32")

    if n < 0:
        # Removing elements from the end
        size = ops.concatenate([tensor_shape[:axis_], [tensor_shape[axis_] + n], tensor_shape[axis_+1:]])
    else:
        # Removing elements from the beginning
        size = ops.concatenate([tensor_shape[:axis_], [tensor_shape[axis_] - n], tensor_shape[axis_+1:]])

    return slice_op(tensor, begin, size)

def generate_batch_indices(tensor):
    """
    Generates a tensor of batch indices for a given input tensor.

    Parameters
    ----------
    tensor : Tensor (Keras tensor)
        An input tensor of shape `(batch_size, dim_1, dim_2, ..., dim_n)`. Must be at least 2-dimensional.

    Returns
    -------
    Tensor (Keras tensor)
        A tensor of batch indices with the same batch size as the input tensor.

    Examples
    --------
    >>> import tensorflow as tf
    >>> tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    >>> batch_indices = generate_batch_indices(tensor)
    >>> print(batch_indices)
    <tf.Tensor: shape=(3, 2, 2), dtype=int32, numpy=
    array([[[0, 0],
            [0, 0]],
           [[1, 1],
            [1, 1]],
           [[2, 2],
            [2, 2]]], dtype=int32)>
    """
    batch_size = ops.shape(tensor)[0]
    range_indices = ops.arange(0, batch_size)
    reshaped_indices = ops.reshape(range_indices, ops.stack([batch_size] + [1] * (ops.ndim(tensor) - 1)))
    tiling_multiples = ops.concatenate([convert_to_tensor([1]), ops.shape(tensor)[1:]])
    return ops.tile(reshaped_indices, tiling_multiples)


def merge_dimensions(tensor, begin: int, end: int):
    """
    Reshapes a tensor by merging a specified range of dimensions.

    Parameters
    ----------
    tensor : Tensor (Keras tensor)
        A tensor with dimensions to be merged.
    begin : int
        The start index of the dimensions to be merged. Negative values count from the end.
    end : int
        The end index (inclusive) of the dimensions to be merged. Negative values count from the end.

    Returns
    -------
    Tensor (Keras tensor)
        A tensor with the same elements as the input tensor but with the specified range of dimensions merged.

    Examples
    --------
    >>> import tensorflow as tf
    >>> tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    >>> reshaped_tensor = merge_dimensions(tensor, -3, -2)
    >>> print(reshaped_tensor)
    <tf.Tensor: shape=(6, 2), dtype=int32, numpy=
    array([[ 1,  2],
           [ 3,  4],
           [ 5,  6],
           [ 7,  8],
           [ 9, 10],
           [11, 12]], dtype=int32)>
    """
    tensor_shape = ops.shape(tensor)
    rank = ops.ndim(tensor)

    begin = ops.switch(ops.less(begin, 0), begin + rank, begin)
    end = ops.switch(ops.less(end, 0), end + rank, end)
    merged_size = ops.prod(tensor_shape[begin:end+1])
    new_shape = ops.concatenate([tensor_shape[:begin], ops.stack([merged_size]), tensor_shape[end+1:]])
    return ops.reshape(tensor, new_shape)