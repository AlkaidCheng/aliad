import tensorflow as tf

def transpose_last_n_dimensions(tensor: tf.Tensor, n: int) -> tf.Tensor:
    """
    Transposes the last N dimensions of a given tensor.

    Args:
    tensor (tf.Tensor): The input tensor.
    n (int): The number of last dimensions to transpose.

    Returns:
    tf.Tensor: A tensor with the last N dimensions transposed.

    Example:
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
    
    rank = tf.rank(tensor)
    indices = tf.range(rank)
    lead_indices = indices[: -n]
    trail_indices = indices[rank - n : rank]
    trail_indices = tf.reverse(trail_indices, axis=[0])
    perm = tf.concat([lead_indices, trail_indices], axis=0)
    return tf.transpose(tensor, perm=perm)

def trim_elements(tensor: tf.Tensor, n: int, axis: int) -> tf.Tensor:
    """
    Removes elements from the beginning or the end along a specified axis of a tensor.

    The function creates a new tensor with the first n elements removed along the specified axis
    if n is positive, or the last n elements removed if n is negative.
    It supports negative axis values, interpreting them as counting from the end of the tensor dimensions.

    Args:
        tensor (tf.Tensor): The input tensor.
        n (int): The number of elements to remove. If n is positive, removes the first n elements; 
                 if n is negative, removes the last n elements.
        axis (int): The axis along which to remove the elements. Supports negative values.

    Returns:
        tf.Tensor: A tensor with the specified elements removed along the specified axis.

    Example:
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
    """
    # Compute the tensor's rank to handle negative axis values and ensure axis is within bounds
    rank = tf.rank(tensor)
    if axis < 0:
        axis_ = rank + axis
    else:
        axis_ = axis
    if n < 0:
        begin = tf.zeros(rank, dtype=tf.int32)
    else:
        begin = tf.one_hot(axis_, rank, on_value=n, dtype=tf.int32)
    tensor_shape = tf.shape(tensor)
    # very hacky right now
    size = tf.concat([tensor_shape[:axis_],
                      [tensor_shape[axis] - n],
                      tensor_shape[axis_+1:]], axis=0)
    # Create and return the trimmed tensor
    return tf.slice(tensor, begin=begin, size=size)

def generate_batch_indices(tensor: tf.Tensor) -> tf.Tensor:
    """
    Generates a tensor of batch indices for a given input tensor.

    Each element in the output tensor represents the batch index of the corresponding element in the input tensor.
    
    Args:
        tensor (tf.Tensor): An input tensor of shape (batch_size, dim_1, dim_2, ..., dim_n).
            It must be at least 2-dimensional.

    Returns:
        tf.Tensor: A tensor of batch indices with the same shape as the input tensor.

    Example:
        >>> tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
        >>> batch_indices = generate_batch_indices(tensor)
        >>> print(batch_indices)
        <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
        array([[[0, 0]],
               [[1, 1]],
               [[2, 2]]], dtype=int32)>
    """
    batch_size = tf.shape(tensor)[0]
    range_indices = tf.range(batch_size)
    # super hacky for the moment
    new_shape = tf.concat([[batch_size], tf.sign(tf.shape(tensor)[1:])], axis=0)
    reshaped_indices = tf.reshape(range_indices, new_shape)
    tiling_multiples = tf.concat([[1], tf.shape(tensor)[1:]], axis=0)
    batch_indices = tf.tile(reshaped_indices, tiling_multiples)
    return batch_indices

def merge_dimensions(tensor: tf.Tensor, begin: int, end: int) -> tf.Tensor:
    """
    Reshapes a tensor by merging a specified range of dimensions.
    
    Args:
        tensor (tf.Tensor): A tensor with dimensions to be merged.
        begin (int): The start index of the dimensions to be merged. Negative values count from the end.
        end (int): The end index (inclusive) of the dimensions to be merged. Negative values count from the end.
        
    Returns:
        tf.Tensor: A tensor with the same elements as the input tensor, but with the specified range of dimensions merged.
        
    Example:
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
    tensor_shape = tf.shape(tensor)
    rank = tf.rank(tensor)
    begin = tf.where(begin < 0, begin + rank, begin)
    end = tf.where(end < 0, end + rank, end)
    merged_size = tf.reduce_prod(tensor_shape[begin:end+1])
    new_shape = tf.concat([tensor_shape[:begin], [merged_size], tensor_shape[end+1:]], axis=0)
    return tf.reshape(tensor, new_shape)
