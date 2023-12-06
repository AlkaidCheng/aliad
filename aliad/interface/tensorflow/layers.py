from typing import Optional, Tuple, List

import tensorflow as tf

from tensorflow.keras.layers import (BatchNormalization, Activation,
                                     Conv2D, Conv3D, Dense, Dropout)

from .operations import (transpose_last_n_dimensions, trim_elements,
                         generate_batch_indices, merge_dimensions)

def DMatrix(A:tf.Tensor, B:tf.Tensor, name:str="DMatrix") -> tf.Tensor:
    """

      Args:
        A: A `Tensor` with shape (batch_dims ..., nobjects, ncoords)
        B: A `Tensor` with same shape as `A`

      Returns:
        A `Tensor` with shape (batch_dims ..., nobjects, nobjects)
        
    """
    with tf.name_scope(name):
        r_A = tf.reduce_sum(A * A, axis=-1, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=-1, keepdims=True)
        m = tf.matmul(A, transpose_last_n_dimensions(B, 2))
        D = r_A - 2 * m + transpose_last_n_dimensions(r_B, 2)
        return D

def SymDMatrix(A:tf.Tensor, name:str="SymDMatrix") -> tf.Tensor:
    with tf.name_scope(name):
        r = tf.reduce_sum(A * A, axis=-1, keepdims=True)
        m = tf.matmul(A, transpose_last_n_dimensions(A, 2))
        D = r - 2 * m + transpose_last_n_dimensions(r, 2)
        return D

def KNN(points: tf.Tensor, K: int, name: str = "KNN") -> tf.Tensor:
    """
    Implements the K-Nearest Neighbors (KNN) algorithm to find the indices of neighboring points.

    Given a tensor representing a batch of point clouds, this function calculates the pairwise distance 
    between points within each point cloud in the batch and identifies the K nearest neighbors for each point.

    Args:
        points (tf.Tensor): A tensor of shape (batch_dims ..., num_points, num_dimensions) representing 
            a batch of point clouds.
            - batch_dims: The dimensions representing different batches. The tensor can have any number of batch dimensions.
            - num_points: The number of points in each point cloud.
            - num_dimensions: The number of dimensions for each point.
        K (int): The number of nearest neighbors to identify for each point.
        name (str, optional): A name for the operation (optional). Defaults to "KNN".

    Returns:
        tf.Tensor: A tensor of shape (merged batch_dims, num_points, K, 2), where the last dimension contains 
                   pairs (batch_index, point_index). Here, `point_index` is the index of the neighbor point 
                   in the flattened version of the point cloud.

    Example:
        >>> points = tf.random.normal((32, 1024, 3))  # 32 batches, each with 1024 points in 3D space
        >>> neighbor_indices = KNN(points, K=16)
        >>> print(neighbor_indices.shape)  # Should print (32 * 1024, 16, 2)

    Note:
        - The input points are assumed to be represented in a Cartesian coordinate system.
        - The function is designed to work with point clouds of any dimensionality (D >= 3), with the first D - 2 
          dimensions representing the batch dimensions.
    """
    with tf.name_scope(name):
        # Calculate pairwise distance matrix
        D = SymDMatrix(points)
        
        # Find the indices of the K+1 nearest neighbors (including the point itself)
        _, indices = tf.nn.top_k(-D, k=K + 1)
        
        # Remove the index corresponding to the point itself
        indices = trim_elements(indices, n=1, axis=-1)

        # Merge the batch dimensions
        indices = merge_dimensions(indices, 0, -3)
            
        # Generate batch indices
        batch_indices = generate_batch_indices(indices)
        
        # Concatenate batch indices and neighbor indices
        indices = tf.concat([tf.expand_dims(batch_indices, -1), tf.expand_dims(indices, -1)], axis=-1)
        
        return indices

def KNNFeature(points: tf.Tensor, features:tf.Tensor, K: int, name: str = "KNNFeature") -> tf.Tensor:
    indices_rs = KNN(points, K)
    # Merge the batch dimensions
    features_rs = merge_dimensions(features, 0, -3)
    knn_fts = tf.gather_nd(features_rs, indices_rs)
    # Recover the batch dimensions
    feature_shape = tf.shape(features)
    knn_fts_shape = tf.concat([feature_shape[:-1], [K], [feature_shape[-1]]], axis=0)
    knn_fts = tf.reshape(knn_fts, knn_fts_shape)
    return knn_fts

def EdgeConv(points, features,
             channels:Tuple[int],
             K:int=16,
             batchnorm:bool=True,
             rel_fts:bool=True,
             activation:str='relu',
             pooling:str='average',
             name:str='EdgeConv',
             conv_type:str='3D'):
    """
    Args
        channels: Tuple of int
            Number of output channels in each convolution
        points: Tensor of shape (nevent, njet, nparticle, ncoords)
            Coordinates of the point cloud particles
        features: Tensor of shape (nevent, njet, nparticle, nfeatures)
            Features of the point cloud particles
        num_points: int
            Number of partcles in the point cloud
        num_jets: int
            Number of jets in an event
        K: int
            Number of nearest-neighbors
    """
    with tf.name_scope(name):
        knn_fts = KNNFeature(points, features, K)
        fts_rank = tf.rank(features)
        tile_shape = tf.one_hot(fts_rank - 1, fts_rank + 1, K, 1)
        knn_fts_center = tf.tile(tf.expand_dims(features, axis=-2), tile_shape)
        if rel_fts:
            knn_fts = tf.subtract(knn_fts, knn_fts_center)
        knn_fts = tf.concat([knn_fts_center, knn_fts], axis=-1)
            
        x = knn_fts
        use_bias = False if batchnorm else True
        if conv_type == '2D':
            conv_fn = Conv2D
            kernal_size = (1, 1)
        elif conv_type == '3D':
            conv_fn = Conv3D
            kernal_size = (1, 1, 1)
        else:
            raise ValueError('conv type must be either "2D" or "3D"')
        for idx, channel in enumerate(channels):
            x = conv_fn(channel, kernel_size=kernal_size, strides=1,
                        data_format='channels_last',
                        use_bias=use_bias,
                        kernel_initializer='glorot_normal',
                        name=f'{name}_conv{idx}')(x)
            if batchnorm:
                x = BatchNormalization(name=f'{name}_bn{idx}')(x)
            if activation:
                x = Activation(activation, name=f'{name}_act{idx}')(x)
        # shape after pooling: (batch dims..., npoints, nchannel)
        if pooling == 'max':
            fts = tf.reduce_max(x, axis=-2)
        else:
            fts = tf.reduce_mean(x, axis=-2)

        # shortcut
        sc = conv_fn(channels[-1], kernel_size=kernal_size, strides=1,
                     data_format='channels_last',
                     use_bias=use_bias,
                     kernel_initializer='glorot_normal',
                     name=f'{name}_sc_conv')(tf.expand_dims(features, axis=-2))
        if batchnorm:
            sc = BatchNormalization(name=f'{name}_sc_bn')(sc)
        sc = tf.squeeze(sc, axis=-2)
        x = tf.add(sc, fts)
        # shape: (batch dims..., npoints, nchannel)
        if activation:
            return Activation(activation, name=f'{name}_sc_act')(x)
        else:
            return x
