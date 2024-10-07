from typing import Optional, Tuple, List, Union
from numbers import Number

import keras
from keras import Input
from keras.layers import (
    Layer,
    Conv2D,
    Conv3D,
    Activation,
    BatchNormalization
)
from keras.models import Model
from keras.initializers import GlorotNormal
from keras import initializers, regularizers, constraints

try:
    from keras import ops
    from keras.ops import ones_like, top_k, add, subtract, matmul, concat
except ImportError:
    from keras.backend import ones_like
    from keras import backend as ops
    from tensorflow.math import top_k, add, subtract
    from tensorflow import matmul, concat

# to fix: keras does not have gather, gather_nd and one_hot yet
import tensorflow as tf
    
from aliad.core.assertions import assert_range
from .ops import (
    transpose_last_n_dimensions,
    trim_elements,
    merge_dimensions,
    generate_batch_indices
)
from .regularizers import MinMaxRegularizer

class DMatrix(Layer):
    """
    Computes the pairwise squared Euclidean distance matrix between points in tensors `A` and `B`.

    Parameters
    ----------
    A : tf.Tensor
        A tensor of shape (batch_dims ..., nobjects, ncoords), where nobjects is the number of objects
        and ncoords is the number of coordinates (dimensions) per object.
    B : tf.Tensor
        A tensor with the same shape as `A`.

    Returns
    -------
    tf.Tensor
        A tensor of shape (batch_dims ..., nobjects, nobjects) representing the pairwise distances 
        between points in `A` and `B`.

    Example
    -------
    >>> A = tf.random.normal((32, 1024, 3))  # 32 batches, 1024 points, 3 coordinates per point
    >>> B = tf.random.normal((32, 1024, 3))  # Same shape as A
    >>> result = DMatrix()(A, B)
    >>> print(result.shape)  # Output: (32, 1024, 1024)
    """
    
    def __init__(self, **kwargs):
        super(DMatrix, self).__init__(**kwargs)

    def call(self, A: "tf.Tensor", B: "tf.Tensor") -> "tf.Tensor":
        """
        Computes the pairwise squared Euclidean distance matrix between points in tensors `A` and `B`.

        Parameters
        ----------
        A : tf.Tensor
            A tensor of shape (batch_dims ..., nobjects, ncoords), where nobjects is the number of objects
            and ncoords is the number of coordinates (dimensions) per object.
        B : tf.Tensor
            A tensor with the same shape as `A`.

        Returns
        -------
        tf.Tensor
            A tensor of shape (batch_dims ..., nobjects, nobjects) representing the pairwise distances 
            between points in `A` and `B`.
        """
        r_A = ops.sum(A * A, axis=-1, keepdims=True)
        r_B = ops.sum(B * B, axis=-1, keepdims=True)
        m = matmul(A, transpose_last_n_dimensions(B, 2))
        D = r_A - 2 * m + transpose_last_n_dimensions(r_B, 2)
        return D

class SymDMatrix(Layer):
    """
    Computes the pairwise squared Euclidean distance matrix for points in tensor `A`.

    Parameters
    ----------
    A : tf.Tensor
        A tensor of shape (batch_dims ..., nobjects, ncoords), where nobjects is the number of objects
        and ncoords is the number of coordinates (dimensions) per object.

    Returns
    -------
    tf.Tensor
        A tensor of shape (batch_dims ..., nobjects, nobjects) representing the pairwise distances 
        between points in `A`.

    Example
    -------
    >>> A = tf.random.normal((32, 1024, 3))  # 32 batches, 1024 points, 3 coordinates per point
    >>> result = SymDMatrix()(A)
    >>> print(result.shape)  # Output: (32, 1024, 1024)
    """

    def __init__(self, **kwargs):
        super(SymDMatrix, self).__init__(**kwargs)

    def call(self, A: "tf.Tensor") -> "tf.Tensor":
        """
        Computes the pairwise squared Euclidean distance matrix for points in tensor `A`.

        Parameters
        ----------
        A : tf.Tensor
            A tensor of shape (batch_dims ..., nobjects, ncoords), where nobjects is the number of objects
            and ncoords is the number of coordinates (dimensions) per object.

        Returns
        -------
        tf.Tensor
            A tensor of shape (batch_dims ..., nobjects, nobjects) representing the pairwise distances 
            between points in `A`.
        """
        r = ops.sum(A * A, axis=-1, keepdims=True)
        m = matmul(A, transpose_last_n_dimensions(A, 2))
        D = r - 2 * m + transpose_last_n_dimensions(r, 2)
        return D

class KNN(Layer):
    """
    Implements the K-Nearest Neighbors (KNN) algorithm to find the indices of neighboring points.

    Given a tensor representing a batch of point clouds, this layer calculates the pairwise distance 
    between points within each point cloud in the batch and identifies the K nearest neighbors for each point.

    Parameters
    ----------
    points : tf.Tensor
        A tensor of shape (batch_dims ..., num_points, num_dimensions) representing a batch of point clouds.
        - batch_dims: The dimensions representing different batches. The tensor can have any number of batch dimensions.
        - num_points: The number of points in each point cloud.
        - num_dimensions: The number of dimensions for each point.
    K : int
        The number of nearest neighbors to identify for each point.

    Returns
    -------
    tf.Tensor
        A tensor of shape (merged batch_dims, num_points, K, 2), where the last dimension contains 
        pairs (batch_index, point_index). Here, `point_index` is the index of the neighbor point 
        in the flattened version of the point cloud.

    Example
    -------
    >>> points = tf.random.normal((32, 1024, 3))  # 32 batches, each with 1024 points in 3D space
    >>> neighbor_indices = KNN(K=16)(points)
    >>> print(neighbor_indices.shape)  # Output: (32 * 1024, 16, 2)

    Notes
    -----
    - The input points are assumed to be represented in a Cartesian coordinate system.
    - The function is designed to work with point clouds of any dimensionality (D >= 3), with the first D - 2 
      dimensions representing the batch dimensions.
    """

    def __init__(self, K: int, **kwargs):
        super(KNN, self).__init__(**kwargs)
        self.K = K

    def call(self, points: "tf.Tensor") -> "tf.Tensor":
        """
        Performs the k-nearest neighbors computation and returns the indices of the K nearest neighbors for each point.
        
        Parameters
        ----------
        points : tf.Tensor
            A tensor of shape (batch_dims ..., num_points, num_dimensions) representing a batch of point clouds.

        Returns
        -------
        tf.Tensor
            A tensor of shape (merged batch_dims, num_points, K, 2) containing pairs (batch_index, point_index).
        """
        # Calculate pairwise distance matrix
        D = SymDMatrix()(points)
        
        # Find the indices of the K+1 nearest neighbors (including the point itself)
        _, indices = top_k(-D, k=self.K + 1)
        
        # Remove the index corresponding to the point itself (the point is its own nearest neighbor)
        indices = trim_elements(indices, n=1, axis=-1)

        # Merge the batch dimensions to create a flat structure
        indices = merge_dimensions(indices, 0, -3)
        
        # Generate batch indices to accompany the neighbor indices
        batch_indices = generate_batch_indices(indices)
        
        # Concatenate batch indices and neighbor indices
        indices = concat([ops.expand_dims(batch_indices, -1), ops.expand_dims(indices, -1)], axis=-1)
        
        return indices

class KNNFeature(Layer):
    """
    Implements the K-Nearest Neighbors (KNN) algorithm to find the features of neighboring points.

    Given a tensor representing a batch of point clouds and their associated features, this layer calculates 
    the K nearest neighbors for each point and returns the corresponding features.

    Parameters
    ----------
    points : tf.Tensor
        A tensor of shape (batch_dims ..., num_points, num_dimensions) representing a batch of point clouds.
        - batch_dims: The dimensions representing different batches. The tensor can have any number of batch dimensions.
        - num_points: The number of points in each point cloud.
        - num_dimensions: The number of dimensions for each point.
    features : tf.Tensor
        A tensor of shape (batch_dims ..., num_points, feature_dim) representing the features of each point in the point cloud.
        - feature_dim: The number of features for each point.
    K : int
        The number of nearest neighbors to identify for each point.

    Returns
    -------
    tf.Tensor
        A tensor of shape (batch_dims ..., num_points, K, feature_dim) representing the features of the K nearest neighbors for each point.

    Example
    -------
    >>> points = tf.random.normal((32, 1024, 3))  # 32 batches, each with 1024 points in 3D space
    >>> features = tf.random.normal((32, 1024, 64))  # 64-dimensional features for each point
    >>> knn_features = KNNFeature(K=16)(points, features)
    >>> print(knn_features.shape)  # Output: (32, 1024, 16, 64)

    Notes
    -----
    - The input points are assumed to be represented in a Cartesian coordinate system.
    - The function is designed to work with point clouds of any dimensionality (D >= 3), with the first D - 2 
      dimensions representing the batch dimensions.
    """

    def __init__(self, K: int, **kwargs):
        super(KNNFeature, self).__init__(**kwargs)
        self.K = K

    def call(self, points: tf.Tensor, features: tf.Tensor) -> tf.Tensor:
        """
        Finds the K nearest neighbors for each point in the point cloud and returns the corresponding features.

        Parameters
        ----------
        points : tf.Tensor
            A tensor of shape (batch_dims ..., num_points, num_dimensions) representing a batch of point clouds.
        features : tf.Tensor
            A tensor of shape (batch_dims ..., num_points, feature_dim) representing the features for each point.
        K : int
            The number of nearest neighbors to identify for each point.

        Returns
        -------
        tf.Tensor
            A tensor of shape (batch_dims ..., num_points, K, feature_dim) containing the features of the K nearest neighbors for each point.
        """
        # Get the indices of the K nearest neighbors
        knn_indices = KNN(K=self.K)(points)

        # Merge the batch dimensions to prepare for gather operation
        fts_merged = merge_dimensions(features, 0, -3)

        # Gather the features of the K nearest neighbors
        knn_fts = tf.gather_nd(fts_merged, knn_indices)

        # Recover the original batch dimensions
        feature_shape = ops.shape(features)
        knn_fts_shape = concat([feature_shape[:-1], [self.K], [feature_shape[-1]]], axis=0)
        knn_fts = ops.reshape(knn_fts, knn_fts_shape)

        return knn_fts

class EdgeConv(Layer):
    """
    Performs edge convolutions on point cloud data.
    
    This layer applies k-nearest neighbors (KNN) and convolutions to extract local features from point clouds.

    Parameters
    ----------
    channels : Tuple[int]
        Number of output channels in each convolution.
    points : tf.Tensor
        Tensor of shape (batch_dims, <num_clouds>, num_points, num_coords) representing the coordinates of the point cloud particles.
    features : tf.Tensor
        Tensor of shape (batch_dims, <num_clouds>, num_points, num_features) representing the features of the point cloud particles.
    K : int, optional
        Number of nearest neighbors, by default 16.
    batchnorm : bool, optional
        Whether to apply batch normalization, by default True.
    rel_fts : bool, optional
        Whether to compute relative features between points and their neighbors, by default True.
    activation : str, optional
        Activation function to use, by default 'relu'.
    pooling : str, optional
        Pooling method to use ('average' or 'max'), by default 'average'.
    seed : Optional[int], optional
        Random seed for kernel initialization, by default None.

    Returns
    -------
    tf.Tensor
        The output tensor of shape (batch_dims ..., num_points, out_channels).

    Example
    -------
    >>> points = tf.random.normal((32, 1024, 3))  # 32 batches, 1024 points, 3D coordinates
    >>> features = tf.random.normal((32, 1024, 64))  # 64-dimensional features
    >>> edge_conv = EdgeConv(channels=(64, 128), K=16)
    >>> output = edge_conv(points, features)
    >>> print(output.shape)  # Output: (32, 1024, 128)
    """
    
    def __init__(self, channels, K=16, batchnorm=True, rel_fts=True, 
                 activation='relu', pooling='average', conv_type='3D',
                 seed=None, **kwargs):
        super(EdgeConv, self).__init__(**kwargs)
        
        if conv_type == '2D':
            conv_fn = Conv2D
            kernel_size = (1, 1)
        elif conv_type == '3D':
            conv_fn = Conv3D
            kernel_size = (1, 1, 1)
        else:
            raise ValueError('conv_type must be either "2D" or "3D"')

        if pooling in ['max']:
            pool_fn = tf.reduce_max
        elif pooling in ['average', 'mean']:
            pool_fn = tf.reduce_mean
        else:
            raise ValueError('pooling must be "max" or "average"("mean")')
            
        self.channels = channels
        self.K = K
        self.batchnorm = batchnorm
        self.rel_fts = rel_fts
        self.activation = activation
        self.seed = seed
        self.conv_fn = conv_fn
        self.kernel_size = kernel_size
        self.pool_fn = pool_fn

    def call(self, points, features):
        """
        Applies edge convolution to the input points and features.
        
        Parameters
        ----------
        points : tf.Tensor
            Tensor of shape (batch_dims ..., num_points, num_coords) representing the point coordinates.
        features : tf.Tensor
            Tensor of shape (batch_dims ..., num_points, num_features) representing the point features.
        
        Returns
        -------
        tf.Tensor
            Output tensor after applying the edge convolution.
        """
        # Get K-nearest neighbors' features
        knn_fts = KNNFeature(K=self.K)(points, features)

        # Get the number of feature dimensions
        fts_rank = tf.rank(features)
        tile_shape = tf.one_hot(fts_rank - 1, fts_rank + 1, self.K, 1)
        knn_fts_center = ops.tile(ops.expand_dims(features, axis=-2), tile_shape)

        # Optionally compute relative features
        if self.rel_fts:
            knn_fts = subtract(knn_fts, knn_fts_center)
        
        # Concatenate center and relative features
        knn_fts = concat([knn_fts_center, knn_fts], axis=-1)

        if fts_rank == 3:
            conv_fn = Conv2D
            kernel_size = (1, 1)
        elif fts_rank > 3:
            conv_fn = self.conv_fn
            kernel_size = self.kernel_size
        else:
            raise ValueError('input shape must be at least rank 3 (nbatch, npoint, nfeatures) '
                             'for convolution operation')

        # Apply convolutional layers
        x = knn_fts
        for idx, channel in enumerate(self.channels):
            x = conv_fn(channel, kernel_size=kernel_size, strides=1,
                        data_format='channels_last', use_bias=not self.batchnorm,
                        kernel_initializer=GlorotNormal(self.seed),
                        name=f'conv_{idx}')(x)
            if self.batchnorm:
                x = BatchNormalization(name=f'bn_{idx}')(x)
            if self.activation:
                x = Activation(self.activation, name=f'act_{idx}')(x)
                
        fts = self.pool_fn(x, axis=-2)

        # Shortcut connection
        sc = conv_fn(self.channels[-1], kernel_size=kernel_size, strides=1,
                     data_format='channels_last', use_bias=not self.batchnorm,
                     kernel_initializer=GlorotNormal(self.seed),
                     name='shortcut')(tf.expand_dims(features, axis=-2))
        if self.batchnorm:
            sc = BatchNormalization(name='shortcut_bn')(sc)
        sc = ops.squeeze(sc, axis=-2)

        # Add shortcut and output
        x = add(sc, fts)
        if self.activation:
            return Activation(self.activation, name='shortcut_act')(x)
        return x

class LikelihoodRatio(Layer):
    
    def __init__(self, fs: Model, kappa: Number | Model = 1.0,
                 epsilon: float = 1e-8, **kwargs):
        super(LikelihoodRatio, self).__init__(**kwargs)
        self.fs = fs
        self.kappa = kappa
        self.epsilon = epsilon

    def call(self, inputs: Union[keras.Input, Tuple[keras.Input, ...]]) -> Layer:
        if isinstance(self.kappa, Model):
            _, param_input = inputs
            kappa_out = self.kappa(param_input)
        else:
            kappa_out = self.kappa
        fs_out = self.fs(inputs)
        return kappa_out * fs_out / (1 - fs_out + self.epsilon)

class SemiWeakly(Layer):

    def __init__(self, *fs_list,
                 log_mu: bool = True,
                 log_alpha: bool = False,
                 init_mu: float = 0.01,
                 min_mu: float = 5e-5,
                 rstr_mu: float = 1.0,
                 init_alpha: float | List[float] | None = None,
                 rstr_alpha: float | List[float] = 1.,
                 **kwargs):
        super(LikelihoodRatio, self).__init__(**kwargs)
        self.fs_list = fs_list
        self.mu = None
        self.alpha = None
        assert_range('init_mu', init_mu, rmin=0, rmax=1)
        assert_range('min_mu', min_mu, rmin=0, rmax=1)
        self.init_mu = init_mu
        self.min_mu = min_mu
        self.rstr_mu = rstr_mu
        nsig = len(fs_list)
        if nsig > 1:
            if init_alpha is None:
                init_alpha = [1.0 / nsig] * (nsig - 1)
            elif (nsig == 2) and isinstance(init_alpha, Number):
                init_alpha = [float(init_alpha)]
            else:
                init_alpha = [float(val) for val in init_alpha]
            init_alpha = np.array(init_alpha)
            assert len(init_alpha) == (nsig - 1)
            for val in init_alpha:
                assert_range('init_alpha', val, rmin=0, rmax=1)
            if (nsig == 2) and isinstance(rstr_alpha, Number):
                rstr_alpha = [float(rstr_alpha)]
            else:
                rstr_alpha = [float(val) for val in rstr_alpha]
            rstr_alpha = np.array(rstr_alpha)
            assert len(rstr_alpha) == (nsig - 1)
        else:
            default_alpha = None
            rstr_alpha = None
        self.default_alpha = default_alpha
        self.rstr_alpha = rstr_alpha
        
    def build(self, input_shape):
        init_mu = np.log(self.init_mu) if self.log_mu else self.init_mu
        min_mu = np.log(self.min_mu) if self.log_mu else self.min_mu
        max_mu = 0. if self.log_mu else 1.0
        if self.init_alpha is not None:
            init_alpha = np.log(self.init_alpha) if self.log_alpha else self.init_alpha
        else:
            init_alpha = None
        self.mu = self.add_weight(
            shape=(),
            initializer=initializers.Constant(init_mu),
            regularizer=MinMaxRegularizer(min_mu, max_mu, self.rstr_mu),
            name='mu')
        
        
        self.weight = self.add_weight(
            shape=(),  # A scalar weight
            initializer=initializers.Constant(self.default_mu),  # Use the specified initializer
            constraint=self.constraint,  # Apply the constraint, if any
            regularizer=self.regularizer,  # Apply the regularizer, if any
            trainable=True,
            name="single_weight"
        )

    def call(self, inputs):
        # Perform element-wise multiplication of the input by the single weight
        return inputs * self.weight