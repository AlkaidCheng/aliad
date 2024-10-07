import tensorflow as tf
import keras
from keras import backend
from keras.src.losses import LossFunctionWrapper

def scaled_binary_crossentropy(
    y_true, y_pred, offset=0., scale=1., from_logits=False, label_smoothing=0.0, axis=-1
):
    """Computes the scaled binary crossentropy loss.

    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        offset: Offset applied to the loss value.
        scale: Scale factor applied to the loss value (after offset).
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
            squeezing them towards 0.5, that is,
            using `1. - 0.5 * label_smoothing` for the target class
            and `0.5 * label_smoothing` for the non-target class.
        axis: The axis along which the mean is computed. Defaults to `-1`.

    Returns:
        Scaled binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)

    def _smooth_labels():
        return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    y_true = tf.__internal__.smart_cond.smart_cond(
        label_smoothing, _smooth_labels, lambda: y_true
    )

    return (backend.mean(
        backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits),
        axis=axis,
    ) + offset) * scale

def scaled_negative_loglikelihood(
    y_true, y_pred, offset=0., scale=1., from_logits=False, label_smoothing=0.0, axis=-1
):
    """Computes the scaled negative log-likelihood loss.

    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        offset: Offset applied to the loss value.
        scale: Scale factor applied to the loss value (after offset).
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
            squeezing them towards 0.5, that is,
            using `1. - 0.5 * label_smoothing` for the target class
            and `0.5 * label_smoothing` for the non-target class.
        axis: The axis along which the mean is computed. Defaults to `-1`.

    Returns:
        Scaled negative log-likelihood loss value. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)

    def _smooth_labels():
        return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    y_true = tf.__internal__.smart_cond.smart_cond(
        label_smoothing, _smooth_labels, lambda: y_true
    )

    if from_logits:
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target, logits=output
        )

    epsilon_ = tf.convert_to_tensor(backend.epsilon(), y_pred.dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    nll = - y_true * (tf.math.log(y_pred))
    
    return (backend.mean(
        nll,
        axis=axis,
    ) + offset) * scale

@keras.saving.register_keras_serializable(package="ScaledBinaryCrossentropy")
class ScaledBinaryCrossentropy(LossFunctionWrapper):
    """
    Custom loss function that applies an offset and scaling to the binary cross-entropy loss.

    Args:
        offset (float): The offset to be added to the binary cross-entropy loss. Default is 0.0.
        scale (float): The scaling factor to be applied to the (binary cross-entropy loss + offset). Default is 1.0.
        from_logits (bool): Whether `y_pred` is expected to be a logits tensor. By default, we assume that `y_pred` 
                            encodes a probability distribution. Default is False.
        label_smoothing (float): Float in [0, 1]. When 0, no smoothing occurs. When > 0, we compute the loss between
                                 the predicted labels and a smoothed version of the true labels, where the smoothing 
                                 squeezes the labels towards 0.5. Default is 0.
        axis (int): The axis along which to compute cross-entropy. Default is -1.
        **kwargs: Additional keyword arguments for the base class.
    """

    def __init__(
        self,
        offset=0.0,
        scale=1.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="scaled_binary_crossentropy",
    ):
        super().__init__(
            scaled_binary_crossentropy,
            name=name,
            offset=offset,
            scale=scale,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.offset = offset
        self.scale = scale        
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    @classmethod
    def from_config(cls, config):
        config.pop("fn", None)
        config.setdefault("offset", 0)
        config.setdefault("scale", 1)
        return cls(**config)    

    def get_config(self):
        return {
            "name": self.name,
            "offset": self.offset,
            "scale": self.scale,
            "reduction": self.reduction,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
            "axis": self.axis,
        }

@keras.saving.register_keras_serializable(package="ScaledMaximumLikelihood")
class ScaledNLLLoss(LossFunctionWrapper):
    """
    Custom loss function that applies an offset and scaling to the negative log-likelihood loss.

    Args:
        offset (float): The offset to be added to the binary cross-entropy loss. Default is 0.0.
        scale (float): The scaling factor to be applied to the (binary cross-entropy loss + offset). Default is 1.0.
        from_logits (bool): Whether `y_pred` is expected to be a logits tensor. By default, we assume that `y_pred` 
                            encodes a probability distribution. Default is False.
        label_smoothing (float): Float in [0, 1]. When 0, no smoothing occurs. When > 0, we compute the loss between
                                 the predicted labels and a smoothed version of the true labels, where the smoothing 
                                 squeezes the labels towards 0.5. Default is 0.
        axis (int): The axis along which to compute maximum likelihood. Default is -1.
        **kwargs: Additional keyword arguments for the base class.
    """

    def __init__(
        self,
        offset=0.0,
        scale=1.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="scaled_maximum_likelihood",
    ):
        super().__init__(
            scaled_maximum_likelihood,
            name=name,
            offset=offset,
            scale=scale,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.offset = offset
        self.scale = scale        
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    @classmethod
    def from_config(cls, config):
        config.pop("fn", None)
        config.setdefault("offset", 0)
        config.setdefault("scale", 1)
        return cls(**config)    

    def get_config(self):
        return {
            "name": self.name,
            "offset": self.offset,
            "scale": self.scale,
            "reduction": self.reduction,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
            "axis": self.axis,
        }