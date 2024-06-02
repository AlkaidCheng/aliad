import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Aliad', name='MinMaxRegularizer')
class MinMaxRegularizer(tf.keras.regularizers.Regularizer):
  def __init__(self, min_val, max_val, l=1.0):
      self.min_val = min_val
      self.max_val = max_val
      self.l = l

  def __call__(self, x):
      x_under = tf.cast(tf.less(x, self.min_val), dtype=tf.float32)
      x_over  = tf.cast(tf.greater(x, self.max_val), dtype=tf.float32)
      under_penalty = (tf.exp(self.min_val - x) - 1)
      over_penalty = (tf.exp(x - self.max_val) - 1)
      penalty = self.l * tf.math.reduce_sum(((x_under * under_penalty) + (x_over * over_penalty)))
      return penalty

  def get_config(self):
      return {'l': float(self.l),
              'min_val': float(self.min_val),
              'max_val': float(self.max_val)}