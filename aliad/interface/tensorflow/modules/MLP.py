import tensorflow as tf
from tensorflow.keras.layers import Dense

from aliad.interface.tensorflow.utils import parse_layer_specs

class MLP(tf.Module):
    def __init__(self, layer_specs, name=None):
        super().__init__(name=name)
        with self.name_scope:
            self.layers = parse_layer_specs(layer_specs, layer_cls=Dense)

    @tf.Module.with_name_scope
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x