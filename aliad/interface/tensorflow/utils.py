from typing import Optional, Union, Type, Dict, Tuple, List
import tensorflow as tf
from tensorflow.keras.layers import Layer

def create_layer_from_spec(
    layer_spec: Union[Tuple, Dict],
    default_layer: Optional[Union[str, Type[Layer]]] = None
) -> Layer:
    """
    Create a TensorFlow layer based on the given specifications and class.

    Parameters:
    - layer_spec (Union[Tuple, Dict]): A tuple or dictionary specifying the parameters for the layer.
    - default_layer (Optional[Union[str, Type[Layer]]]): The class or string name of the layer to create by default. If None,
      'layer' key in layer_spec is used to determine the layer class.

    Returns:
    - Layer: An instance of the specified TensorFlow layer.
    """

    if default_layer is None:
        if isinstance(layer_spec, dict) and 'layer' in layer_spec:
            layer_cls = layer_spec.pop('layer')
        else:
            raise ValueError(
                "If layer_cls is None, layer_spec must be a dictionary with a 'layer' key."
            )
    else:
        layer_cls = default_layer

    if isinstance(layer_cls, str):
        layer_cls = getattr(tf.keras.layers, layer_cls)
        
    assert isinstance(layer_cls, Layer)

    if isinstance(layer_spec, tuple):
        return layer_cls(*layer_spec)
    elif isinstance(layer_spec, dict):
        return layer_cls(**layer_spec)
    else:
        raise ValueError("layer_spec must be either a tuple or a dictionary")

def parse_layer_specs(layer_specs: List,
                      default_layer: Optional[Union[str, Type[Layer]]] = None):
    layers = []
    for spec in layer_specs:
        layer = create_layer_from_spec(spec, default_layer=default_layer)
        layers.append(layer)
    return layers