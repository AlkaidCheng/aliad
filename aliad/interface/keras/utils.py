from typing import Union, Optional, Dict, List, TypeVar

T = TypeVar('T')

from quickstats import cached_import

CUSTOM_OBJECTS_LOADED = False

def is_keras_model(obj: T) -> bool:
    keras = cached_import('keras')
    return isinstance(obj, keras.Model)

def load_custom_objects(override: bool = False) -> None:
    global CUSTOM_OBJECTS_LOADED
    if CUSTOM_OBJECTS_LOADED and (not override):
        return
    from aliad.components import activations
    from aliad.interface.keras import regularizers
    from aliad.interface.tensorflow import losses
    from keras.utils import get_custom_objects
    from quickstats import get_registry
    custom_objects = get_custom_objects()
    activation_registry = get_registry('aliad.activations')
    if activation_registry:
        for key, obj in activation_registry.all_entries(True).items():
            custom_objects[key] = obj
    regularizer_registry = get_registry('keras.regularizers')
    if regularizer_registry:
        for key, obj in regularizer_registry.all_entries(True).items():
            custom_objects[key] = obj
    loss_registry = get_registry('keras.losses')
    if loss_registry:
        for key, obj in loss_registry.all_entries(True).items():
            custom_objects[key] = obj
    CUSTOM_OBJECTS_LOADED = True

def load_model(path: str, custom_objects: Optional[Dict] = None) -> 'keras.Model':
    load_custom_objects()
    keras = cached_import('keras')
    custom_objects = custom_objects or {}
    return keras.models.load_model(path, custom_objects=custom_objects)