from quickstats.core.modules import require_module, get_module_version

def _get_keras_ops():
    """Determine the correct Keras operations module based on the Keras version."""
    require_module("keras")  # Ensure Keras is installed
    keras_version = get_module_version("keras")

    if keras_version.major == 3:
        from keras import ops as keras_ops
    elif keras_version.major == 2:
        from keras import backend as keras_ops
    else:
        raise RuntimeError(
            f"Unsupported Keras version: {keras_version} "
            "(must be Keras 2 or Keras 3)"
        )
    return keras_version.major, keras_ops

KERAS_VERSION_MAJOR, keras_ops = _get_keras_ops()

__all__ = ["KERAS_VERSION_MAJOR", "keras_ops"]