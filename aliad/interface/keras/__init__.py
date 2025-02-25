from quickstats.core.modules import require_module
require_module("keras")

from .utils import (
    load_model,
    load_custom_objects
)