from __future__ import annotations
import math
from typing import Union, Any, Dict
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike

from quickstats import cached_import
from quickstats.core.registries import get_registry, create_registry_metaclass

from aliad.core.mixins import BackendMixin, ConfigMixin

Registry = get_registry('aliad.activations')
RegistryMeta = create_registry_metaclass(Registry)

EPSILON = 1E-7

__all__ = [
    "Activation", "InvertibleActivation", "Logistic", "Sigmoid", "Logit",
    "Exponential", "Log", "Scale", "Linear"
]

class Activation(BackendMixin, ConfigMixin, metaclass=RegistryMeta):
    """Base class for activation functions."""

    BACKENDS = {"python", "tensorflow", "pytorch"}

    BACKEND_REQUIRES = {
        "python": {"modules": ["numpy"]},
        "tensorflow": {
            "modules": ["tensorflow", "numpy"],
            "versions": {"tensorflow": {"minimum": "2.15.0"}}
        },
        "pytorch": {
            "modules": ["torch", "numpy"],
            "versions": {"torch": {"minimum": "1.8.0"}}
        }
    }

    def __init__(self, backend: str = "python"):
        super().__init__(backend=backend)
        self._set_backend_ops()

    def set_backend(self, backend: str):
        """Set a new backend and update backend-specific operations."""
        super().set_backend(backend)
        self._set_backend_ops()

    def _set_backend_ops(self):
        """Load backend-specific operations (e.g., keras_ops for TensorFlow)."""
        if self.backend == "tensorflow":
            from aliad.interface.keras.ops import ops as keras_ops
            self.keras_ops = keras_ops
        else:
            self.__dict__.pop("keras_ops", None)

    def get_value(self, x: Any, *args, **kwargs) -> Any:
        """Applies the activation function."""
        return self._backend_dispatch("get_value", x, *args, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        return {"backend": self.backend}

    def _cast_python(self, x: Any, *args, **kwargs) -> Union[float, np.ndarray]:
        if np.ndim(x) == 0:
            return float(x)
        return np.asarray(x, dtype=np.float64)

    def _cast_tensorflow(self, x: Any, *args, **kwargs) -> "Tensor":
        tf = cached_import('tensorflow')
        return tf.convert_to_tensor(x, dtype=tf.float32)

    def _cast_pytorch(self, x: Any, *args, **kwargs) -> "Tensor":
        torch = cached_import('torch')
        return torch.as_tensor(x)

    def cast(self, x: Any, *args, **kwargs) -> Any:
        return self._backend_dispatch("cast", x, *args, **kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        """Applies the activation function."""
        return self.get_value(x, *args, **kwargs)

Registry.remove('Activation')

class InvertibleActivation(Activation):
    """Activation function with an invertible transformation."""

    @cached_property
    def inverse(self) -> Activation:
        raise NotImplementedError("Inverse function not implemented")

    def get_inverse(self, x: Any, *args, **kwargs) -> Any:
        """Apply the inverse activation function."""
        return self.inverse.get_value(x, *args, **kwargs)

class DifferentiableActivation(Activation):
    """Activation function with a differentiable transformation."""

    def get_derivative(self, x: Any, *args, **kwargs) -> Any:
        return self._backend_dispatch("get_derivative", x, *args, **kwargs)
        
Registry.remove('InvertibleActivation')

class Logistic(InvertibleActivation, DifferentiableActivation):
    """Logistic (sigmoid) activation function."""

    __registry_aliases__ = ['Sigmoid']

    @cached_property
    def inverse(self) -> Activation:
        return Logit(backend=self.backend)

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        x = self.cast(x)
        if np.ndim(x) == 0:
            if x >= 0:
                return 1 / (1 + math.exp(-x))
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)

        pos_mask = x >= 0
        exp_x = np.exp(x)

        return np.where(pos_mask, 1 / (1 + np.exp(-x)), exp_x / (1 + exp_x))

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        keras = cached_import("keras")
        return keras.activations.sigmoid(x)

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        x = self.cast(x)
        return torch.sigmoid(x)

    def get_derivative(self, x: Any) -> Any:
        sigma = self.get_value(x)
        return sigma * (1 - sigma)

# Alias for Logistic
Sigmoid = Logistic

class Logit(InvertibleActivation, DifferentiableActivation):
    """Logit function (inverse of sigmoid)."""

    @cached_property
    def inverse(self) -> Activation:
        return Logistic(backend=self.backend)

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        x = self.cast(x)
        x = np.clip(x, EPSILON, 1 - EPSILON)
        
        if np.ndim(x) == 0:
            return math.log(x / (1 - x))

        return np.log(x / (1 - x))

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        tf = cached_import("tensorflow")
        x = self.cast(x)
        x = tf.clip_by_value(x, EPSILON, 1 - EPSILON)
        return self.keras_ops.log(x / (1 - x))

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        x = self.cast(x)
        x = torch.clamp(x, min=EPSILON, max=1 - EPSILON)
        return torch.log(x / (1 - x))

    def get_derivative(self, x: Any) -> Callable:
        x = self.cast(x)
        return 1 / (x * (1 - x))

class Exponential(InvertibleActivation, DifferentiableActivation):
    """Exponential activation function."""

    @cached_property
    def inverse(self) -> Activation:
        return Log(backend=self.backend)

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        x = self.cast(x)
        if np.ndim(x) == 0:
            return math.exp(x)
        return np.exp(x)

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        return self.keras_ops.exp(x)

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        x = self.cast(x)
        return torch.exp(x)

    def get_derivative(self, x: Any) -> Any:
        return self.get_value(x)

class Log(InvertibleActivation, DifferentiableActivation):
    """Natural logarithm activation function."""

    @cached_property
    def inverse(self) -> Activation:
        return Exponential(backend=self.backend)

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        x = self.cast(x)
        x = np.clip(x, EPSILON, None)
        if np.ndim(x) == 0:
            return math.log(x + EPSILON)
        return np.log(x + EPSILON)

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        tf = cached_import("tensorflow")
        x = self.cast(x)
        x = tf.clip_by_value(x, EPSILON, float("inf"))
        return self.keras_ops.log(x)

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        x = self.cast(x)
        x = torch.clamp(x, min=EPSILON)
        return torch.log(x)

    def get_derivative(self, x: Any) -> Any:
        x = self.cast(x)
        return 1 / x

class Scale(InvertibleActivation, DifferentiableActivation):
    """Scaling activation function that multiplies input by a factor."""

    def __init__(self, factor: float = 1.0, *args, **kwargs):
        """Initialize scaling factor."""
        super().__init__(*args, **kwargs)
        self._factor = float(factor)

    @cached_property
    def inverse(self) -> Activation:
        return Scale(1.0 / self._factor, backend=self.backend)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["factor"] = self._factor
        return config

    def _get_value(self, x: ArrayLike) -> Any:
        x = self.cast(x)
        return x * self._factor

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        return self._get_value(x)

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        return self._get_value(x)

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        return self._get_value(x)

    def _get_derivative_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        return np.ones_like(x) * self._factor

    def _get_derivative_tensorflow(self, x: "Tensor") -> "Tensor":
        tf = cached_import('tensorflow')
        return tf.ones_like(x) * self._factor

    def _get_derivative_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        return torch.ones_like(x) * self._factor


class Linear(InvertibleActivation, DifferentiableActivation):
    """Linear activation function (identity function)."""

    @cached_property
    def inverse(self) -> Activation:
        return self

    def _get_value(self, x: ArrayLike) -> Any:
        return self.cast(x)

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        return self._get_value(x)

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        return self._get_value(x)

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        return self._get_value(x)

    def _get_derivative_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        return np.ones_like(x)

    def _get_derivative_tensorflow(self, x: "Tensor") -> "Tensor":
        tf = cached_import('tensorflow')
        return tf.ones_like(x)

    def _get_derivative_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        return torch.ones_like(x)