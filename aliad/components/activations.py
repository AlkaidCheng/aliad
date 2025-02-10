from __future__ import annotations
import math
from typing import Union, Any, Dict
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike

from quickstats import cached_import
from aliad.core.mixins import BackendMixin, ConfigMixin

EPSILON = 1E-7

__all__ = [
    "Activation", "InvertibleActivation", "Logistic", "Sigmoid", "Logit",
    "Exponential", "Log", "Scale", "Linear"
]

class Activation(BackendMixin, ConfigMixin):
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
            from aliad.interface.keras import keras_ops
            self.keras_ops = keras_ops
        else:
            self.__dict__.pop("keras_ops", None)

    def get_value(self, x: Any, *args, **kwargs) -> Any:
        """Applies the activation function."""
        return self._backend_dispatch("get_value", x, *args, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        return {"backend": self.backend}

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)

    def __call__(self, x: Any) -> Any:
        """Applies the activation function."""
        return self.get_value(x)


class InvertibleActivation(Activation):
    """Activation function with an invertible transformation."""

    @cached_property
    def inverse(self) -> Activation:
        raise NotImplementedError("Inverse function not implemented")

    def get_inverse(self, x: Any) -> Any:
        """Apply the inverse activation function."""
        return self.inverse.get_value(x)


class Logistic(InvertibleActivation):
    """Logistic (sigmoid) activation function."""

    @cached_property
    def inverse(self) -> Activation:
        return Logit(backend=self.backend)

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        if np.ndim(x) == 0:
            x = float(x)
            if x >= 0:
                return 1 / (1 + math.exp(-x))
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)

        x = np.asarray(x, dtype=np.float64)
        pos_mask = x >= 0
        exp_x = np.exp(x)

        return np.where(pos_mask, 1 / (1 + np.exp(-x)), exp_x / (1 + exp_x))

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        keras = cached_import("keras")
        return keras.activations.sigmoid(x)

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        x = torch.as_tensor(x)
        return torch.sigmoid(x)


# Alias for Logistic
Sigmoid = Logistic


class Logit(InvertibleActivation):
    """Logit function (inverse of sigmoid)."""

    @cached_property
    def inverse(self) -> Activation:
        return Logistic(backend=self.backend)

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        if np.ndim(x) == 0:
            x = float(x)
            return math.log(x / (1 - x + EPSILON))

        x = np.asarray(x, dtype=np.float64)
        return np.log(x / (1 - x + EPSILON))

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        return self.keras_ops.log(x / (1 - x + EPSILON))

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        x = torch.as_tensor(x)
        return torch.log(x / (1 - x + EPSILON))


class Exponential(InvertibleActivation):
    """Exponential activation function."""

    @cached_property
    def inverse(self) -> Activation:
        return Log(backend=self.backend)

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        if np.ndim(x) == 0:
            return math.exp(float(x))
        return np.exp(np.asarray(x, dtype=np.float64))

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        return self.keras_ops.exp(x)

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        x = torch.as_tensor(x)
        return torch.exp(x)


class Log(InvertibleActivation):
    """Natural logarithm activation function."""

    @cached_property
    def inverse(self) -> Activation:
        return Exponential(backend=self.backend)

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        if np.ndim(x) == 0:
            return math.log(float(x) + EPSILON)
        return np.log(np.asarray(x, dtype=np.float64) + EPSILON)

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        return self.keras_ops.log(x + EPSILON)

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        x = torch.as_tensor(x)
        return torch.log(x + EPSILON)


class Scale(InvertibleActivation):
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

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        if np.ndim(x) == 0:
            return float(x) * self._factor
        return np.asarray(x, dtype=np.float64) * self._factor

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        tf = cached_import('tensorflow')
        return tf.convert_to_tensor(x) * self._factor

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        return torch.as_tensor(x) * self._factor


class Linear(InvertibleActivation):
    """Linear activation function (identity function)."""

    @cached_property
    def inverse(self) -> Activation:
        return self

    def _get_value_python(self, x: ArrayLike) -> Union[float, np.ndarray]:
        if np.ndim(x) == 0:
            return float(x)
        return np.asarray(x, dtype=np.float64)

    def _get_value_tensorflow(self, x: "Tensor") -> "Tensor":
        tf = cached_import('tensorflow')
        return tf.convert_to_tensor(x)

    def _get_value_pytorch(self, x: "Tensor") -> "Tensor":
        torch = cached_import('torch')
        return torch.as_tensor(x)
