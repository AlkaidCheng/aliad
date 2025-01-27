from __future__ import annotations
from typing import Union, Any
from functools import cached_property

import numpy as np
from keras import activations
try:
    from keras import ops
except ImportError:
    from keras import backend as ops

# Type alias for better readability
Tensor = Any

class Activation:
    """Base class for activation functions with inverse property.
    
    Provides interface for activation functions that can be inverted,
    with methods to apply the activation and get numpy values.
    """
    @cached_property
    def inverse(self) -> Activation:
        """Returns the inverse activation function."""
        raise NotImplementedError("Inverse function not implemented")
    
    def get_value(self, x: Tensor) -> np.ndarray:
        """Applies activation and converts result to numpy array.
        
        Args:
            x: Input tensor or array
            
        Returns:
            numpy.ndarray: Activated values as numpy array
        """
        return np.array(self.__call__(x))

    def get_inverse(self, x: Tensor) -> np.ndarray:
        return np.array(self.inverse.__call__(x))

    def get_config(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)

    def __call__(self, x: Tensor) -> Tensor:
        """Applies the activation function.
        
        Args:
            x: Input tensor or array
            
        Returns:
            Tensor: Activated values
        """
        raise NotImplementedError("Activation function not implemented")


class Logistic(Activation):
    """Logistic (sigmoid) activation function."""
    
    @cached_property
    def inverse(self) -> Activation:
        return Logit()
    
    def __call__(self, x: Tensor) -> Tensor:
        return activations.sigmoid(x)


# Type alias for better readability
Sigmoid = Logistic


class Logit(Activation):
    """Logit function (inverse of sigmoid)."""
    
    @cached_property
    def inverse(self) -> Activation:
        return Logistic()
    
    def __call__(self, x: Tensor) -> Tensor:
        return ops.log(x / (1 - x + ops.epsilon()))  # Added epsilon for numerical stability


class Exponential(Activation):
    """Exponential activation function."""
    
    @cached_property
    def inverse(self) -> Activation:
        return Log()
    
    def __call__(self, x: Tensor) -> Tensor:
        return ops.exp(x)


class Log(Activation):
    """Natural logarithm activation function."""
    
    @cached_property
    def inverse(self) -> Activation:
        return Exponential()  # Fixed: previously returned Log instead of Exponential
    
    def __call__(self, x: Tensor) -> Tensor:
        return ops.log(x + ops.epsilon())  # Added epsilon for numerical stability


class Scale(Activation):
    """Scaling activation function that multiplies input by a factor."""
    
    def __init__(self, factor: float = 1.0):
        """Initialize scaling factor.
        
        Args:
            factor: Multiplication factor for scaling
        """
        self._factor = float(factor)  # Ensure factor is float
    
    @cached_property
    def inverse(self) -> Activation:
        return Scale(1.0 / self._factor)

    def get_config(self) -> Dict[str, Any]:
        return {
            'factor': self._factor
        }
    
    def __call__(self, x: Tensor) -> Tensor:
        return x * self._factor

class Linear(Activation):
    
    @cached_property
    def inverse(self) -> Activation:
        return self
    
    def __call__(self, x: Tensor) -> Tensor:
        return x