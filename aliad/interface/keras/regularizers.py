from keras.regularizers import Regularizer
try:
    from keras import ops
except ImportError:
    from keras import backend as ops

class MinMaxRegularizer(Regularizer):
    """
    Custom regularizer that penalizes weights falling outside a specified min-max range.
    
    The regularizer applies penalties when the values of `x` fall below `min_val` or above `max_val`.
    Penalties can be chosen from a variety of functions, specified by the `penalty_type` argument.

    Parameters:
    -----------
    min_val : float
        The minimum allowed value for the weights.
    max_val : float
        The maximum allowed value for the weights.
    strength : float, optional (default=1.0)
        The scaling factor for the penalty.
    penalty_type : str, optional (default='exponential')
        The type of penalty function to apply. Options are:
        - 'exponential': Penalty increases exponentially as the value deviates from the min/max.
        - 'quadratic': Penalty increases quadratically as the value deviates from the min/max.
        - 'absolute': Penalty increases linearly as the value deviates from the min/max.
    """
    
    def __init__(self, min_val: float, max_val: float, strength: float = 1.0, penalty_type: str = 'exponential') -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.strength = strength
        
        if penalty_type == 'exponential':
            self.penalty_fn = self._exponential_penalty
        elif penalty_type == 'quadratic':
            self.penalty_fn = self._quadratic_penalty
        elif penalty_type == 'absolute':
            self.penalty_fn = self._absolute_penalty
        else:
            raise ValueError(f"unsupported penalty type: {penalty_type}")

    def _exponential_penalty(self, x_under, x_over, x):
        """Compute exponential penalties for values outside min/max."""
        under_penalty = ops.exp(self.min_val - x) - 1
        over_penalty = ops.exp(x - self.max_val) - 1
        return (x_under * under_penalty) + (x_over * over_penalty)

    def _quadratic_penalty(self, x_under, x_over, x):
        """Compute quadratic penalties for values outside min/max."""
        under_penalty = (self.min_val - x) ** 2
        over_penalty = (x - self.max_val) ** 2
        return (x_under * under_penalty) + (x_over * over_penalty)

    def _absolute_penalty(self, x_under, x_over, x):
        """Compute absolute penalties for values outside min/max."""
        under_penalty = ops.abs(self.min_val - x)
        over_penalty = ops.abs(x - self.max_val)
        return (x_under * under_penalty) + (x_over * over_penalty)

    def __call__(self, x) -> float:
        """
        Apply the regularization penalty to the input tensor `x`.

        Parameters:
        -----------
        x : tensor-like
            The weights or inputs to which the regularizer is applied.

        Returns:
        --------
        penalty : float
            The computed regularization penalty.
        """
        x_under = ops.cast(ops.less(x, self.min_val), dtype="float32")
        x_over = ops.cast(ops.greater(x, self.max_val), dtype="float32")

        # Use the precompiled penalty function to avoid checking penalty_type during each call
        penalty = self.strength * ops.sum(self.penalty_fn(x_under, x_over, x))
        return penalty

    def get_config(self) -> dict:
        """
        Return the configuration of the regularizer for serialization.

        Returns:
        --------
        config : dict
            A dictionary containing the regularizer's configuration.
        """
        return {
            'min_val': float(self.min_val),
            'max_val': float(self.max_val),
            'strength': float(self.strength),
            'penalty_type': self.penalty_fn.__name__.replace('_penalty', '')
        }