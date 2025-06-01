from typing import Optional, Tuple

import numpy as np

def get_poisson_weights(
    size: int | Tuple[int],
    seed: Optional[int] = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.poisson(size=size)