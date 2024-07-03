from __future__ import annotations

import numpy as np

from .dict import Dict


class OptimizerOptions(Dict):
    """
    Options of the optimizer
    """

    maxiter: int
    maxfun: int
    bounds: np.ndarray
