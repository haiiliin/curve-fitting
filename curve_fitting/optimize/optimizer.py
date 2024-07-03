from __future__ import annotations

from typing import Any, Callable, Dict, Iterable

import numpy as np

from .fitness import FitnessBase
from .options import OptimizerOptions
from .result import OptimizerResult


class OptimizerBase:
    """
    Base class for optimizers
    """

    #: number of objectives
    n_obj: int = 1

    #: Objective function to optimize, the function's signature must be: f(x: Iterable) -> float
    objective_function: FitnessBase | Callable

    #: Initial parameters
    x0: np.ndarray

    #: Parameter bounds
    bounds: np.ndarray | None

    #: Additional arguments for the objective function
    args: tuple

    #: options
    options: OptimizerOptions
    defaultOptions: Dict = {}

    #: callback function that will be evaluated after each iteration
    user_callback: Callable[[Iterable], Any] | None

    def __init__(
        self,
        objective_function: FitnessBase | Callable,
        x0: np.ndarray,
        *,
        args: tuple = (),
        bounds: np.ndarray | None = None,
        callback: Callable[[Iterable], Any] | None = None,
        maxiter: int = 1000,
        maxfun: int = 10000,
        **options,
    ):
        """Initialize the optimizer

        Parameters
        ----------
        objective_function : FitnessBase | Callable
            Objective function
        x0 : np.ndarray
            Initial parameters
        args : tuple
            Additional arguments for the objective function
        bounds : np.ndarray
            Bounds of the parameter
        maxiter : int
            Maximal number of iterations
        maxfun : int
            Maximal number of function evaluations
        callback : Callable
            Callback function that will be evaluated after each iteration
        options : int | float | bool | dict | Any
            Optimizer-specific options
        """
        self.objective_function = objective_function
        self.x0 = x0
        self.bounds = bounds
        self.user_callback = callback
        self.args = args
        self.options = OptimizerOptions(self.defaultOptions)
        self.options.update(options)
        self.options.update({"maxiter": maxiter, "maxfun": maxfun})
        self.setup()

    def setup(self):
        """Post set up the optimizer"""
        pass

    def callback(self, x: np.ndarray | Any):
        """Callback function that will be evaluated after each iteration"""
        self.user_callback(x) if self.user_callback else None

    def optimize(self) -> OptimizerResult:
        """Optimize the objective function"""
        raise NotImplementedError

    @classmethod
    def fmin(cls, *args, **kwargs) -> OptimizerResult:
        """Static method to do the optimization

        Parameters
        ----------
        args, kwargs
            Positional and keyword arguments for the optimizer

        Returns
        -------
        res : OptimizeResult
            Optimize result
        """
        return cls(*args, **kwargs).optimize()
