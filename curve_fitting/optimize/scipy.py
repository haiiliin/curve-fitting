from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, List, Sequence

import numpy as np

from .optimizer import OptimizerBase
from .options import OptimizerOptions
from .registry import register
from .result import OptimizerResult

if TYPE_CHECKING:
    from scipy.optimize import Bounds


class ScipyOptimizerResult(OptimizerResult):
    """Represents the optimization result for scipy optimize algorithms

    Notes
    -----
        ``OptimizeResult`` may have additional attributes not listed here depending
        on the specific solver being used. Since this class is essentially a
        subclass of dict with attribute accessors, one can see which
        attributes are available using the `OptimizeResult.keys` method.
    """

    #: Whether the optimizer exited successfully.
    success: bool = False
    #: Termination status of the optimizer. Its value depends on the underlying solver. Refer to ``message`` for
    #: details.
    status: int = -1
    #: Description of the cause of the termination.
    message: str = ""
    #: Values of Jacobin
    jac: List[float] = []
    #: Values of Hessian, The Hessians may be approximations, see the documentation of the function in question.
    hess: List[List[float]] = []
    #: Number of evaluations of the jacobin
    njev: int = 0
    #: Number of evaluations of the hessian
    nhev: int = 0
    #: The maximum constraint violation.
    maxcv: float = 0.0


class ScipyOptimizerOptions(OptimizerOptions):
    disp: bool
    bounds: np.ndarray | Sequence["Bounds"]


class ScipyOptimizer(OptimizerBase, ABC):
    method: str
    options: ScipyOptimizerOptions

    def setup(self):
        if "maxiter" in self.options and self.method in ["Nelder-Mead", "Powell"]:
            self.options["maxfev"] = self.options.pop("maxfun")

    def optimize(self) -> ScipyOptimizerResult:
        from scipy import optimize

        res = optimize.minimize(
            self.objective_function,
            x0=self.x0,
            args=self.args,
            bounds=self.bounds,
            method=self.method,
            callback=self.callback,
            options=self.options,
        )
        return ScipyOptimizerResult(**res)


class LBFGSBOptimizerOptions(ScipyOptimizerOptions):
    maxcor: int
    ftol: float
    gtol: float
    eps: float
    maxiter: int
    maxfun: int
    iprint: int
    maxls: int
    finite_diff_rel_step: np.ndarray


@register("Scipy-LBFGSB")
class LBFGSBOptimizer(ScipyOptimizer):
    method = "L-BFGS-B"
    options: LBFGSBOptimizerOptions


class NelderMeadOptimizerOptions(ScipyOptimizerOptions):
    maxfev: int
    return_all: bool
    initial_simplex: np.ndarray
    xatol: float
    fatol: float
    adaptive: bool


@register("Scipy-NelderMead")
class NelderMeadOptimizer(ScipyOptimizer):
    method = "Nelder-Mead"
    options: NelderMeadOptimizerOptions


class PowellOptimizerOptions(ScipyOptimizerOptions):
    maxfev: int
    direc: np.ndarray
    tol: float


@register("Scipy-Powell")
class PowellOptimizer(ScipyOptimizer):
    method = "Powell"
    options: PowellOptimizerOptions


class TNCOptimizerOptions(ScipyOptimizerOptions):
    eps: float
    scale: List[float]
    offset: float
    maxCGit: int
    eta: float
    stepmx: float
    accuracy: float
    minfev: float
    ftol: float
    xtol: float
    gtol: float
    rescale: float
    finite_diff_rel_step: np.ndarray


@register("Scipy-TNC")
class TNCOptimizer(ScipyOptimizer):
    method = "TNC"
    options: TNCOptimizerOptions
