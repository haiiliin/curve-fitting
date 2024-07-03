from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
    Type,
    Union,
)

import numpy as np

from .optimizer import OptimizerBase
from .options import OptimizerOptions
from .registry import register
from .result import OptimizerResult

if TYPE_CHECKING:
    from cma import CMAEvolutionStrategy


class PYCMAOptimizerResult(OptimizerResult):
    """
    Optimize result for PYCMA package, just for annotation
    """

    #: overall standard deviation
    sigma: float = 0.0
    #: number of function evaluations
    evals_best: int = 0
    #: stop message
    stop: Dict[str, Any] = {}


class PYCMAOptimizerOptions(OptimizerOptions):
    min_iterations: int
    n_jobs: int
    AdaptSigma: bool
    CMA_active: bool
    CMA_active_injected: float
    CMA_cmean: float
    CMA_const_trace: bool
    CMA_diagonal: float
    CMA_eigenmethod: Callable
    CMA_elitist: bool
    CMA_injections_threshold_keep_len: float
    CMA_mirrors: bool
    CMA_mirrormethod: int
    CMA_mu: int
    CMA_on: float
    CMA_sampler: Type
    CMA_sampler_options: dict
    CMA_rankmu: float
    CMA_rankone: float
    CMA_recombination_weights: Tuple = (None,)
    CMA_dampsvec_fac: float
    CMA_dampsvec_fade: float
    CMA_teststds: np.ndarray
    CMA_stds: np.ndarray
    CMA_dampfac: float
    CSA_damp_mueff_exponent: float
    CSA_disregard_length: bool
    CSA_clip_length_value: Tuple[float, float]
    CSA_squared: bool
    BoundaryHandler: str
    bounds: Tuple[float | Iterable[float], float | Iterable[float]]
    conditioncov_alleviate: Tuple[float, float]
    eval_final_mean: bool
    fixed_variables: Dict[int, float]
    ftarget: float
    integer_variables: List[int]
    is_feasible: Callable
    maxfevals: int
    maxiter: int
    mean_shift_line_samples: bool
    mindx: float
    minstd: float | Iterable[float]
    maxstd: float | Iterable[float]
    pc_line_samples: bool
    popsize: int
    popsize_factor: float
    randn: Callable
    scaling_of_variables: Iterable[float]
    seed: int
    signals_filename: str
    termination_callback: Union[Callable, Iterable[Callable]]
    timeout: float
    tolconditioncov: float
    tolfacupx: float
    tolupsigma: float
    tolflatfitness: int
    tolfun: float
    tolfunhist: float
    tolfunrel: float
    tolstagnation: int
    tolx: float
    transformation: tuple
    typical_x: Iterable[float]
    updatecovwait: int
    verbose: int
    verb_append: int
    verb_disp: int
    verb_filenameprefix: str
    verb_log: int
    verb_log_expensive: int
    verb_plot: int
    verb_time: bool
    vv: dict


@register("PYCMA-CMAES")
class PYCMAOptimizer(OptimizerBase):
    defaultOptions: Dict = dict(
        min_iterations=10,
        n_jobs=1,
        verb_disp=100,
    )
    options: PYCMAOptimizerOptions

    #: Standard deviations
    stds: np.ndarray | None = None

    def setup(self):
        self.stds = self.options.pop("stds", None)

    def callback(self, x: "CMAEvolutionStrategy"):
        # TODO: implement
        pass

    def optimize(self) -> PYCMAOptimizerResult:
        import cma

        # Filter CMA options
        CMAOptions = {key: value for key, value in self.options.items() if key in cma.CMAOptions().keys()}
        CMAOptions["CMA_stds"] = self.stds
        CMAOptions["bounds"] = None if self.bounds is None else self.bounds.T.tolist()

        # Optimize
        es = cma.CMAEvolutionStrategy(x0=self.x0, sigma0=1.0, inopts=CMAOptions)
        es.optimize(
            objective_fct=self.objective_function,
            maxfun=self.options.maxfun,
            iterations=self.options.maxiter,
            min_iterations=self.options.min_iterations,
            args=self.args,
            verb_disp=self.options.verb_disp,
            callback=self.callback,
            n_jobs=self.options.n_jobs,
        )

        return PYCMAOptimizerResult(
            x=es.result.xbest,
            fun=es.result.fbest,
            stds=es.result.stds,
            nfev=es.result.evaluations,
            nit=es.result.iterations,
            evals_best=int(es.result.evals_best),
            stop=es.result.stop,
        )
