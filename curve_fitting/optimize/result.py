from __future__ import annotations

import json
from typing import Dict, List

from micromechanical.pydantic import HashableBaseModelIO


class OptimizerResult(HashableBaseModelIO, extra="allow"):
    """
    OptimizeResult for the optimizers
    """

    #: The solution of the optimization.
    x: List[float] = []
    #: standard deviations of the parameters
    stds: List[float] = []
    #: Values of objective function
    fun: float = 0.0
    #: Number of evaluations of the objective functions
    nfev: int = 0
    #: Number of iterations performed by the optimizer.
    nit: int = 0
    #: A dict of parameters
    parameters: Dict[str, float] = {}
    #: Optimize keys
    optimizeKeys: List[str] = []
    #: Duration
    duration: float = 0.0

    #: fitness for all experiments and lines
    fitness: Dict[str, Dict[str, float]] = {}
    #: weights for all experiments and lines
    weights: Dict[str, Dict[str, float]] = {}

    def save(self, filepath: str):
        """Save to a Json file"""
        with open(filepath, "w+", encoding="utf-8") as f:
            json.dump(self, f, default=lambda x_: {}, indent=4)
