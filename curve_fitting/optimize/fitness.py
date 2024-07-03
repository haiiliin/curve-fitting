from __future__ import annotations

from abc import ABC
from typing import Dict, Iterable, Tuple

from typing_extensions import Self


class FitnessBase(ABC):
    def __init__(self, **kwargs):
        """Initialize the fitness function"""
        self.setup(**kwargs)

    def setup(self, **kwargs) -> Self:
        """Set up the arguments of the fitness function"""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def parseOptimizeKeys(self):
        """Parse the keys of the fitness function"""
        raise NotImplementedError

    def run(self, x: Iterable | dict, *args, **kwargs) -> float:
        """Evaluate the fitness function."""
        raise NotImplementedError

    def simulate(self, x: Iterable | dict) -> Dict:
        """Simulate the model and return the results as a dictionary of SVExporter objects. If the simulation fails,
        return False instead."""
        raise NotImplementedError

    def evaluate(self, results: Dict) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """Evaluate the fitness function, return the fitness and weights"""
        raise NotImplementedError

    def __call__(self, x: Iterable | dict, *args, **kwargs) -> float:
        """Evaluate the fitness function."""
        return self.run(x, *args, **kwargs)
