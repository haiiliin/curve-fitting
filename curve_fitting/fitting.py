import inspect
import time
from typing import Dict, Iterable, Tuple

import numpy as np
import sympy as sp

from .optimize.registry import registry


def fitness(x: np.ndarray, keys: Iterable[str], xs: Dict[str, np.ndarray], f: np.ndarray, equation: str):
    expr = sp.sympify(equation)
    func = sp.lambdify(list(expr.free_symbols), expr, "numpy")

    data = {**xs, **dict(zip(keys, x))}
    return np.linalg.norm(func(**data) - f) / np.sqrt(f.size)


def curve_fitting(
    xs: Dict[str, Iterable[float]],
    f: Iterable[float],
    equation: str,
    initials: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    stds: Dict[str, float],
    method: str = "PYCMA-CMAES",
    **kwargs,
):
    """Fit a curve to the data

    Parameters
    ----------
    xs : Dict[str, Iterable[float]]
        A dictionary containing the x values for each variable
    y : Iterable[float]
        The y values
    equation : str
        The equation to fit
    initials : Dict[str, float]
        The initial values for each variable
    bounds : Dict[str, Tuple[float, float]]
        The bounds for each variable
    stds : Dict[str, float]
        The standard deviations for each variable
    method : str, optional
        The optimization method, by default "PYCMA-CMAES"
    kwargs
        Keyword arguments for the optimizer
    """
    expr = sp.sympify(equation)
    keys = [str(s) for s in expr.free_symbols if str(s) not in xs]
    initials_array = np.array([initials[var] for var in keys])
    bounds_array = np.array([bounds[var] for var in keys])
    stds_array = np.array([stds[var] for var in keys])
    xs_array = {k: np.asarray(v) for k, v in xs.items()}
    f_array = np.asarray(f)

    kwargs.update(
        objective_function=fitness,
        x0=initials_array,
        bounds=bounds_array,
        stds=stds_array,
        args=(keys, xs_array, f_array, equation),
    )
    optimizer = registry[method](**kwargs)

    # Optimize
    calibration_message = f"""\
    Fitting equation: {equation}

    Parameters: {keys}
    Initial fitness: {fitness(initials_array, keys, xs_array, f_array, equation)}
    Initials: {initials}
    Bounds: {bounds}
    Standard deviations: {stds}
    Method: {method}
    """
    print(inspect.cleandoc(calibration_message))
    start_time = time.time()
    result = optimizer.optimize()
    print(f"Calibration finished, took {time.time() - start_time:.2f} seconds")

    # Process results
    result.duration = time.time() - start_time
    result.optimizeKeys = [str(key) for key in keys]
    result.parameters = {key: result.x[idx] for idx, key in enumerate(keys)}

    print(f"Optimized parameters: {result.parameters}")
    return result
