import numpy as np

from curve_fitting import curve_fitting


def test_curve_fitting():
    x, y, z = np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    f = x**3 + y**2 + z

    xs = dict(x=x, y=y, z=z)
    equation = "a * x ^ 3 + b * y ^2 + c * z"
    initials = dict(a=1.5, b=0.5, c=2.0)
    bounds = dict(a=(0.0, 2.0), b=(0.0, 2.0), c=(0.0, 2.0))
    stds = dict(a=0.1, b=0.1, c=0.1)

    result = curve_fitting(xs, f, equation, initials, bounds, stds, maxiter=1000, maxfun=10000, n_jobs=1)
    assert result.fun < 1e-5, "Failed to fit the curve"
    assert np.allclose(result.x, [1, 1, 1], atol=1e-1), "Failed to fit the curve"
