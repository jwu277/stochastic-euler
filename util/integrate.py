import numpy as np
from scipy.integrate import solve_ivp


def ivp(fn, x0, tmax, dt, method):
    sol = solve_ivp(fn, (0, tmax), x0, max_step=dt, method=method, vectorized=True)
    return np.transpose(sol.y)[np.argsort(sol.t)]

