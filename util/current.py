## Util for generating input currents


import numpy as np
from scipy.stats import uniform


def constant(tmax, dt, ampl=1):
    return ampl * np.ones(int(tmax / dt) + 1)


def poisson(tmax, t_avg, dt, ampl=1):
    rvs = uniform.rvs(size=int(tmax / dt) + 1)
    return (ampl * t_avg / dt) * (rvs < dt / t_avg)

