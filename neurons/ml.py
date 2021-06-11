### Binary Markov Neuron Model ###


from util import ito
import numpy as np
from scipy.stats import binom


class MorrisLecar:

    ## TODO: make neuron stochastic

    ## Neuron governed by following equations:
    ## m = 0.5 * (1 + tanh((v - V1) / V2))
    ## alpha = 0.5 * phi * cosh((v - V3) / (2 * V4)) * (1 + tanh((v - V3) / V4))
    ## beta = 0.5 * phi * cosh((v - V3) / (2 * V4)) * (1 - tanh((v - V3) / V4))
    ## dv/dt = (I - gCa * m * (v - VCa) - gK * w * (v - VK) - gL * (v - VL)) / C
    ## dw/dt = alpha * (1 - w) - beta * w
    ## I is e.g. synaptic input
    def __init__(self, I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic=None, Nk=None):
        self._I = I
        self._phi = phi
        self._C = C
        self._gL = gL
        self._gCa = gCa
        self._gK = gK
        self._VL = VL
        self._VCa = VCa
        self._VK = VK
        self._V1 = V1
        self._V2 = V2
        self._V3 = V3
        self._V4 = V4
        self._dt = dt
        self._stochastic = stochastic
        self._Nk = Nk
    

    # Deterministic Euler part
    # x is a 2-array
    def _a(self, t, x):

        v = x[0]
        w = x[1]

        m = 0.5 * (1 + np.tanh((v - self._V1) / self._V2))
        alpha = 0.5 * self._phi * np.cosh((v - self._V3) / (2 * self._V4)) * (1 + np.tanh((v - self._V3) / self._V4))
        beta = 0.5 * self._phi * np.cosh((v - self._V3) / (2 * self._V4)) * (1 - np.tanh((v - self._V3) / self._V4))

        dv = (self._I[int(t / self._dt)] - self._gCa * m * (v - self._VCa) - self._gK * w * (v - self._VK) - self._gL * (v - self._VL)) / self._C

        if self._stochastic == 'gillespie':
            dw = (binom.rvs(int(self._Nk * (1 - w)), alpha * self._dt) - binom.rvs(int(self._Nk * w), beta * self._dt)) / (self._Nk * self._dt)
        else:
            dw = alpha * (1 - w) - beta * w

        return np.array([dv, dw])


    def _b_euler(self, t, x):

        v = x[0]
        w = x[1]

        alpha = 0.5 * self._phi * np.cosh((v - self._V3) / (2 * self._V4)) * (1 + np.tanh((v - self._V3) / self._V4))
        beta = 0.5 * self._phi * np.cosh((v - self._V3) / (2 * self._V4)) * (1 - np.tanh((v - self._V3) / self._V4))

        dw = np.sqrt(alpha * (1 - w) + beta * w) / np.sqrt(self._Nk)

        return np.array([0, dw])


    # tmax = time to simulate up to
    # x0 = [v0, w0] = IC
    def signal(self, tmax, x0):

        a = lambda t, x: self._a(t, x)

        if self._stochastic == 'euler':
            b = lambda t, x: self._b_euler(t, x)
        else:
            b = lambda t, x: 0

        return ito.sim(a, b, tmax, self._dt, x0)
