### Morris-Lecar Neuron Model ###


from util import ito
from util import integrate
import numpy as np
from scipy.stats import binom
from neurons.mlr import MorrisLecarR
from neurons.mll import MorrisLecarLin


class MorrisLecar:


    ## Neuron governed by following equations:
    ## m = 0.5 * (1 + tanh((v - V1) / V2))
    ## alpha = 0.5 * phi * cosh((v - V3) / (2 * V4)) * (1 + tanh((v - V3) / V4))
    ## beta = 0.5 * phi * cosh((v - V3) / (2 * V4)) * (1 - tanh((v - V3) / V4))
    ## dv = [(I - gCa * m * (v - VCa) - gK * w * (v - VK) - gL * (v - VL)) / C] dt
    ## dw = [alpha * (1 - w) - beta * w] dt + stochastics
    ## I is e.g. synaptic input
    def __init__(self, I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic=None, Nk=None, method='SE'):

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
        self._method = method

        self._stochastic_store = self._stochastic
        self._method_store = self._method
    

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

        if self._method == 'RK45':
            a = lambda t, x: self._a(t, x)
            return integrate.ivp(a, x0, tmax, self._dt, method='RK45')
        else:

            # Default = stochastic euler

            a = lambda t, x: self._a(t, x)

            if self._stochastic == 'euler':
                b = lambda t, x: self._b_euler(t, x)
            else:
                b = lambda t, x: 0

            return ito.sim(a, b, tmax, self._dt, x0)


    # Sets time direction
    # +1 = forwards
    # -1 = backwards
    def set_time_dir(self, dir):
        self._dt = dir * abs(self._dt)
        if dir < 0:
            self.set_noise(False)


    def set_noise(self, noiseb):
        if noiseb:
            self._stochastic = self._stochastic_store
        else:
            self._stochastic = None
    

    def set_method(self, method):
        self._method = method
    

    def restore_method(self):
        self._method = self._method_store


    ## Neuron needs to be stochastic and have Nk defined to get MLR model
    def gen_mlr(self):
        return MorrisLecarR(self._phi, self._C, self._gL, self._gCa, self._gK, self._VL, self._VCa, self._VK, self._V1, self._V2, self._V3, self._V4, self._dt, self._Nk)

    
    ## Neuron needs to be stochastic and have Nk defined to get MLL model
    def gen_mll(self):
        return MorrisLecarLin(self._phi, self._C, self._gL, self._gCa, self._gK, self._VL, self._VCa, self._VK, self._V1, self._V2, self._V3, self._V4, self._dt, self._Nk)

