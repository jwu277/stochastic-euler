### Model B of Morris-Lecar neuron ###


import numpy as np
from util import ito


class ModelB:


    ## Neuron governed by following equations:
    ## m = 0.5 * (1 + tanh((v - V1) / V2))
    ## alpha = 0.5 * phi * cosh((v - V3) / (2 * V4)) * (1 + tanh((v - V3) / V4))
    ## beta = 0.5 * phi * cosh((v - V3) / (2 * V4)) * (1 - tanh((v - V3) / V4))
    ## dv = f(v, w) dt
    ## dw = g(v, w) dt + h(v, w) dBt
    def __init__(self, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, Nk):

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
        self._Nk = Nk

        self._init_params()


    ## Initialize Model B parameters based on ML parameters
    def _init_params(self):
        pass


    class _Res:


        def __init__(self, t, p):
            self.t = t
            self.p = p

    
    ## Indistinct subthreshold ##
    def _is(self, psi):

        t = 0
        p = 0

        return self._Res(t, p)


    ## Distinct subthreshold ##
    def _ds(self, psi):

        t = 0
        p = 0

        return self._Res(t, p)


    ## Spiking oscillations ##
    def _so(self, psi):
        
        t = 0
        p = 0

        return self._Res(t, p)


    ## Simulate spike times in model B
    ## tmax = time to simulate up to
    ## psi0 = initial psi value
    def sim_st(self, tmax, psi0):

        st = [] # array of spike times

        t = 0
        psi = psi0

        while t < tmax:
            
            if psi < self._psi_a:
                # Indistinct subthreshold #
                fn = self._is
            elif psi < self._psi_b:
                # Distinct subthreshold #
                fn = self._ds
            else:
                # Spiking oscillations #
                fn = self._so
                st.append(t)
            
            res = fn(psi)
            t += res.t
            psi = res.p

        return st

