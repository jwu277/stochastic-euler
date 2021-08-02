### Model B of Morris-Lecar neuron ###


import random
import numpy as np
from scipy import stats


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
        
        # NOMINAL VALUES ; TODO = CHANGE #

        self._psi_a = 0.004
        self._psi_b = 0.02

        self._is_expscale = 100.0
        self._is_exit = 0.005 # nominal value, look at histogram to determine exit...

        # TODO: consider IS transition probability (anywhere in ellipse, not just L) and conditional histogram
        
        # Spiking vs not spiking probability is roughly logistic vs psi
        # TODO: confirm this more accurately
        # Convention: logistic coefficients are for spiking
        # psi0 = logistic 50/50 point, r = logistic rate
        self._spikep_psi0 = 0.017
        self._spikep_r = 220
        
        self._t_inner_mu = 79
        self._t_inner_sigma = 3

        self._t_outer_mu = 102
        self._t_outer_sigma = 1.5

        # Polyval these coefficients
        self._p_inner_mu = [0.5, 0]
        self._p_inner_sigma = [0.1, 0.0005]

        self._p_outer_mu = 0.022
        self._p_outer_sigma = 0.0008


    def _logistic(self, psi):
        return 1 / (1 + np.exp(-self._spikep_r * (psi - self._spikep_psi0)))


    class _Res:


        def __init__(self, t, p):
            self.t = t
            self.p = p

    
    ## Indistinct subthreshold ##
    def _is(self, psi):

        t = stats.expon.rvs(scale=self._is_expscale)
        p = self._is_exit

        return self._Res(t, p)


    ## Distinct subthreshold ##
    def _ds(self, psi):

        if random.random() < self._logistic(psi):
            # Inner distribution
            t = stats.norm.rvs(loc=self._t_inner_mu, scale=self._t_inner_sigma)
            p = stats.norm.rvs(loc=np.polyval(self._p_inner_mu, psi), scale=np.polyval(self._p_inner_sigma, psi))
        else:
            # Outer distribution
            t = stats.norm.rvs(loc=self._t_outer_mu, scale=self._t_outer_sigma)
            p = stats.norm.rvs(loc=self._p_outer_mu, scale=self._p_outer_sigma)

        return self._Res(t, p)


    ## Spiking oscillations ##
    def _so(self, psi):
        
        t = stats.norm.rvs(loc=self._t_outer_mu, scale=self._t_outer_sigma)
        p = stats.norm.rvs(loc=self._p_outer_mu, scale=self._p_outer_sigma)

        return self._Res(t, p)


    ## Simulate spike times in model B
    ## tmax = time to simulate up to
    ## psi0 = initial psi value
    def sim_st(self, tmax, psi0):

        st = [] # array of spike times

        t = 0
        psi = psi0

        tv = [t]
        psiv = [psi]

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

            tv.append(t)
            psiv.append(psi)

        return st, tv, psiv

