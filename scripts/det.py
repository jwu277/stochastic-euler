import numpy as np
import matplotlib.pyplot as plt

from neurons.ml import MorrisLecar
from neurons.mll import MorrisLecarLin
from neurons.mlj import MorrisLecarJacobi

from util import current
from util import dyn
from util import plot

import time


def _gen_neuron(dt):

    # C ~ uF
    # g ~ mS
    # V ~ mV
    # t ~ ms
    
    C = 20
    gL = 2.0
    gCa = 4.4
    gK = 8
    VL = -60
    VCa = 120
    VK = -84
    V1 = -1.2
    V2 = 18.0
    V3 = 2.0
    V4 = 30.0
    
    tmax_I = 1500.0

    Nk = np.Inf

    phi = 0.04

    I_ampl = 90
    I = current.constant(tmax_I, dt, I_ampl)

    return MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='euler', Nk=Nk)


def _get_eq(neuron, dt):

    eq0 = np.array([-30, 0.13])
    dist = np.array([0.1, 0.001])

    tmax1 = 1000.0
    tmax2 = 1100.0

    return dyn.get_fixed_pt(neuron, eq0, tmax1, tmax2, dist, dt)


def main():

    ## 1. Generate neurons ##
    dt = 0.1
    neuron = _gen_neuron(dt)
    mll = neuron.gen_model(MorrisLecarLin)
    mlj = neuron.gen_model(MorrisLecarJacobi)

    ## 2. Get equilibrium point ##
    eq = _get_eq(neuron, dt)

    ## 3. Initialize neurons ##

    dv = 0.2
    dw = 0.002
    mll.init(eq, dv, dw)

    mlj.init(eq)

    ## 4. Generate signal ##
    
    tmax = 200.0

    ## 4a. IC ##
    cdist = 30.0
    alpha = 0.0
    x0 = mll.new2og(cdist * np.array([[np.cos(alpha), np.sin(alpha)]]), np.array([0]))[0]

    ## 4b. Compute trajectories ##
    t = time.time()
    xn = neuron.signal(tmax, x0)
    xl = mll.signal(tmax, x0)
    xj = mlj.signal(tmax, x0)
    print(f'Computation time: {time.time() - t}')

    ## 5. Plot results ##

    ## 5a. Plot ellipse ##

    fig, ax = plt.subplots()

    ax.set_title('Deterministic Trajectories')
    ax.set_xlabel('v')
    ax.set_ylabel('w')
    
    numpts = 1000
    theta = np.linspace(0, 2 * np.pi, numpts)
    circ = cdist * np.transpose(np.array([np.cos(theta), np.sin(theta)]))
    elpts = mll.new2og(circ, np.zeros(numpts))

    ax.plot(elpts[:,0], elpts[:,1], ls='dotted', c='blue', label='Generated FPE')

    ## 5b. Plot trajectories ##

    plot.pp(xn, ax=ax, label='Nonlinear')
    plot.pp(xl, ax=ax, label='Linear')
    # plot.pp(xj, label='Jacobi')

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()

