from neurons.ml import MorrisLecar
from util import current
import numpy as np
from util import plot
import matplotlib.pyplot as plt


def main():

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

    dt = 0.01
    tmax = 100.0

    Nk = 1000

    phi = 0.04

    t_avg = 0.1
    I_ampl = 100.0
    I = current.constant(tmax, dt, I_ampl)
    # I = current.poisson(tmax, t_avg, dt, I_ampl)

    x0 = np.array([-40.0, 0.2])

    neuron = MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='euler', Nk=Nk)
    x = neuron.signal(tmax, x0)

    # plot.tr(neuron.signal(tmax, x0)[:,0], dt)

    # plot.pp(x)
    plot.pp_scatter(x, step=1)
    plt.xlabel('v')
    plt.ylabel('w')
    plt.show()


if __name__ == "__main__":
    main()

