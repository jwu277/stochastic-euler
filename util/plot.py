## Util for generating input currents


import numpy as np
import matplotlib.pyplot as plt


# Plot time response
def tr(x, dt):
    plt.plot(np.arange(0, dt * len(x), dt), x)


# Plot smooth phase portrait
# x = [x1(t), x2(t)]
def pp(x):
    plt.plot(x[:,0], x[:,1])


# Plot scattered phase portrait
# x = [x1(t), x2(t)]
def pp_scatter(x, step=1, c='k'):
    plt.scatter(x[::step,0], x[::step,1], s=12, c=c, zorder=2.5)

