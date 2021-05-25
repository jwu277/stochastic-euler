### Simulates Ito processes ###


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


## Simulation function
## dXt = a(t, Xt) dt + b(t, Xt) dBt
## 
## Inputs
##   a:     function handle
##   b:     function handle
##   tmax:  simulation end time
##   dt:    time increment
##   X0:    (optional, default=0) initial value
## Output: Xt array
def sim(a, b, tmax, dt, X0=0):

    # Initialize X
    X = np.empty(int(tmax / dt) + 1)
    X[0] = X0

    # Generate standard normal RVs
    N = norm.rvs(size=len(X))

    # Euler iterations
    for i in range(1, len(X)):
        X[i] = X[i-1] + a(i * dt, X[i-1]) * dt + b(i * dt, X[i-1]) * np.sqrt(dt) * N[i]
    
    return X


def main():

    tmax = 10
    dt = 0.0011

    a = lambda t, x: 0.5
    b = lambda t, x: 1

    t = np.arange(0, tmax, dt)
    B = sim(a, b, tmax, dt)

    plt.plot(t, B)
    plt.show()


if __name__ == "__main__":
    main()

