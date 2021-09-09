### Simulates Ito processes ###


import numpy as np
from scipy.stats import norm


## Simulation function
## dXt = a(t, Xt) dt + b(t, Xt) dBt
## 
## Inputs
##   a:     function handle
##   b:     function handle
##   tmax:  simulation end time
##   dt:    time increment
##   X0:    (optional, default=0) initial value
##   mat:   (optional, default=False) whether dBt is a vector being multiplied by a matrix
## Output: Xt array
def sim(a, b, tmax, dt, X0=0, mat=False):

    # Initialize X
    X = np.empty((int(tmax / abs(dt)) + 1,) + np.shape(X0))
    X[0] = X0

    # Generate standard normal RVs
    N = norm.rvs(size=X.shape)

    # Euler iterations
    if mat:
        for i in range(1, len(X)):
            X[i] = X[i-1] + a(i * dt, X[i-1]) * dt + np.sqrt(np.heaviside(dt, 0)) * np.matmul(b(i * dt, X[i-1]), N[i])
    else:
        for i in range(1, len(X)):
            X[i] = X[i-1] + a(i * dt, X[i-1]) * dt + b(i * dt, X[i-1]) * np.sqrt(np.heaviside(dt, 0)) * N[i]
    
    return X

