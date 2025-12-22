import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d


torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

arr_kwargs = {'dtype':dtype, 'device':device}


## default parameters

Force = 15              # external forcing
n = 960                 # state dimension
K = 32                  # smoothing parameter
dt = 1/40               # corresponding to 3 hours


def dxdt(x,K,n):

    return prodsum_self(x, K, n) - x + Force


def M_phys(x0):
    """
    RK4 method for solving the Lorenz_2005 equation.
    Input: n x N
    Output: n x N
    """

    n = x0.shape[0]
    K = int(n/30)

    ## make sure input to dxdt has correct shape
    if len(x0.shape) == 1:
        x0 = np.expand_dims(x0, axis = -1).T
    else:
        x0 = np.atleast_2d(x0).T


    ## RK4 time propagation
    k1 = dxdt(x0,K,n)
    k2 = dxdt(x0 + k1*dt/2,K,n)
    k3 = dxdt(x0 + k2*dt/2,K,n)
    k4 = dxdt(x0 + k3*dt,K,n)

    x_new = x0 + 1/6*dt*(k1+2*k2+2*k3+k4)

    ## reshape output to n x N
    x_new = x_new.T.squeeze()

    return x_new


def summation_kernel(width):
    r = width // 2  # "radius"
    weights = np.ones(2 * r + 1)
    if width != len(weights):
        weights[0] = weights[-1] = 0.5
    inds0 = np.arange(-r, r + 1)
    return r, weights, inds0

def boxcar(x, K, n, method="direct"):   # "manual", "fft"or "direct"

    r, weights, inds0 = summation_kernel(K)

    if method == "manual":

        def mod(ind):
            return np.mod(ind, n)

        a = np.zeros_like(x)
        for i in range(n):
            a[..., i] = x[..., mod(i + inds0)] @ weights
            # for i, w in zip(inds0, weights):
            #     a[..., m] += x[..., mod(m + i)] * w

    elif method in ["fft", "oa"]:
        if method == "fft":
            from scipy.signal import fftconvolve as convolver
        else:
            from scipy.signal import oaconvolve as convolver
        weights = weights[... if x.ndim == 1 else None]  # dim compatibility
        xxx = np.hstack([x[..., -r:], x, x[..., :r]])  # wrap
        a = convolver(xxx, weights, axes=-1)
        a = a[..., 2 * r : -2 * r]  # Trim (rm wrapped edges)

    else:  # method == "direct":
        a = convolve1d(x, weights, mode="wrap")

    a /= K
    return a


def shift(x, k):
    return np.roll(x, -k, axis=-1)


def prodsum_self(x, K,n):
    W = boxcar(x, K,n)
    WW = shift(W, -2 * K) * shift(W, -K)
    WX = shift(W, -K) * shift(x, K)
    WX = boxcar(WX, K, n)
    return -WW + WX

#######################################
## Low-Resolution version
#######################################

def M_low(x,r_low):

  ## input: n x N, output: n x N

    n = x.shape[0]

############################
## 1) Upscale x0
#############################

    if len(x.shape) == 1:
        x = np.expand_dims(x, axis = -1)

    x0 = x[::int(n/r_low),:]

###############################################
## 2) propagate lower resolution model forward
###############################################

    # input: r x N, output: r x N

    x0 = M_phys(x0)

    if len(x0.shape) == 1:
        x0 = np.expand_dims(x0, axis = -1)

##############################################
## 3) Downscale x0 using linear interpolation
##############################################

    fine_grid = np.linspace(0,n,n+1)
    coarse_grid = np.linspace(0,n,r_low+1)

# extend periodically

    N = x.shape[-1]

    Y = np.zeros((r_low+1,N) )
    Y[:-1,:] = x0
    Y[-1,:] = x0[0,:]

    x_out = np.zeros((n+1,N))

    for i in range(N):
      x_out[:,i] = np.interp(fine_grid, coarse_grid, Y[:,i])

    x_out = x_out[:-1,:]

    return x_out.squeeze()



