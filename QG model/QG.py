import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.backends.cudnn.deterministic = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

arr_kwargs = {'dtype':dtype, 'device':device}


def reshape_psi_to_X(psi):

    ## input shape: (N_X,nl,nx,nx)
    ## output shape: (n,N_X)

    N_X = psi.shape[0]
    nx = psi.shape[-1]

    X = psi[:,0,:,:]                # we only care about upper layer, shape: (N_X,nx,nx)
    X = X.reshape(N_X, nx**2)       # reshape to (N_X,n)
    X = X.transpose(0,1)            # reshape to (n,N_X)

    return X


def reshape_X_to_psi(X):

    ## input shape: (n,N_X)
    ## output shape: (N_X,nl,nx,nx)

    N_X = X.shape[-1]
    nx = int(np.sqrt(X.shape[0]))

    nl = 2
    psi = torch.zeros((N_X,nl,nx,nx), device = device, dtype = dtype)

    X = X.transpose(0,1)          # reshape to (N_X,n)

    X = X.reshape(N_X,nx,nx)      # reshape to (N_X,nx,nx)
    psi[:,0,:,:] = X

    return psi


## default parameters

Lx = 3072e3             # length in x direction (m)
Ly = 3072e3             # length in y direction (m)
H = 1664                # depth (m)
heights = [H, 1e99]
nl = 2                  # number of layers
gprime = [0.025,0.025]  # reduced gravities (m/s^2)
f0 = 8.4e-5             # coriolis (s^-1)
a_2 = 0.                # laplacian diffusion coeff (m^2/s)
a_4 = 1.03e10           # increase to 4e8 for stability
beta = 1.88e-11         # coriolis gradient (m^-1 s^-1)
delta_ek = 0.0          # eckman height (m)
dt = 1*3600*6           # time step (in seconds)
bcco = 0.0              # boundary condition coeff (non-dim.)
tau0 = 1.7e-4           # wind stress magnitude (m/s^2)

nx = 128
ny = 128


zfbc = bcco / (1 + 0.5*bcco)
zfbc = torch.tensor(zfbc, **arr_kwargs) # convert to Tensor for tracing

## create grid
y0 = 0.5*Ly

dx = Lx / (nx-1)
dy = Ly / (ny-1)

## functions to go from pressure to streamfunction

scaling_nondim = heights[0]*beta / (tau0 * f0)

def compute_psi_from_p(p):
    return scaling_nondim * p

def compute_p_from_psi(psi):
    return 1/scaling_nondim * psi


## Functions to solve elliptic equation with homogeneous boundary conditions

def compute_laplace_dst(nx, ny, dx, dy, arr_kwargs):
    """Discrete sine transform of the 2D centered discrete laplacian
    operator."""
    x, y = torch.meshgrid(torch.arange(1,nx-1, **arr_kwargs),
                          torch.arange(1,ny-1, **arr_kwargs),
                          indexing='ij')
    return 2*(torch.cos(torch.pi/(nx-1)*x) - 1)/dx**2 + 2*(torch.cos(torch.pi/(ny-1)*y) - 1)/dy**2


def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform."""
    return torch.fft.irfft(-1j*F.pad(x, (1,1)), dim=-1, norm=norm)[...,1:x.shape[-1]+1]


def dstI2D(x, norm='ortho'):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).transpose(-1,-2), norm=norm).transpose(-1,-2)


def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
       using float32 discrete sine transform."""
    return dstI2D(dstI2D(f.type(torch.float32)) / operator_dst).type(torch.float64)


## Discrete spatial differential operators

def jacobi_h(f, g):
    """Arakawa discretisation of Jacobian J(f,g).
       Scalar fields f and g must have the same dimension.
       Grid is regular and dx = dy."""
    dx_f = f[...,2:,:] - f[...,:-2,:]
    dx_g = g[...,2:,:] - g[...,:-2,:]
    dy_f = f[...,2:] - f[...,:-2]
    dy_g = g[...,2:] - g[...,:-2]
    return (
            (   dx_f[...,1:-1] * dy_g[...,1:-1,:] - dx_g[...,1:-1] * dy_f[...,1:-1,:]  ) +
            (   (f[...,2:,1:-1] * dy_g[...,2:,:] - f[...,:-2,1:-1] * dy_g[...,:-2,:]) -
                (f[...,1:-1,2:]  * dx_g[...,2:] - f[...,1:-1,:-2] * dx_g[...,:-2])     ) +
            (   (g[...,1:-1,2:] * dx_f[...,2:] - g[...,1:-1,:-2] * dx_f[...,:-2]) -
                (g[...,2:,1:-1] * dy_f[...,2:,:] - g[...,:-2,1:-1] * dy_f[...,:-2,:])  )
           ) / 12.


def laplacian_h_boundaries(f, fc):
    return fc*(torch.cat([f[...,1,1:-1],f[...,-2,1:-1], f[...,1], f[...,-2]], dim=-1) -
               torch.cat([f[...,0,1:-1],f[...,-1,1:-1], f[...,0], f[...,-1]], dim=-1))


def laplacian_h_nobc(f):
    return (f[...,2:,1:-1] + f[...,:-2,1:-1] + f[...,1:-1,2:] + f[...,1:-1,:-2]
            - 4*f[...,1:-1,1:-1])

def matmul(M, f):
    return (M @ f.reshape(f.shape[:-2] + (-1,))).reshape(f.shape)


def laplacian_h(f, fc):
    delta_f = torch.zeros_like(f)
    delta_f[...,1:-1,1:-1] = laplacian_h_nobc(f)
    delta_f_bound = laplacian_h_boundaries(f, fc)
    nx, ny = f.shape[-2:]
    delta_f[...,0,1:-1] = delta_f_bound[...,:ny-2]
    delta_f[...,-1,1:-1] = delta_f_bound[...,ny-2:2*ny-4]
    delta_f[...,0] = delta_f_bound[...,2*ny-4:nx+2*ny-4]
    delta_f[...,-1] = delta_f_bound[...,nx+2*ny-4:2*nx+2*ny-4]
    return delta_f


def grad_perp(f):
    """Orthogonal gradient computed ...,on staggered grid."""
    return f[...,:-1] - f[...,1:], f[...,1:,:] - f[...,:-1,:]


def curl_wind(tau, dx, dy):
    tau_x = 0.5 * (tau[:-1,:,0] + tau[1:,:,0])
    tau_y = 0.5 * (tau[:,:-1,1] + tau[:,1:,1])
    curl_stagg = (tau_y[1:] - tau_y[:-1]) / dx - (tau_x[:,1:] - tau_x[:,:-1]) / dy
    return  0.25*(curl_stagg[:-1,:-1] + curl_stagg[:-1,1:] + curl_stagg[1:,:-1] + curl_stagg[1:,1:])


## Initialize matrices

def compute_A_matrix():

    A = torch.zeros((nl,nl), **arr_kwargs)
    A[0,0] = 1./(heights[0] * gprime[0])
    A[0,1] = -1./(heights[0]*gprime[0])
    for i in range(1,nl-1):
        A[i,i-1] = -1./(heights[i]*gprime[i-1])
        A[i,i] = 1./heights[i]*(1/gprime[i] + 1/gprime[i-1])
        A[i,i+1] = -1./(heights[i]*gprime[i])
    A[-1,-1] = 1./(heights[nl-1]*gprime[nl-2])
    A[-1,-2] = -1./(heights[nl-1]*gprime[nl-2])

    A = A.unsqueeze(0)

    return A


def compute_layer_to_mode_matrices(A):

    A = A[0]

    lambd_r, R = torch.linalg.eig(A)
    lambda_l, L = torch.linalg.eig(A.T)
    lambd = lambd_r.real
    R, L = R.real, L.real
    Cl2m = torch.diag(1./torch.diag(L.T @ R)) @ L.T
    Cm2l = R
    Cl2m.unsqueeze_(0)
    Cm2l.unsqueeze_(0)
    
    return lambd, Cl2m, Cm2l

def compute_helmholtz_dst(lambd,nx,ny,dx,dy):

    helmholtz_dst = compute_laplace_dst(nx, ny, dx, dy, arr_kwargs).reshape((1, nx-2, ny-2)) / f0**2 - lambd.reshape((nl,1,1) )

    constant_field = torch.ones((nl,nx,ny), **arr_kwargs) / (nx*ny)
    s_solutions = torch.zeros_like(constant_field)
    s_solutions[:,1:-1,1:-1] = inverse_elliptic_dst(constant_field[:,1:-1,1:-1], helmholtz_dst)

    homogeneous_sol = (constant_field + s_solutions*lambd.reshape((nl,1,1) ) )[:-1]

    helmholtz_dst.unsqueeze_(0)
    homogeneous_sol.unsqueeze_(0)

    return helmholtz_dst.type(torch.float32), homogeneous_sol


def compute_alpha_matrix(Cm2l, Cl2m, hom_sol):


    Cm2l, Cl2m, hom_sol = Cm2l[0], Cl2m[0], hom_sol[0]
    M = (Cm2l[1:] - Cm2l[:-1])[:nl-1,:nl-1] * hom_sol.mean((1,2)).reshape((1,nl-1) )
    M_inv = torch.linalg.inv(M)
    alpha_matrix = -M_inv @ (Cm2l[1:,:-1] - Cm2l[:-1,:-1] )
    
    alpha_matrix = alpha_matrix.unsqueeze(0)

    return alpha_matrix
    

def compute_q_over_f0_from_p(p):

    nx = p.shape[-2]
    ny = p.shape[-1]
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x,y = torch.meshgrid(torch.linspace(0, Lx, nx, **arr_kwargs), torch.linspace(0, Ly, ny,**arr_kwargs), indexing='ij')

    A = compute_A_matrix()

    Ap = (A @ p.reshape(p.shape[:len(p.shape)-2]+(-1,))).reshape(p.shape)
    q_over_f0 = laplacian_h(p, zfbc) / (f0*dx)**2 - Ap + (beta / f0) * (y - y0)

    return q_over_f0

def compute_u(p):
    """Compute velocity on staggered grid."""
    dx = Lx / (p.shape[-1] - 1)
    return grad_perp(p/(f0*dx))



def advection_rhs(p, q_over_f0):
    """Advection diffusion RHS for vorticity, only inside domain"""

    nx = p.shape[-2]
    ny = p.shape[-1]
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    diff_coef = a_2 / f0**2 / dx**4
    hyperdiff_coef = (a_4 / f0**2) / dx**6
    jac_coef = 1 / (f0*dx*dy)
    bottom_friction_coef = -delta_ek / (2*np.abs(f0)*dx**2*heights[-1])

    rhs = jac_coef * jacobi_h(q_over_f0, p)

    delta2_p = laplacian_h(p, zfbc)
    if a_2 != 0.:
        rhs += diff_coef * laplacian_h_nobc(delta2_p)
    if a_4 != 0.:
        rhs -= hyperdiff_coef * laplacian_h_nobc(laplacian_h(delta2_p, zfbc))


    ## wind forcing
    tau = torch.zeros((nx,ny,2), **arr_kwargs)
    tau[:,:,0] = -tau0 * torch.cos(2*torch.pi*(torch.arange(ny, **arr_kwargs)+0.5)/ny).reshape((1, ny))
    wind_forcing = (curl_wind(tau, dx, dy) / (f0 * heights[0])).unsqueeze(0)
    wind_forcing.unsqueeze_(0)

    rhs[...,0:1,:,:] += wind_forcing
    rhs[...,-1:,:,:] += bottom_friction_coef * laplacian_h_nobc(p[...,-1:,:,:])

    return rhs


def compute_time_derivatives(p,q_over_f0):
    # advect vorticity inside of the domain

    nx = p.shape[-2]
    ny = p.shape[-1]

    dx = Lx / (nx-1)
    dy = Ly / (ny-1)

    dq_over_f0 = F.pad(advection_rhs(p,q_over_f0), (1,1,1,1))

    # Solve helmoltz eq for pressure

    A = compute_A_matrix()
    lambd, Cl2m, Cm2l = compute_layer_to_mode_matrices(A)
    helmholtz_dst,hom_sol = compute_helmholtz_dst(lambd,nx,ny,dx,dy)


    rhs_helmholtz = matmul(Cl2m, dq_over_f0)
    dp_modes = F.pad(inverse_elliptic_dst(rhs_helmholtz[...,1:-1,1:-1], helmholtz_dst), (1,1,1,1))

    # Ensure mass conservation
    alpha_matrix = compute_alpha_matrix(Cm2l, Cl2m,hom_sol)
    dalpha =  (alpha_matrix @ dp_modes[...,:-1,:,:].mean((-2,-1)).unsqueeze(-1)).unsqueeze(-1)
    dp_modes[...,:-1,:,:] += dalpha * hom_sol
    dp = matmul(Cm2l, dp_modes)


    dp_bound = torch.cat([dp[...,0,1:-1], dp[...,-1,1:-1], dp[...,:,0], dp[...,:,-1]], dim=-1)
    delta_p_bound = laplacian_h_boundaries(dp/(f0*dx)**2, zfbc)
    dq_over_f0_bound = delta_p_bound - A @ dp_bound
    dq_over_f0[...,0,1:-1] = dq_over_f0_bound[...,:ny-2]
    dq_over_f0[...,-1,1:-1] = dq_over_f0_bound[...,ny-2:2*ny-4]
    dq_over_f0[...,0] = dq_over_f0_bound[...,2*ny-4:nx+2*ny-4]
    dq_over_f0[...,-1] = dq_over_f0_bound[...,nx+2*ny-4:2*nx+2*ny-4]

    return dp, dq_over_f0


#####################################
## RK4 method
#####################################

def step_RK4(P,Q,dt):

    k1_p,k1_q = compute_time_derivatives(P,Q)
    q = Q + dt/2*k1_q
    p = P + dt/2*k1_p

    k2_p,k2_q = compute_time_derivatives(p,q)
    q = Q + dt/2*k2_q
    p = P + dt/2*k2_p

    k3_p,k3_q = compute_time_derivatives(p,q)
    q = Q + k3_q*dt
    p = P + k3_p*dt

    k4_p,k4_q = compute_time_derivatives(p,q)

    p = P + 1/6*dt*(k1_p+2*k2_p+2*k3_p+k4_p)
    q = Q + 1/6*dt*(k1_q+2*k2_q+2*k3_q+k4_q)

    return p,q


######################################
## Stability-preserving RK3 method
######################################

def step_SSP_RK3(P,Q):
        """ Time itegration with SSP-RK3 scheme."""

        # q is q over f0
        dp_0, dq_0 = compute_time_derivatives(P,Q)
        q = Q + dt * dq_0
        p = P + dt * dp_0

        dp_1, dq_1 = compute_time_derivatives(p,q)
        q = q + (dt/4)*(dq_1 - 3*dq_0)
        p = p + (dt/4)*(dp_1 - 3*dp_0)

        dp_2, dq_2 = compute_time_derivatives(p,q)
        q = q + (dt/12)*(8*dq_2 - dq_1 - dq_0)
        p = p + (dt/12)*(8*dp_2 - dp_1 - dp_0)

        return p,q


#######################################
## RK2 method
#######################################

def step_RK2(P,Q):

    dp_0, dq_0 = compute_time_derivatives(P,Q)

    q = Q + dt*dq_0
    p = P + dt*dp_0

    dp_1, dq_1 = compute_time_derivatives(p,q)

    q = q + dt* 0.5 * (dq_1 - dq_0)
    p = p + dt * 0.5 * (dp_1 - dp_0)

    return p,q


## Compute CFL number

def compute_CFL(p):
    u,v = compute_u(p)

    nx = p.shape[-1]

    dx = Lx/(nx-1)
    dy = dx

    CFL = dt*(u.max() / dx + v.max() / dy)

    return CFL


##################################################################3

### Up-and downsampling


def UpDownsample(psi_in, nx):

    psi_out = F.interpolate(psi_in, size = (nx,nx), mode = 'nearest')
    return psi_out


def M_low(X,r,psi_true_k):

  ## input shape: n x N, output shape: n x N
  ## r: lower resolution

    N_X = X.shape[-1]
#############################
## 1) Upscale
#############################

    ## reshape input
    psi_in = reshape_X_to_psi(X)
    nx = psi_in.shape[-1]

    ## upscale
    psi_r = UpDownsample(psi_in,r)

###############################################
## 2) propagate lower resolution model forward
###############################################

    # input: (N_X,nl,r,r), output: (N_X,nl,r,r)

    p_old = compute_p_from_psi(psi_r)
    q_old = compute_q_over_f0_from_p(p_old)

    p,q = step_RK4(p_old, q_old,dt)
    psi = compute_psi_from_p(p)


    dpsi = psi #psi - psi_r          # increment in LR space (CHange to increments !!)

##############################################
## 3) Downscale using linear interpolation
##############################################

    ## downsample ensemble

    ## input: (N_X,nl,r,r) with r low dimension size
    ## output: (N_X,nl,n,n)

    dpsi_inter = UpDownsample(dpsi,nx) 

    ## reshape output
    nl = 2
    dpsi_new = torch.zeros((N_X,nl,nx,nx), **arr_kwargs)
    dpsi_new[:,0,:,:] = dpsi_inter[:,0,:,:].squeeze()


    psi_new = torch.zeros((N_X,nl,nx,nx), **arr_kwargs)
    psi_new[:,0,:,:] = dpsi_new[:,0,:,:]

## keep boundary fixed
    psi_new[...,0,1:-1] = psi_true_k[...,0,1:-1]
    psi_new[...,-1,1:-1] = psi_true_k[...,-1,1:-1]
    psi_new[...,0] = psi_true_k[...,0]
    psi_new[...,-1] = psi_true_k[...,-1]


    X_inter = reshape_psi_to_X(psi_new)


    return X_inter
