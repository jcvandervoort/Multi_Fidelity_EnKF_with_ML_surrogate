
import torch
import numpy as np

from QG import step_RK4, compute_q_over_f0_from_p, compute_psi_from_p, compute_p_from_psi, step_SSP_RK3

from MF_EnKF import remove_boundary, add_boundary

from QG_surrogate import M_surr


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

arr_kwargs = {'dtype':dtype, 'device':device}

import matplotlib.pyplot as plt


def reshape_psi_to_X(psi):

    ## input shape: (N_X,nl,nx,ny)
    ## output shape: (n,N_X)

    N_X = psi.shape[0]
    nx = psi.shape[-2]
    ny = psi.shape[-1]

    X = psi[:,0,:,:]                # we only care about upper layer, shape: (N_X,nx,nx)
    X = X.reshape(N_X, nx*ny)       # reshape to (N_X,n)
    X = X.transpose(0,1)            # reshape to (n,N_X)
    
    return X


def reshape_X_to_psi(X,nl,nx,ny):

    ## input shape: (n,N_X)
    ## output shape: (N_X,nl,nx,nx)

    N_X = X.shape[-1]
    #nx = int(np.sqrt(X.shape[0]))

    #nl = 2
    psi = torch.zeros((N_X,nl,nx,ny), device = device, dtype = dtype)

    X = X.transpose(0,1)          # reshape to (N_X,n)

   # X = X.T
    X = X.reshape(N_X,nx,ny)      # reshape to (N_X,nx,nx)
    psi[:,0,:,:] = X

    return psi


###########################################
## EnKF forecast step
###########################################

def EnKF_forecast(X,dt,nl,nx,ny, mode,k,k_obs,psi_true_k):

  # input: ensemble matrix X, shape: (n,N_X)

  if mode == 'HR':
  ## qg expects input shape: (nens,1,nx,nx), so first reshape input
    psi_old = reshape_X_to_psi(X,nl,nx,ny)

    p_old = compute_p_from_psi(psi_old)   # transform streamfunction to pressure 

    q_old = compute_q_over_f0_from_p(p_old)
    p,q = step_RK4(p_old, q_old,dt)

    psi = compute_psi_from_p(p)       # transform pressure to streamfunction

    ## output psi has shape (nens,2,nx,nx)
    X = reshape_psi_to_X(psi)

  elif mode == 'ML':
    if k % 4 == 0:
      X = M_surr(X, psi_true_k)

  return X

########################################
## EnKF analysis step
########################################

def EnKF_analysis(X,y,H, s_obs, k, rho_X, alpha_X, enkf_type,psi_true_k,remove_bound,loc):

##############################################
  ## calculate prior characteristics
##############################################

  N_X = X.shape[-1]
  m = H.shape[0]


  ## Remove boundary
  if remove_bound == True:
    X = remove_boundary(X)
  
  ## 1) Principal variate
  mu_X = torch.mean(X, axis = -1)
  HX = H @ X
  mu_HX = torch.mean(HX, axis = -1)

  A_X = 1/np.sqrt(N_X-1) * (X - mu_X[:,None] )

  A_HX = H @ A_X

  S_HX = A_HX @ A_HX.T
  S_XHX = A_X @ A_HX.T

##########################################
## apply localization to total variate
##########################################

  if loc:

    HT = (H.T).to_sparse_csc()
    H = H.to_sparse_csr()

    S_XHX = (rho_X @ HT)*S_XHX 
    S_HX = (H @ rho_X @ HT)*S_HX

    S_XHX = S_XHX.to_dense()
    S_HX = S_HX.to_dense()

  R = s_obs**2 * torch.eye(m, device = device)

##################################################
  ## Perturbed observations EnKF
##################################################

  if enkf_type == 'pert':

    K = S_XHX @ torch.linalg.inv(S_HX + R)

    eta_X = torch.randn(m,N_X, device = device)

    E_eta_X = R**(1/2) @ eta_X

    A_eta_X = (E_eta_X - torch.mean(E_eta_X, axis = -1 )[:,None] ) / np.sqrt(N_X - 1)
    A_X = A_X - K @ (A_HX - A_eta_X)

    mu_X = mu_X - K @ (mu_HX - y)

###################################################
  ## Deterministic EnKF
###################################################

  if enkf_type == 'det':

    K = S_XHX @ torch.linalg.inv(S_HX + R)

    A_X = A_X - 0.5*K @ A_HX

    mu_X = mu_X - K @ (mu_HX - y)

##################################
  ### apply inflation
##################################
  A_X = alpha_X * A_X

  X = mu_X[:,None] + np.sqrt(N_X-1)*A_X

  ## add back the boundary
  if remove_bound == True:
    X = add_boundary(X,psi_true_k)

  return X



######################################################
## EnKF run
######################################################

def EnKF_run(N_X,s_obs,rho_X,alpha_X, k_obs,y_obs, X_init,K_da, psi_true,enkf_type,H,dt,remove_bound,
             nl, nx,ny,mode, loc):


###########################
  ### Storage
###########################

  n = X_init.shape[0]

  rmse_Z = np.ones(K_da)*np.inf

  ## store ensemble means
  Z_mfenkf = torch.zeros((K_da,n), **arr_kwargs )

#############################################
## Initialization
#############################################

  X_mfenkf = X_init[:,:N_X].to(device)

  ## initial means
  mu_X = torch.mean(X_mfenkf, axis = -1)
  Z_mfenkf[0] = mu_X

########################################################
## Loop over time
########################################################


  for k in range(1,K_da):

  ##################################
  ## forecast step
  ##################################
    psi_true_k = psi_true[k,0,:,:].unsqueeze(0)

    X_b = EnKF_forecast(X_mfenkf,dt,nl,nx,ny,mode, k, k_obs,psi_true_k)

  ##################################
  ## analysis step
  ##################################

    if k % k_obs == 0:

      X_mfenkf = EnKF_analysis(X_b, y_obs[k,:],H,s_obs, k, rho_X, alpha_X,enkf_type,psi_true_k,remove_bound,loc)

    else:
      X_mfenkf = X_b

  
  ## means
    mu_X = torch.mean(X_mfenkf, axis = -1)

    Z_mfenkf[k,:] = mu_X

    P_mfenkf = torch.var(X_mfenkf, dim = -1)

    ## Calculate RMSE
    burn_in = int(0.10*K_da)


    if k > burn_in:

      a1 = Z_mfenkf[burn_in:k].squeeze()
      a2 = psi_true[burn_in:k,0,:,:].reshape(-1,nx*ny).squeeze()

      a1 = a1.to(device)
      a2 = a2.to(device)

      rmse_Z[k] = torch.sqrt(torch.mean((a1-a2)**2) )
    
      if k % 100 == 0:
        print('RMSE is: ' + str(rmse_Z[k]))

  return rmse_Z, Z_mfenkf, P_mfenkf, X_mfenkf


