### Hybrid Covariance Super-Resolution EnKF

import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import mean_squared_error

from QG import step_RK4, compute_q_over_f0_from_p, compute_psi_from_p, compute_p_from_psi, M_low

#from EnKF import reshape_psi_to_X,reshape_X_to_psi

from QG_surrogate import M_surr

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

arr_kwargs = {'dtype':dtype, 'device':device}


def reshape_psi_to_X(psi):

    ## input shape: (N_X,nl,nx,ny)
    ## output shape: (n,N_X)

    N_X = psi.shape[0]
    nl = psi.shape[-3]
    nx = psi.shape[-2]
    ny = psi.shape[-1]

    X = psi[:,0,:,:]                # we only care about upper layer, shape: (N_X,nx,nx)
    X = X.reshape(N_X, nx*ny)       # reshape to (N_X,n)
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

###########################################
## Remove / add boundary from ensemble

def remove_boundary(X):
    psi_X = reshape_X_to_psi(X)
    psi_noBC = psi_X[:,:,1:-1,1:-1]
    X_noBC = reshape_psi_to_X(psi_noBC)
    return X_noBC


def add_boundary(X, psi):
    N_X = X.shape[1]
    nx = int(np.sqrt(X.shape[0]) + 2)
   
   
    psi = psi.reshape(-1,nx,nx)
    nl = 2
    psi_boundary = torch.zeros((N_X,nl,nx,nx))
    psi_boundary[...,0,1:-1] = psi[...,0,1:-1]
    psi_boundary[...,-1,1:-1] = psi[...,-1,1:-1]
    psi_boundary[...,:,0] = psi[...,:,0]
    psi_boundary[...,:,-1] = psi[...,:,-1]

    psi_noBC = reshape_X_to_psi(X)

    psi_boundary[...,1:-1,1:-1] = psi_noBC

    X_new = reshape_psi_to_X(psi_boundary)
    return X_new


###########################################
## MF-EnKF forecast step
###########################################


def EnKF_forecast(X,dt):

  # input: ensemble matrix X, shape: (n,N_X)

  ## qg expects input shape: (nens,1,nx,nx), so first reshape input
  psi_old = reshape_X_to_psi(X)

  p_old = compute_p_from_psi(psi_old)   # transform streamfunction to pressure 

  q_old = compute_q_over_f0_from_p(p_old)
  p,q = step_RK4(p_old, q_old,dt)

  psi = compute_psi_from_p(p)       # transform pressure to streamfunction

  ## output psi has shape (nens,2,nx,nx)

  X_new = reshape_psi_to_X(psi)

  return X_new


def baseline_forecast(X, U, mode, r_low,dt,k,k_obs,psi_true_k):

  X = EnKF_forecast(X,dt)

  if mode == 'LR':
    
    U = M_low(U,r_low)

  elif mode == 'ML':

    if k % k_obs == 0:
      U = M_surr(U,psi_true_k).to(device)

  return X, U


########################################
## MF-EnKF analysis step
########################################


def baseline_analysis(X,U,y,H,s_obs, rho_Z, alpha_Z, enkf_type,psi_true_k,remove_bound,recenter):

##############################################
  ## calculate prior characteristics
##############################################

  N_X = X.shape[-1]
  n = X.shape[0]

  N_U = U.shape[-1]
  m = H.shape[0]

  mu_X = torch.mean(X, axis = -1)
  mu_U = torch.mean(U, axis = -1)

  if N_U != 1:
    A_U = 1/np.sqrt(N_U-1) * (U - mu_U[:,None] )
  else:
    A_U = torch.zeros((n,1) )

###########################################
  ## recenter ML ensemble with HR mean
###########################################

  if recenter == True:
    U = mu_X[:,None] + np.sqrt(N_U-1)*A_U

#############################################
## combine ensemble into one
#############################################

  Z = torch.concat((X,U), dim = -1)  # shape: (n,N_X+N_U)

###########################################
  ## remove boundary
###########################################  

  if remove_bound == True:
    Z = remove_boundary(Z)

######################################
  ## Total variate characteristics
######################################

  mu_Z = torch.mean(Z, axis = -1)
  mu_HZ = H @ mu_Z
  HZ = H @ Z

  N_Z = N_X + N_U

  A_Z = 1/np.sqrt(N_Z-1) * (Z - mu_Z[:,None] )
  A_HZ = 1/np.sqrt(N_Z-1) * (HZ - mu_HZ[:,None] )
  

  S_HZ = A_HZ @ A_HZ.T

  S_ZHZ = A_Z @ A_HZ.T


##########################################
## apply localization to total variate
##########################################

  if rho_Z != None:
     
    HT = (H.T).to_sparse_csc() 
    H = H.to_sparse_csr()

    S_ZHZ = (rho_Z @ HT)*S_ZHZ
    S_HZ = (H @ rho_Z @ HT)*S_HZ

    S_ZHZ = S_ZHZ.to_dense()
    S_HZ = S_HZ.to_dense()

  R = s_obs**2 * torch.eye(m, device = device, dtype = dtype)


############################################
  # Deterministic EnKF
############################################

  if enkf_type == 'det':
    S_eta_Z = R

    K = S_ZHZ @ torch.linalg.inv(S_HZ + S_eta_Z)

    A_Z = A_Z - 0.5*K @ A_HZ

    mu_Z = mu_Z - K @ (mu_HZ - y)



###########################################
  ## Perturbed obs EnKF

  elif enkf_type == 'pert':

    S_eta_Z = R
    K = S_ZHZ @ torch.linalg.inv(S_HZ + S_eta_Z)

    eta_Z = torch.randn(m,N_Z, **arr_kwargs)

    E_eta_Z = S_eta_Z**(1/2) @ eta_Z

    A_eta_Z = (E_eta_Z - torch.mean(E_eta_Z, axis = -1 )[:,None] ) / np.sqrt(N_Z - 1)

    A_Z = A_Z - K @ (A_HZ - A_eta_Z)
  
    mu_Z = mu_Z - K @ (mu_HZ - y)


##################################
  ### apply inflation
################################
  A_Z = alpha_Z * A_Z

###############################################
  ## add back the boundary
################################################  

  ## update Z
  Z = mu_Z[:,None] + np.sqrt(N_Z-1)*A_Z
 
  if remove_bound == True:
    Z = add_boundary(Z,psi_true_k)


  X = Z[:,:N_X]  # shape: (n,N_X)
  U = Z[:,N_X:] # shape: (n,N_U)


  return X,U

#################################################
## MF-EnKF run
#################################################


from sklearn.metrics import mean_squared_error


def baseline_run(N_X,N_U,s_obs,rho_Z, alpha_Z, k_obs,y_obs,
                Z_init, K_da, psi_true, mode, H, r_low, enkf_type, dt, remove_bound,recenter):


###########################
  ### Storage
###########################

  n = Z_init.shape[0]
  nx = int(np.sqrt(n) )


  ## store RMSE values
  rmse_Z = np.ones(K_da)*np.inf 

  mu_Z = torch.zeros((K_da,n), **arr_kwargs)

  mu_X = torch.zeros((K_da,n), **arr_kwargs)
  mu_U = torch.zeros((K_da,n), **arr_kwargs)


  P_X = np.zeros(K_da)
  P_U = np.zeros(K_da)
  P_Z = np.zeros(K_da)

#############################################
## Initialization
#############################################

  ## initial means
  N_Z = N_X + N_U
  mu_Z[0] = torch.mean(Z_init[:,:N_Z], axis = -1)

  X = Z_init[:,:N_X].to(device)  # shape: (n,N_X)
  U = Z_init[:,N_X:N_X+N_U].to(device) # shape: (n,N_U)

  mu_X[0] = torch.mean(X, axis = -1)
  mu_U[0] = torch.mean(U, axis = -1)

  P_X[0] = torch.var(X,dim=-1).mean()
  P_U[0] = torch.var(U,dim=-1).mean()
  P_Z[0] = torch.var(Z_init[:,:N_X+N_U], dim = -1).mean()

########################################################
## Loop over time
########################################################


  for k in range(1,K_da):


  ##################################
  ## forecast step
  ##################################
 
    psi_true_k = psi_true[k,:].unsqueeze(0)

    X_b, U_b = baseline_forecast(X, U, mode, r_low,dt,k,k_obs,psi_true_k)

  ##################################
  ## analysis step
  ##################################

    if k % k_obs == 0:
      psi_true_k = psi_true[k,:].unsqueeze(0)
      X, U = baseline_analysis(X_b,U_b,y_obs[k,:],H,s_obs, rho_Z, alpha_Z, enkf_type,psi_true_k,remove_bound,recenter)
  
    else:
      X, U = X_b, U_b

  
  ## means
    X = X.to(device)
    U = U.to(device)
    Z = torch.concat((X,U), dim = -1)  # shape: (n,N_X+N_U)

    ## either use mean of Z or the mean of X (so excl DL members)
    mu_Z[k] = torch.mean(Z, axis = -1)
    mu_X[k] = torch.mean(X, axis = -1)
    mu_U[k] = torch.mean(U, axis = -1)

    # add [k] to store at every time step
    P_X[k]= torch.var(X, dim = -1).mean()
    P_U[k] = torch.var(U, dim = -1).mean()

    P_Z[k] = torch.var(Z, dim = -1).mean()

    
    ## Calculate RMSE
    burn_in = int(0.10*K_da)

    if k > burn_in:

      a1 = mu_Z[burn_in:k,:].squeeze()
      a2 = psi_true[burn_in:k,0,:,:].reshape(-1,n).squeeze()

      a1.to(device)
      a2.to(device)

      rmse_Z[k] = torch.sqrt(torch.mean((a1-a2)**2))


  return rmse_Z,Z,P_Z, P_X, P_U, X, U







