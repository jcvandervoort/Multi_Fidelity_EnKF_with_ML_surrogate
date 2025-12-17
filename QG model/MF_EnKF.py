### MF-EnKF

import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import mean_squared_error

from QG import step_RK4, compute_q_over_f0_from_p, compute_psi_from_p, compute_p_from_psi, M_low


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
    #X = X.T
    
    return X


def reshape_X_to_psi(X):

    ## input shape: (n,N_X)
    ## output shape: (N_X,nl,nx,nx)

    N_X = X.shape[-1]
    nx = int(np.sqrt(X.shape[0]))

    nl = 2
    psi = torch.zeros((N_X,nl,nx,nx), device = device, dtype = dtype)

    X = X.transpose(0,1)          # reshape to (N_X,n)

   # X = X.T
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
    #print('Shape of psi is: ' + str(psi.shape) )
    #print('Shape of X is: ' + str(X.shape) )
    psi = psi.reshape(-1,nx,nx)
    nl = 2
    psi_boundary = torch.zeros((N_X,nl,nx,nx))
    #print('nx is: ' + str(nx) )
    #print('Shape of psi boundary: ' + str(psi_boundary.shape))
    psi_boundary[...,0,1:-1] = psi[...,0,1:-1]
    psi_boundary[...,-1,1:-1] = psi[...,-1,1:-1]
    psi_boundary[...,:,0] = psi[...,:,0]
    psi_boundary[...,:,-1] = psi[...,:,-1]

    psi_noBC = reshape_X_to_psi(X)
    #print('Shape of psi_noBC is: ' + str(psi_noBC.shape) )
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


def MFEnKF_forecast(X,Uhat, U, mode,r_low, dt,k,psi_true_k):

  X = EnKF_forecast(X,dt) 

  if mode == 'LR':
    
    Uhat = M_low(Uhat,r_low,psi_true_k) 
    U = M_low(U,r_low,psi_true_k)


  elif mode == 'ML':

    if k % 4 == 0:
      Uhat = M_surr(Uhat,psi_true_k) 
      U = M_surr(U, psi_true_k) 


  return X,Uhat,U


########################################
## MF-EnKF analysis step
########################################


def MFEnKF_analysis(X,Uhat,U,y,H,s_obs, k, lambda_, rho_Z, alpha_X,alpha_Uhat, alpha_U,recenter, adjust_corr,control,enkf_type,psi_true_k,remove_bound):

##############################################
  ## calculate prior characteristics
##############################################

  N_X = X.shape[-1]
  N_U = U.shape[-1]
  m = H.shape[0]

###########################################
  ## remove boundary
###########################################  

  if remove_bound == True:
    X = remove_boundary(X)
    Uhat = remove_boundary(Uhat)
    U = remove_boundary(U)


## 1) Principal variate
  mu_X = torch.mean(X, axis = -1)
  HX = H @ X
  mu_HX = torch.mean(HX, axis = -1)
  
  if N_X > 1:
    A_X = 1/np.sqrt(N_X-1) * (X - mu_X[:,None] )
    A_HX = 1/np.sqrt(N_X-1) * (HX - mu_HX[:,None] )
  else:
    A_X = torch.zeros((X.shape[0],1), **arr_kwargs)
    A_HX = torch.zeros((m,1), **arr_kwargs )

  ## 2) Control variate

  ## enable/disable control variate
  if control == False:
    Uhat = X

  mu_Uhat = torch.mean(Uhat, axis = -1)
  HUhat = H @ Uhat
  mu_HUhat = torch.mean(HUhat, axis = -1)

  if N_X > 1:
    A_Uhat = 1/np.sqrt(N_X-1) * (Uhat - mu_Uhat[:,None] )
    A_HUhat = 1/np.sqrt(N_X-1) * (HUhat - mu_HUhat[:,None] )
  else:
    A_Uhat = torch.zeros((X.shape[0],1), **arr_kwargs)
    A_HUhat = torch.zeros((m,1), **arr_kwargs )


  ## 3) Ancillary variate
  mu_U = torch.mean(U, axis = -1)
  A_U = 1/np.sqrt(N_U-1) * (U - mu_U[:,None] )

  HU = H @ U
  mu_HU = torch.mean(HU, axis = -1)
  A_HU = 1/np.sqrt(N_U-1) * (HU - mu_HU[:,None] )


######################################
  ## Total variate characteristics
######################################

  mu_Z = mu_X - lambda_*(mu_Uhat - mu_U)
  mu_HZ = mu_HX - lambda_*(mu_HUhat - mu_HU)


  if N_X > 1:
    S_HZ = A_HX @ A_HX.T + lambda_**2*(A_HUhat @ A_HUhat.T) + lambda_**2*(A_HU @ A_HU.T) - lambda_*(A_HX @ A_HUhat.T) - lambda_*(A_HUhat @ A_HX.T) 

    S_ZHZ = A_X @ A_HX.T + lambda_**2*(A_Uhat @ A_HUhat.T) + lambda_**2*(A_U @ A_HU.T) - lambda_*(A_X @ A_HUhat.T) - lambda_*(A_Uhat @ A_HX.T)

  else:
    S_HZ = A_HX @ A_HX.T + lambda_**2*(A_HUhat @ A_HUhat.T) + lambda_**2*(A_HU @ A_HU.T) 

    S_ZHZ = A_X @ A_HX.T + lambda_**2*(A_Uhat @ A_HUhat.T) + lambda_**2*(A_U @ A_HU.T)

##########################################
## apply localization to total variate
##########################################

  if rho_Z != None:

    # correct 
    HT = (H.T).to_sparse_csc() 
    H = H.to_sparse_csr()

    S_ZHZ = (rho_Z @ HT)*S_ZHZ
    S_HZ = (H @ rho_Z @ HT)*S_HZ

    S_ZHZ = S_ZHZ.to_dense()
    S_HZ = S_HZ.to_dense()


  R = s_obs**2 * torch.eye(m, **arr_kwargs)


############################################
  # Deterministic EnKF
############################################

  if enkf_type == 'det':

    K = S_ZHZ @ torch.linalg.inv(S_HZ + R)

    A_X = A_X - 0.5*K @ A_HX
    A_Uhat = A_Uhat - 0.5*K @ A_HUhat
    A_U = A_U - 0.5*K @ A_HU

    mu_Z = mu_Z - K @ (mu_HZ - y)

###########################################
  ## Perturbed obs EnKF

  elif enkf_type == 'pert':

    K = S_ZHZ @ torch.linalg.inv(S_HZ + R)

    S_eta_X  = R
    S_eta_U = R

    eta_X = torch.randn(m,N_X, **arr_kwargs)
    eta_U = torch.randn(m,N_U, **arr_kwargs) 

    E_eta_X = S_eta_X**(1/2) @ eta_X
    E_eta_Uhat = E_eta_X
    E_eta_U = S_eta_U**(1/2) @ eta_U


    if N_X > 1:
      A_eta_X = (E_eta_X - torch.mean(E_eta_X, axis = -1 )[:,None] ) / np.sqrt(N_X - 1)
      A_eta_Uhat = (E_eta_Uhat - torch.mean(E_eta_Uhat, axis = -1)[:,None] ) / np.sqrt(N_X - 1)
    else:
      A_eta_X = torch.zeros((m,1) )
      A_eta_Uhat = torch.zeros((m,1) )
    
    A_eta_U = (E_eta_U - torch.mean(E_eta_U, axis = -1)[:,None] ) / np.sqrt(N_U - 1)

    A_X = A_X - K @ (A_HX - A_eta_X)
    A_Uhat = A_Uhat - K @ (A_HUhat - A_eta_Uhat)  
    A_U = A_U - K @ (A_HU - A_eta_U)

    mu_Z = mu_Z - K @ (mu_HZ - y)

##################################
  ### apply inflation
################################
  A_X = alpha_X * A_X
  A_Uhat = alpha_Uhat * A_Uhat
  A_U = alpha_U * A_U

  ## force A_Uhat = A_X to ensure high correlation between X and Uhat
  if adjust_corr == True:
    A_Uhat = A_X


## recenter means
  
  if recenter == 'all':  
    X = mu_Z[:,None] + np.sqrt(N_X-1)*A_X
    Uhat = mu_Z[:,None] + np.sqrt(N_X-1)*A_Uhat
    U = mu_Z[:,None] + np.sqrt(N_U-1)*A_U

  elif recenter == 'control':
    mu_X = mu_X - K @ (mu_HX - y)
    mu_Uhat = mu_Uhat - K @ (mu_HUhat - y)
    mu_U = mu_U - K @ (mu_HU - y)

    X = mu_X[:,None] + np.sqrt(N_X-1)*A_X
    Uhat = mu_U[:,None] + np.sqrt(N_X-1)*A_Uhat
    U = mu_U[:,None] + np.sqrt(N_U-1)*A_U

  elif recenter == 'none':
    mu_X = mu_X - K @ (mu_HX - y)
    mu_Uhat = mu_Uhat - K @ (mu_HUhat - y)
    mu_U = mu_U - K @ (mu_HU - y)

    X = mu_X[:,None] + np.sqrt(N_X-1)*A_X
    Uhat = mu_Uhat[:,None] + np.sqrt(N_X-1)*A_Uhat
    U = mu_U[:,None] + np.sqrt(N_U-1)*A_U

  elif recenter == 'ML':
    mu_X = mu_X - K @ (mu_HX - y)
    mu_Uhat = mu_Uhat - K @ (mu_HUhat - y)
    mu_U = mu_U - K @ (mu_HU - y)

    X = mu_X[:,None] + np.sqrt(N_X-1)*A_X
    Uhat = mu_Z[:,None] + np.sqrt(N_X-1)*A_Uhat
    U = mu_Z[:,None] + np.sqrt(N_U-1)*A_U


###############################################
  ## add back the boundary
################################################  

  if remove_bound == True:
 
    X = add_boundary(X,psi_true_k)
    Uhat = add_boundary(Uhat,psi_true_k)
    U = add_boundary(U,psi_true_k)

  return X,Uhat,U

#################################################
## MF-EnKF run
#################################################


def MF_EnKF_run(lambda_,N_X,N_U,s_obs,rho_X, alpha_X, alpha_Uhat, alpha_U, k_obs,y_obs,
                X_init, K_da, psi_true, mode, H,r_low,s_mod_X,s_mod_U,recenter,adjust_corr,
                control,enkf_type,recenter_forecast,dt,nl,nx,ny,remove_bound):

###########################
  ### Storage
###########################

  n = X_init.shape[0]

  ## store RMSE values
  rmse_Z = np.ones(K_da)*np.inf 

  Z = torch.zeros((K_da,n), **arr_kwargs)

  X_store = torch.zeros((K_da, n), **arr_kwargs)

  P_X = np.zeros(K_da)
  P_Uhat = np.zeros(K_da)
  P_U = np.zeros(K_da)

  P_Z = np.zeros(K_da)

#############################################
## Initialization
#############################################

  X = X_init[:,:N_X] 
  Uhat = X_init[:,:N_X] 
  U = X_init[:,N_X:N_X+N_U] 

  ## initial means
  mu_X = torch.mean(X, axis = -1)
  mu_Uhat = torch.mean(Uhat, axis = -1)
  mu_U = torch.mean(U, axis = -1)

  X_store[0] = mu_X

  mu_Z = mu_X - lambda_*(mu_Uhat - mu_U)
  Z[0] = mu_Z

  P_X[0] = torch.mean(torch.sqrt(torch.var(X, dim = -1) )).item()
  P_Uhat[0] = torch.mean(torch.sqrt(torch.var(Uhat,dim=-1) )).item()
  P_U[0] = torch.mean(torch.sqrt(torch.var(U,dim=-1)) ).item()

  ## calculate covariance for Z

  if N_X > 1:
    A_X = 1/np.sqrt(N_X-1) * (X - mu_X[:,None] ).type(dtype)
    A_Uhat = 1/np.sqrt(N_X-1) * (Uhat - mu_Uhat[:,None] ).type(dtype)
    A_U = 1/np.sqrt(N_U-1) * (U - mu_U[:,None]).type(dtype)
    S_Z = A_X @ A_X.T + lambda_**2*(A_Uhat @ A_Uhat.T) + lambda_**2*(A_U @ A_U.T) - lambda_*(A_X @ A_Uhat.T) - lambda_*(A_Uhat @ A_X.T) 
    P_Z[0] = torch.sqrt((torch.sum(torch.diag(S_Z)) / n ))

  else:
    P_Z[0] = 0

########################################################
## Loop over time
########################################################


  for k in range(1,K_da):

  ##################################
  ## forecast step
  ##################################

    psi_true_k = psi_true[k,:].unsqueeze(0)
    X_b, Uhat_b,U_b = MFEnKF_forecast(X,Uhat, U, mode,r_low, dt,k,psi_true_k)

  ##################################
  ## analysis step
  ##################################

    if k % k_obs == 0:
      psi_true_k = psi_true[k,:].unsqueeze(0)
      X, Uhat, U = MFEnKF_analysis(X_b,Uhat_b,U_b,y_obs[k,:],H,s_obs, k, lambda_, rho_X, alpha_X, alpha_Uhat, alpha_U,
                                                        recenter, adjust_corr,control,enkf_type,psi_true_k,remove_bound)

    else:
      X, Uhat, U = X_b.to(device), Uhat_b.to(device), U_b.to(device)

  
  ## means
    mu_X = torch.mean(X, axis = -1)
    mu_Uhat = torch.mean(Uhat, axis = -1)
    mu_U = torch.mean(U, axis = -1)
    mu_Z = mu_X - lambda_*(mu_Uhat - mu_U)

    X_store[k] = mu_X
    Z[k] = mu_Z

    # add [k] to store at every time step
    P_X[k] = torch.mean(torch.sqrt(torch.var(X, dim = -1) )).item()
    P_Uhat[k] = torch.mean(torch.sqrt(torch.var(Uhat,dim=-1) )).item()
    P_U[k] = torch.mean(torch.sqrt(torch.var(U,dim=-1)) ).item()
    
    if k == K_da - 1:
      if N_X > 1:
        A_X = 1/np.sqrt(N_X-1) * (X - mu_X[:,None] ).type(dtype)
        A_Uhat = 1/np.sqrt(N_X-1) * (Uhat - mu_Uhat[:,None] ).type(dtype)
        A_U = 1/np.sqrt(N_U-1) * (U - mu_U[:,None]).type(dtype)
        S_Z = A_X @ A_X.T + lambda_**2*(A_Uhat @ A_Uhat.T) + lambda_**2*(A_U @ A_U.T) - lambda_*(A_X @ A_Uhat.T) - lambda_*(A_Uhat @ A_X.T) 

        P_Z[k] = torch.sqrt((torch.sum(torch.diag(S_Z)) / n ))
  
      else:
        P_Z[k] = 0

    ## Calculate RMSE

    burn_in = int(0.10*K_da)

    if k > burn_in:

      a1 = Z[burn_in:k].squeeze()
      a2 = psi_true[burn_in:k,0,:,:].reshape(-1,nx*ny).squeeze()

      a1 = a1.to(device)
      a2 = a2.to(device)

      rmse_Z[k] = torch.sqrt(torch.mean((a1-a2)**2) )

  return rmse_Z, Z, P_X, P_Uhat, P_U, P_Z, mu_X, mu_Uhat, mu_U, X, Uhat, U 






