
import numpy as np

from Lorenz2005 import M_phys, M_low
from Lorenz2005_surrogate import M_surr


###########################################
## MF-EnKF forecast step
###########################################


def MF_EnKF_forecast_baseline(X, U, mode, r_low,k,k_obs):

  X = M_phys(X)

  if mode == 'LR':
    
    U = M_low(U,r_low)

  elif mode == 'ML':

    U = M_surr(U)

  return X, U


def MF_EnKF_analysis_baseline(X,U,y,H,s_obs, rho_Z, alpha_Z, enkf_type,recenter, loc,k):

##############################################
  ## calculate prior characteristics
##############################################

  N_X = X.shape[-1]
  n = X.shape[0]

  N_U = U.shape[-1]
  m = H.shape[0]

  mu_X = np.sum(X, axis = -1) / N_X
  mu_U = np.sum(U, axis = -1) / N_U

  if N_U != 1:
    A_U = 1/np.sqrt(N_U-1) * (U - mu_U[:,None] )
  else:
    A_U = np.zeros((n,1) )

###########################################
  ## recenter ML ensemble with HR mean
###########################################

  if recenter == True:
    U = mu_X[:,None] + np.sqrt(N_U-1)*A_U

#############################################
## combine ensemble into one
#############################################

  Z = np.concatenate((X,U), axis = -1)  # shape: (n,N_X+N_U)

######################################
  ## Total variate characteristics
######################################

  mu_Z = np.mean(Z, axis = -1)
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

  if loc == 'yes':
    S_ZHZ = (rho_Z @ H.T) * S_ZHZ
    S_HZ = (H @ rho_Z @ H.T) * S_HZ

  R = s_obs**2 * np.eye(m)


############################################
  # Deterministic EnKF
############################################

  if enkf_type == 'det':
    S_eta_Z = R

    K = S_ZHZ @ np.linalg.inv(S_HZ + S_eta_Z)

    A_Z = A_Z - 0.5*K @ A_HZ

    mu_Z = mu_Z - K @ (mu_HZ - y)


###########################################
  ## Perturbed obs EnKF
###########################################

  elif enkf_type == 'pert':

    S_eta_Z = R
    K = S_ZHZ @ np.linalg.inv(S_HZ + S_eta_Z)

    np.random.seed(k)
    eta_Z = np.random.randn(m,N_Z)

    E_eta_Z = S_eta_Z**(1/2) @ eta_Z

    A_eta_Z = (E_eta_Z - np.mean(E_eta_Z, axis = -1 )[:,None] ) / np.sqrt(N_Z - 1)

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
 
  X = Z[:,:N_X]  # shape: (n,N_X)
  U = Z[:,N_X:] # shape: (n,N_U)


  return X,U

#################################################
## MF-EnKF run
#################################################



def MF_EnKF_run_base(N_X,N_U,s_obs,rho_Z, alpha_Z, k_obs,y_obs,
                Z_init, K_da, psi_true, mode, H, r_low, enkf_type, recenter,loc):


###########################
  ### Storage
###########################

  n = Z_init.shape[0]

  ## store RMSE values
  rmse_Z = np.ones(K_da)*np.inf 
  rmse_X = np.ones(K_da)*np.inf

  mu_Z = np.zeros((K_da,n))

  mu_X = np.zeros((K_da,n))
  mu_U = np.zeros((K_da,n))


  P_X = np.zeros(K_da)
  P_U = np.zeros(K_da)
  P_Z = np.zeros(K_da)

#############################################
## Initialization
#############################################

  ## initial means
  N_Z = N_X + N_U
  mu_Z[0] = np.mean(Z_init[:,:N_Z], axis = -1)

  X = Z_init[:,:N_X]  # shape: (n,N_X)
  U = Z_init[:,N_X:N_X+N_U] # shape: (n,N_U)

  mu_X[0] = np.mean(X, axis = -1)
  mu_U[0] = np.mean(U, axis = -1)

  P_X[0] = np.sqrt(np.var(X, axis =-1).mean())
  P_U[0] = np.sqrt(np.var(U, axis =-1).mean())
  P_Z[0] = np.sqrt(np.var(Z_init[:,:N_X+N_U], axis = -1).mean())

########################################################
## Loop over time
########################################################


  for k in range(1,K_da):


  ##################################
  ## forecast step
  ##################################
 
    X_b, U_b = MF_EnKF_forecast_baseline(X, U, mode, r_low,k,k_obs)

  ##################################
  ## analysis step
  ##################################

    if k % k_obs == 0:
      #print(psi_true_k.shape)
      #try:
      X, U = MF_EnKF_analysis_baseline(X_b,U_b,y_obs[k,:],H,s_obs, rho_Z, alpha_Z, enkf_type,recenter, loc,k)
  
    else:
      X, U = X_b, U_b

  
  ## means
    Z = np.concatenate((X,U), axis = -1)  # shape: (n,N_X+N_U)

    ## either use mean of Z or the mean of X (so excl DL members)
    mu_Z[k] = np.mean(Z, axis = -1)
    mu_X[k] = np.mean(X, axis = -1)
    mu_U[k] = np.mean(U, axis = -1)

    # add [k] to store at every time step
    P_X[k]= np.sqrt(np.var(X, axis = -1).mean())
    P_U[k] = np.sqrt(np.var(U, axis = -1).mean())

    P_Z[k] = np.sqrt(np.var(Z, axis = -1).mean())

    
    ## Calculate RMSE

    burn_in = int(0.10*K_da)

    if k > burn_in:

      a1 = mu_Z[burn_in:k,:].squeeze()
      a2 = psi_true[burn_in:k,:]
      
      rmse_Z[k] = np.sqrt(np.mean((a1-a2)**2))

      a3 = mu_X[burn_in:k,:].squeeze()
      
      rmse_X[k] = np.sqrt(np.mean((a3-a2)**2) )


  return rmse_Z, rmse_X, Z,P_Z, P_X, P_U, X, U







