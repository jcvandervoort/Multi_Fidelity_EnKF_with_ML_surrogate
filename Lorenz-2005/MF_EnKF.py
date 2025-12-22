### MF-EnKF

import numpy as np

from Lorenz2005 import M_phys, M_low

from Lorenz2005_surrogate import M_surr

from sklearn.metrics import mean_squared_error

###########################################
## MF-EnKF forecast step
############################################

def MFEnKF_forecast(X,Uhat, U, mode, r_low):

  """
  Forecast step of Multi-Fidelity EnKF
  """

  ## inputs: 
  # X: ensemble of principal variates (shape: n x N_X)
  # Uhat: ensemble of control variates (shape: n x N_U)
  # U: ensemble of auxiliary variates (shape: n x N_U)

  X = M_phys(X)


  if mode == 'LR':
    
    Uhat = M_low(Uhat,r_low)
    U = M_low(U,r_low)

  elif mode == 'ML':

    Uhat = M_surr(Uhat)
    U = M_surr(U)


  return X,Uhat,U


########################################
## MF-EnKF analysis step
########################################

def MFEnKF_analysis(X,Uhat,U,y,H,s_obs, k, lambda_, rho_Z, alpha_X,alpha_Uhat, alpha_U,recenter, adjust_corr,control,enkf_type,loc,pert_option):

  """
  Analysis step of Multi-Fidelity EnKF
  """

##############################################
  ## calculate prior characteristics
##############################################

  N_X = X.shape[-1]
  n = X.shape[0]
  N_U = U.shape[-1]
  m = H.shape[0]

  ## 1) Principal variate
  mu_X = np.mean(X, axis = -1)
  HX = H @ X
  mu_HX = np.mean(HX, axis = -1)
  
  if N_X > 1:
    A_X = 1/np.sqrt(N_X-1) * (X - mu_X[:,None] )
    A_HX = 1/np.sqrt(N_X-1) * (HX - mu_HX[:,None] )
  else:
    A_X = np.zeros((n,1) )
    A_HX = np.zeros((m,1) )

  ## 2) Control variate

  ## enable/disable control variate
  if control == False:
    Uhat = X

  mu_Uhat = np.mean(Uhat, axis = -1)
  HUhat = H @ Uhat
  mu_HUhat = np.mean(HUhat, axis = -1)

  if N_X > 1:
    A_Uhat = 1/np.sqrt(N_X-1) * (Uhat - mu_Uhat[:,None] )
    A_HUhat = 1/np.sqrt(N_X-1) * (HUhat - mu_HUhat[:,None] )
  else:
    A_Uhat = np.zeros((n,1) )
    A_HUhat = np.zeros((m,1) )


  ## 3) Ancillary variate
  mu_U = np.mean(U, axis = -1)
  A_U = 1/np.sqrt(N_U-1) * (U - mu_U[:,None] )

  HU = H @ U
  mu_HU = np.mean(HU, axis = -1)
  A_HU = 1/np.sqrt(N_U-1) * (HU - mu_HU[:,None] )


######################################
  ## Total variate characteristics
######################################

  mu_Z = mu_X - lambda_*(mu_Uhat - mu_U)
  mu_HZ = mu_HX - lambda_*(mu_HUhat - mu_HU)

  if N_X > 1:
    S_HZ = A_HX @ A_HX.T + lambda_**2*(A_HUhat @ A_HUhat.T) + lambda_**2*(A_HU @ A_HU.T) - lambda_*(A_HX @ A_HUhat.T) - lambda_*(A_HUhat @ A_HX.T) 

    S_ZHZ = A_X @ A_HX.T + lambda_**2*(A_Uhat @ A_HUhat.T) + lambda_**2*(A_U @ A_HU.T) - lambda_*(A_X @ A_HUhat.T) - lambda_*(A_Uhat @ A_HX.T)


  #  S_HZ = H @ S_ZHZ

  else:
    S_HZ = A_HX @ A_HX.T + lambda_**2*(A_HUhat @ A_HUhat.T) + lambda_**2*(A_HU @ A_HU.T) 

    S_ZHZ = A_X @ A_HX.T + lambda_**2*(A_Uhat @ A_HUhat.T) + lambda_**2*(A_U @ A_HU.T)

##########################################
## apply localization to total variate
##########################################

 
  if loc == 'yes':

    S_ZHZ = (rho_Z @ (H.T)) * S_ZHZ

    S_HZ = (H @ rho_Z @ (H.T)) * S_HZ


  R = s_obs**2 * np.eye(m)


############################################
  # Deterministic EnKF
############################################

  if enkf_type == 'det':
    S_eta_Z = R
    K = S_ZHZ @ np.linalg.inv(S_HZ + S_eta_Z)

    A_X = A_X - 0.5*K @ A_HX
    A_Uhat = A_Uhat - 0.5*K @ A_HUhat
    A_U = A_U - 0.5*K @ A_HU

    mu_Z = mu_Z - K @ (mu_HZ - y)


###########################################
  ## Perturbed obs EnKF
############################################

  elif enkf_type == 'pert':

    S_eta_Z = R
    K = S_ZHZ @ np.linalg.inv(S_HZ + S_eta_Z)

    # use fixed seed for reproducibility
    np.random.seed(k)

    ## option 1: total variate consistency
    if pert_option == 1:
      S_eta_X  = R
      S_eta_U = (2-lambda_) / lambda_ * R

      eta_X = np.random.randn(m,N_X)
      eta_U = np.random.randn(m,N_U) 

      E_eta_X = S_eta_X**(1/2) @ eta_X
      E_eta_Uhat = E_eta_X
      E_eta_U = S_eta_U**(1/2) @ eta_U

    ## option 2: control space consistency
    elif pert_option == 2:
      S_eta_X = R
      S_eta_Uhat = 1/lambda_**2 * R
      S_eta_U = 1/lambda_**2*R #S_eta_Uhat

      eta_X = np.random.randn(m,N_X)
      eta_Uhat = np.random.randn(m,N_X)
      eta_U = np.random.randn(m,N_U) 

      E_eta_X = S_eta_X**(1/2) @ eta_X
      E_eta_Uhat = S_eta_Uhat**(1/2) @ eta_Uhat
      E_eta_U = S_eta_U**(1/2) @ eta_U

    ## option 3: simplest way: choose all equal to R
    elif pert_option == 3:
      S_eta_X = R
      S_eta_Uhat = R
      S_eta_U = R

      eta_X = np.random.randn(m,N_X)
      eta_U = np.random.randn(m,N_U) 

      E_eta_X = S_eta_X**(1/2) @ eta_X
      E_eta_Uhat = E_eta_X
      E_eta_U = S_eta_U**(1/2) @ eta_U


    if N_X > 1:
      A_eta_X = (E_eta_X - np.mean(E_eta_X, axis = -1)[:,None] ) / np.sqrt(N_X - 1)
      A_eta_Uhat = (E_eta_Uhat - np.mean(E_eta_Uhat, axis = -1)[:,None] ) / np.sqrt(N_X - 1)
    else:
      A_eta_X = np.zeros((m,1) )
      A_eta_Uhat = np.zeros((m,1) )
    
    A_eta_U = (E_eta_U - np.mean(E_eta_U, axis = -1)[:,None] ) / np.sqrt(N_U - 1)

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

####################################
## Heuristic ensemble corrections
#####################################

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

  return X,Uhat,U


#################################################
## MF-EnKF run
#################################################


def MF_EnKF_run(lambda_,N_X,N_U,s_obs,rho_X, alpha_X, alpha_Uhat, alpha_U, k_obs,y_obs,
                X_init, K_da, psi_true, mode, H,r_low,s_mod_X,s_mod_U,recenter,adjust_corr,
                control,enkf_type,recenter_forecast,loc,pert_option):

###########################
  ### Storage
###########################

  n = X_init.shape[0]

  ## store RMSE values
  rmse_Z = np.ones(K_da)*np.inf 


  ## store ensemble means
  X_store = np.zeros((K_da,n) )
  Uhat_store = np.zeros((K_da,n) )
  U_store = np.zeros((K_da,n) )

  Z = np.zeros((K_da,n))

  ## store variances
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
  mu_X = np.mean(X, axis = -1)
  mu_Uhat = np.mean(Uhat, axis = -1)
  mu_U = np.mean(U, axis = -1)

  mu_Z = mu_X - lambda_*(mu_Uhat - mu_U)

  X_store[0,:] = mu_X
  Uhat_store[0,:] = mu_Uhat
  U_store[0,:] = mu_U
  Z[0,:] = mu_Z

  ## initial variances
  P_X[0] = np.sqrt(np.var(X, axis = -1)).mean()
  P_Uhat[0] = np.sqrt(np.var(Uhat, axis = -1)).mean()
  P_U[0] = np.sqrt(np.var(U, axis = -1)).mean()
 
  if N_X > 1:
    A_X = 1/np.sqrt(N_X-1) * (X - mu_X[:,None] )
    A_Uhat = 1/np.sqrt(N_X-1) * (Uhat - mu_Uhat[:,None] )
    A_U = 1/np.sqrt(N_U-1) * (U - mu_U[:,None])
    S_Z = A_X @ A_X.T + lambda_**2*(A_Uhat @ A_Uhat.T) + lambda_**2*(A_U @ A_U.T) - lambda_*(A_X @ A_Uhat.T) - lambda_*(A_Uhat @ A_X.T) 
    P_Z[0] = np.sqrt((np.sum(np.diag(S_Z)) / n ))

  else:
    P_Z[0] = 0



  ## calculate rmse
  a1 = Z[0]
  a2 = psi_true[0]

  rmse_Z[0] = np.sqrt( mean_squared_error(a1, a2 ) )


########################################################
## Loop over time
########################################################


  for k in range(1,K_da):


  ##################################
  ## forecast step
  ##################################

    X_b, Uhat_b,U_b = MFEnKF_forecast(X, Uhat, U,mode,r_low)

  ##################################
  ## analysis step
  ##################################

    if k % k_obs == 0:
      
      X, Uhat, U = MFEnKF_analysis(X_b,Uhat_b,U_b,y_obs[k,:],H,s_obs, k, lambda_, rho_X, alpha_X, alpha_Uhat, alpha_U,
                                                        recenter, adjust_corr,control,enkf_type,loc,pert_option)

    else:
      X, Uhat, U = X_b, Uhat_b, U_b


    #E_X_mfenkf[k,:,:] = X
  
  ## means
    mu_X = np.mean(X , axis = -1)
    mu_Uhat = np.mean(Uhat , axis = -1)
    mu_U = np.mean(U , axis = -1)
    mu_Z = mu_X - lambda_*(mu_Uhat - mu_U)

    X_store[k,:] = mu_X
    Uhat_store[k,:] = mu_Uhat
    U_store[k,:] = mu_U
    Z[k,:] = mu_Z

    P_X[k] = np.sqrt(np.var(X, axis = -1)).mean()
    P_Uhat[k] = np.sqrt(np.var(Uhat, axis = -1)).mean()
    P_U[k] = np.sqrt(np.var(U, axis = -1)).mean()


    if k == K_da - 1:
      if N_X > 1:
        A_X = 1/np.sqrt(N_X-1) * (X - mu_X[:,None] )
        A_Uhat = 1/np.sqrt(N_X-1) * (Uhat - mu_Uhat[:,None] )
        A_U = 1/np.sqrt(N_U-1) * (U - mu_U[:,None])
        S_Z = A_X @ A_X.T + lambda_**2*(A_Uhat @ A_Uhat.T) + lambda_**2*(A_U @ A_U.T) - lambda_*(A_X @ A_Uhat.T) - lambda_*(A_Uhat @ A_X.T) 

        P_Z[k] = np.sqrt((np.sum(np.diag(S_Z)) / n ))
  
      else:
        P_Z[k] = 0


    ## Calculate RMSE
    burn_in = int(0.10*K_da)

    if k > burn_in:
      a1 = Z[burn_in:k+1,:]
      a2 = psi_true[burn_in:k+1,:]

      rmse_Z[k] = np.sqrt(mean_squared_error(a1,a2) )


  return rmse_Z , Z , P_X, P_Uhat, P_U, P_Z, mu_X, mu_Uhat, mu_U, X, Uhat, U





