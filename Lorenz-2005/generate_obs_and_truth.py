
import numpy as np
import torch
import torch.nn.functional as F


from Lorenz2005 import M_phys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

arr_kwargs = {'dtype':dtype, 'device':device}



####################################################
## Generate observation operator
####################################################


def create_obs_op(n,choice,m,delta_obs,k):

#######################################################
## Regular grid obs
#######################################################

  if choice == 'grid':

    ## create observations

    obs_loc = np.zeros(n)

    delta_obs = int(n/m)

    for i in range(n):
        if (i%delta_obs==0):  # observe every 10th row and column
          obs_loc[i] = 1

    m = len(np.where(obs_loc==1)[0])

    ## create observation matrix

    H = np.zeros((m,n))

    count = 0
    for i in range(n):
      if obs_loc[i] == 1:
        H[count,i] = 1
        count += 1

    return H,obs_loc


####################################
## Randomly placed obs
####################################

  elif choice == 'random':
    
    H = np.zeros((m,n))

    np.random.seed(k)

    ## (!!) Important to sort obs, otherwise the order is different
    obs_loc = np.sort(np.random.choice(n, size = m, replace = False))

    count = 0
    for i in range(n):
        if i in obs_loc:
            H[count,i] = 1
            count += 1

    return H,obs_loc

#########################################################
## Generate truth and obs
#########################################################

def generate_truth(x0,T, T_spinup):

  n = x0.shape[-1]

  for t in range(T_spinup):
    x0 = M_phys(x0)

 # K_n = int(n/30)

  #num_years = 1
  #K = 12*30*8*num_years 

  x_true = np.zeros((T,n) )
  x_true[0,:] = x0

  for k in range(1,T):
    x_true[k,:] = M_phys(x_true[k-1,:])

  print('> Done computing ref solution')
  print('State dimension: ' + str(n) )
  print('Interval length: ' + str(T) )
  print('Model variability psi (std.): ' + str(x_true.std().item() ) )

  return x_true



def generate_obs(x_true,s_obs,H,seed):

  ## set seed
  np.random.seed(seed)

  m = H.shape[0]
  n = H.shape[1]
  K = x_true.shape[0]

  y_obs = np.zeros((K,m))

  for k in range(1,K):
    y_obs[k,:] = H @ (x_true[k,:].reshape(n,)) + s_obs*np.random.randn(m,)

  return y_obs











