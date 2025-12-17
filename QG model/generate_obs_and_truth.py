
import numpy as np
import torch
import torch.nn.functional as F

device = 'cuda'
dtype = torch.float64

from QG import step_RK4, compute_q_over_f0_from_p, compute_psi_from_p, compute_p_from_psi


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

arr_kwargs = {'dtype':dtype, 'device':device}


####################################################
## Generate observation operator
####################################################


def create_obs_op(nx,ny,choice,m,delta_obs,k):

## inputs
# nx:           horizontal dimension
# m:            number of obs (None if not known a priori)
# choice:       one of {'diagonal', 'satellite', 'grid' or 'random'}
# k:            time index (for time dependent obs)
# delta_obs:    distance between obs / satellite tracks

## outputs
# H:            observation matrix
# obs_loc:      vector containing observation locations


###########################################
## Diagonal obs
###########################################

  if choice == 'diagonal':

    #nx = nx-2 # remove boundary points
    obs_loc = np.zeros((nx,ny))
    n = nx*ny

    off_set = -1

    for i in range(nx):
      if i % 1 == 0:
        off_set = off_set + 1
        for j in range(ny):
          if (j%delta_obs==0) :
      
            obs_loc[i,(j+off_set)%(ny-1)] = 1


    m = len(np.where(obs_loc==1)[0])

    obs_loc = obs_loc.flatten()

    ## create observation matrix
    H = np.zeros((m,n))

    count = 0
    for i in range(n):
      if obs_loc[i] == 1:
        H[count,i] = 1
        count += 1

    return torch.tensor(H, **arr_kwargs), obs_loc
  

########################################
## Satellite obs
########################################


  if choice == 'satellite':
    obs_loc = np.zeros((nx,ny))
    n = nx*ny

    off_set = -1

    for i in range(nx):
      if i % 3 == 0:
        off_set = off_set + 1
        for j in range(ny):
          if (j%delta_obs==0) :
      
            obs_loc[i,(j+off_set)%(ny-1)] = 1


    m = len(np.where(obs_loc==1)[0])

    obs_loc = obs_loc.flatten()

    ## create observation matrix
    H = np.zeros((m,n))

    count = 0
    for i in range(n):
      if obs_loc[i] == 1:
        H[count,i] = 1
        count += 1

    return torch.tensor(H, **arr_kwargs), obs_loc


#######################################################
## Regular grid obs
#######################################################

  if choice == 'grid':

    n = nx*ny

    ## create observations

    obs_loc = np.zeros((nx,ny) )

    #delta_obs = int(np.sqrt(n/m))

    for i in range(nx):
      for j in range(ny):
        if (i%delta_obs==0) & (j%delta_obs==0):  # observe every 10th row and column
          obs_loc[i,j] = 1

    m = len(np.where(obs_loc==1)[0])

    obs_loc = obs_loc.flatten()

    ## create observation matrix

    H = np.zeros((m,n))

    count = 0
    for i in range(n):
      if obs_loc[i] == 1:
        H[count,i] = 1
        count += 1

    return torch.tensor(H, **arr_kwargs), obs_loc


####################################
## Randomly placed obs
####################################

  if choice == 'random':

    n = nx*ny       ## state dimension
    
    H = np.zeros((m,n))

    np.random.seed(k)

    one_ind = np.sort(np.random.choice(nx*ny, size = m, replace = False))

    obs_loc = np.zeros(n)
    obs_loc[one_ind] = 1

    count = 0
    for i in range(n):
        if obs_loc[i] == 1:
            H[count,i] = 1
            count += 1

    return torch.tensor(H, **arr_kwargs), obs_loc
  

#########################################################
## Generate truth and obs
#########################################################

def generate_truth(psi0,K,dt,device):

  nl = psi0.shape[-3]
  nx = psi0.shape[-2]
  ny = psi0.shape[-1]
  n = nl*nx*ny

  p = compute_p_from_psi(psi0)
  q = compute_q_over_f0_from_p(p)

  dtype = torch.float64
  arr_kwargs = {'dtype':dtype, 'device':device}

  psi_true = torch.zeros((K,nl,nx,ny), **arr_kwargs)
  #psi_true[0,:] = psi0[:,0,:,:]


  for k in range(1,K):
    p,q = step_RK4(p,q,dt)
    psi = compute_psi_from_p(p)

    psi_true[k,:] = psi[:,:,:,:]
  
    if k % 10 == 0 and torch.isnan(p).any():
        raise ValueError(f'Stopping, NAN number in p at iteration {k}.')

    if k % (int(K/10)) == 0:
      prog = k/K*100
      print('Progress: ' + str(int(prog)) + '%')


  print('> Done computing ref solution')
  print('State dimension: ' + str(n) )
  print('Interval length: ' + str(K) )
  print('Model variability psi (std.): ' + str(psi_true.std().item() ) )

  return psi_true



def generate_obs(psi_true,s_obs,H,seed):

  ## set seed
  torch.manual_seed(seed=seed)

  m = H.shape[0]
  n = H.shape[1]
  K = psi_true.shape[0]

  y_obs = torch.zeros((K,m), **arr_kwargs)

  for k in range(1,K):
    y_obs[k,:] = H @ (psi_true[k,0,:,:].reshape(n,).to(device)) + s_obs*torch.randn((m,), **arr_kwargs)

  return y_obs











