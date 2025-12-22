
############################################################
## Generate true initial condition and initial ensemble
############################################################


import torch
from Lorenz2005 import M_phys


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

arr_kwargs = {'dtype':dtype, 'device':device}


def generate_snapshots(n,K_long, n_snaps):

    x_long = []   # store snapshots of stream function

    np.random.seed(0)
    x = np.random.uniform(0,1,n)


    for k in range(1,K_long):
        x = M_phys(x)

        if k % (K_long/n_snaps) == 0:
            x_long.append(x)

        if k%1000 == 0:
            print('Progress: ' + str(k/K_long*100) + ' %')
           # CFL = compute_CFL(p)
           # print('CFL number is: ' + str(CFL))

            if np.isnan(x).any():
                raise ValueError(f'Stopping, NAN number in p at iteration {k}.')
 
    print('> Done computing long run')
    print('Model variability psi (std.): ' + str(x.std().item() ) )

        

    return x_long


import random as rnd
import numpy as np

def generate_initial_ensemble(x_long, N_X,seed):

    n_snaps = len(x_long)
    rnd_list = range(n_snaps)
    n = x_long[0].shape[-1]

    rnd.seed(seed)
    rnd_ind = rnd.sample(rnd_list, N_X)

    #print('Random indices are: ' + str(rnd_ind))

    X_init = np.zeros((N_X,n) )
    
    for count,k in enumerate(rnd_ind):
        X_init[count,:] = x_long[k].reshape(n)


    return X_init.T


def generate_initial_condition(x_long):

    n_snaps = len(x_long)
    rnd_list = range(n_snaps+1)

    rnd.seed(0)
    rnd_ind = rnd.sample(rnd_list, 2000) # 2000 so there is no overlap with initial ensemble

    ## initial condition for truth run
    x_true_init = x_long[rnd_ind[-1]]

    return x_true_init



