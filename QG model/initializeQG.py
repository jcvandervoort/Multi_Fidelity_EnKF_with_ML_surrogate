
############################################################
## Generate true initial condition and initial ensemble
############################################################

## output: 2000 fields collected from a long model run of length T = 5e5

# K_long = int(5e5)
# n_snaps = 2000

import torch
from QG import compute_q_over_f0_from_p, compute_psi_from_p, step_RK4, compute_CFL


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

arr_kwargs = {'dtype':dtype, 'device':device}


def generate_snapshots(p,K_long, n_snaps,dt):

    psi_long = []   # store snapshots of stream function

    q = compute_q_over_f0_from_p(p)

    psi = compute_psi_from_p(p)

    for k in range(1,K_long):
        p,q = step_RK4(p,q,dt)
        psi = compute_psi_from_p(p)

        if k % (int(K_long/n_snaps)) == 0:
            psi_long.append(psi)

        if k%1000 == 0:     
            print('Progress: ' + str(k/K_long*100) + ' %')
            CFL = compute_CFL(p)
            print('CFL number is: ' + str(CFL))

            if torch.isnan(p).any():
                raise ValueError(f'Stopping, NAN number in p at iteration {k}.')
 
    print('> Done computing long run')
    print('Model variability psi (std.): ' + str(psi.std().item() ) )

        

    return psi_long


import random as rnd
import numpy as np

def generate_initial_ensemble(psi_long, N_X,seed, device):

    n_snaps = len(psi_long)
    rnd_list = range(n_snaps)
    nx = psi_long[0].shape[-2]
    ny = psi_long[0].shape[-1]
    n = nx*ny

    rnd.seed(seed)
    rnd_ind = rnd.sample(rnd_list, N_X)

    dtype = torch.float64
    arr_kwargs = {'dtype':dtype, 'device':device}


    X_init = torch.zeros((N_X,n), **arr_kwargs )
    
    for count,k in enumerate(rnd_ind):
        X_init[count,:] = psi_long[k][:,0,:,:].reshape(n)


    return X_init.T


def generate_initial_condition(psi_long):

    n_snaps = len(psi_long)
    rnd_list = range(n_snaps+1)

    rnd.seed(0)
    rnd_ind = rnd.sample(rnd_list, n_snaps)

    ## initial condition for truth run
    psi_true_init = psi_long[rnd_ind[-1]]

    psi_true_init = psi_long[-1]

    return psi_true_init



