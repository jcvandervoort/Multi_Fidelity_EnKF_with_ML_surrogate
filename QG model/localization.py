## Localization


import torch
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

arr_kwargs = {'dtype':dtype, 'device':device}


def loc_func(dist,radius,name):


    ## Gaspari and Cohn localization
    if name == 'GC':
       
        R = np.sqrt(10/3)*radius  # c = 0.5*r or choose c = np.sqrt(10/3) r to match Gaussian case

        s = dist / R

        if 0 <= dist <= R:
            coeff = 1 - 5/3*s**2 + 5/8*s**3 + 1/2*s**4 - 1/4*s**5

        elif R <= dist <= 2*R:
            coeff = -2/3*1/s + 4 -5*s + 5/3*s**2 + 5/8*s**3 - 1/2*s**4 + 1/12*s**5

        else:
            coeff = 0

    ## Gaussian localization
    elif name == 'Gauss':
       R = radius
       s = dist / R
       coeff = np.exp(-0.5*s**2)

    ## Step function localization
    elif name == 'Step':
       R = radius
       if dist <= R:
          coeff = 1
       else:
          coeff = 0

    return coeff


## calculate distance in grid space

def dist_euler(I,J,nx):

  # convert to grid coordinates
  j = I % nx
  i = (I-j)/nx


  l = J % nx
  k = (J-l)/nx

  # calculate distance in grid space
  d = np.sqrt( (i-k)**2 + (j-l)**2)

  return d

def dist_periodic(i,j,n):
   
   return np.min( [np.abs(i-j), np.abs(n+i-j), np.abs(n+j-i) ] )



def create_loc_mat(r,n,name,mode):
  
  ## inputs:
  ## r          localization radius
  ## n          state dimension
  ## name       can be one of {'GC', 'Gauss', 'Step'}
  ## mode        periodic or Euler distance {'periodic' or 'euler'}

  ## outputs:
  ## rho        localization matrix (sparse representation)

  print('Constructing localization matrix...')

  row = []
  col = []
  data = []

  for i in range(n):


    if i == 0:

      V = np.zeros(n)

      for j in range(n):
        if mode == 'periodic': 
           coeff = loc_func(dist_periodic(i,j,n), r,name)
        elif mode == 'euler':
           coeff = loc_func(dist_euler(i,j,int(np.sqrt(n))), r,name)

        if coeff > 0.0:
          V[j] = coeff
          row.append(i)
          col.append(j)
          data.append(coeff)

          if j == i:
            data[-1] *= 0.5

      cut_off = np.max(np.where(V>0))


    else:
      low = i
      upp = np.min([i+cut_off+1,n])
      for j in range( low,upp):
        if mode == 'periodic':
           coeff = loc_func(dist_periodic(i,j,n), r,name)
        elif mode == 'euler':
           coeff = loc_func(dist_euler(i,j,int(np.sqrt(n))), r,name)

        if coeff > 0.0:
          row.append(i)
          col.append(j)
          data.append(coeff)

          if j == i:
            data[-1] *= 0.5

  rho = torch.sparse_coo_tensor([row, col],data, (n,n))

  rho = rho + rho.T

  print('Done!')


  return rho.to_sparse_csr().to(device)



