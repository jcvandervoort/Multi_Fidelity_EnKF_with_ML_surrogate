## ML surrogate for QG model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

arr_kwargs = {'dtype':dtype, 'device':device}

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


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'), # add 'replicate' here
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)
    


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p
    


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
    

num_channels = 32

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_convolution_1 = DownSample(1, num_channels)                       # 32 channels
        self.down_convolution_2 = DownSample(num_channels, num_channels*2)          # 32, 64

        self.bottle_neck = DoubleConv(num_channels*2,num_channels*4)                # 64, 128

        self.up_convolution_1 = UpSample(num_channels*4, num_channels*2)            # 128, 64
        self.up_convolution_2 = UpSample(num_channels*2, num_channels)              # 64, 32

        self.out = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=1) 

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)  ## down-sample
        down_2, p2 = self.down_convolution_2(p1) ## down-sample

        b = self.bottle_neck(p2)

        up_1 = self.up_convolution_1(b, down_2)     ## up-sample and skip connection
        up_2 = self.up_convolution_2(up_1, down_1)  ## up-sample and skip connection

        out = self.out(up_2)
        return out



#########################################


## load model weights

model = UNet()

# 4 steps
W = torch.load('/home/jeffreyvanderv/QG_model/QGmodel_weights_Unet', weights_only = True)
model.load_state_dict(W)


model = model.to(device)




def M_surr(U,psi_true_k):

  # input: ensemble matrix X_k of size n x N_X
  # output: ensemble matrix X_{k+1} of size n x N_X

  N_X = U.shape[-1]
  nx = int(np.sqrt(U.shape[0]))

  # reshape input
  psi_old = reshape_X_to_psi(U)
  psi_in = psi_old[:,0,:,:].unsqueeze(1)

  ## propagate forward
  psi_in = psi_in.type(torch.float32)
  
  model.eval()

  with torch.no_grad():
    out = model(psi_in)
    out = out.type(torch.float64)

  nl = 2
  dpsi = torch.zeros((N_X,nl,nx,nx), device = device, dtype = dtype)

  dpsi[:,0,:,:] = out.squeeze(1)


# reshape output
  nl = 2
  psi_new = torch.zeros((N_X,nl,nx,nx), **arr_kwargs)
  psi_new[:,0,:,:] = psi_old[:,0,:,:] + dpsi[:,0,:,:]

## keep boundary fixed

  psi_new[...,0,1:-1] = psi_true_k[...,0,1:-1]
  psi_new[...,-1,1:-1] = psi_true_k[...,-1,1:-1]
  psi_new[...,0] = psi_true_k[...,0]
  psi_new[...,-1] = psi_true_k[...,-1]


  U_new = reshape_psi_to_X(psi_new).to(device)

  return U_new
