import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dtype = torch.float32

arr_kwargs = {'dtype':dtype, 'device':device}

## NN architecture


class NN(nn.Module):

  def __init__(self):
    super().__init__()

    K = 32

    self.layer1 = nn.BatchNorm1d(1)

    self.layer2a = nn.Conv1d(1,32, 3*K, padding = 'same', padding_mode = 'circular')

    self.layer2b = nn.Conv1d(1,16, 4*K, padding = 'same', padding_mode = 'circular')

    self.layer2c = nn.Conv1d(1,16, 5*K, padding = 'same', padding_mode = 'circular')

    self.layer3 = nn.Conv1d(32,16,5*K, padding = 'same', padding_mode = 'circular')

    self.layer4 = nn.Conv1d(16, 1, 1)


## Forward pass

  def forward(self,x):

    x = self.layer1(x)

    x1 = F.relu(self.layer2a(x) )
    x2 = F.relu(self.layer2b(x) )
    x3 = F.relu(self.layer2c(x) )

    x1a,x1b = torch.split(x1,split_size_or_sections = 16, dim = 1 )

    x = torch.cat([x2*x1a, x3*x1b], dim = 1)

    x = F.relu(self.layer3(x) )

    x = self.layer4(x)


    return x


    
model = NN()

W = torch.load('/home/jeffreyvanderv/Lorenz2005_model/lorenz05_weights_CNN', weights_only = True)

model.load_state_dict(W)
model = model.to(device)



def M_surr(X):

  # input: X numpy array of shape n by Ne
  # output: numpy array of shape n by Ne

  ## prep data
  X = np.expand_dims(X.T, axis = 1)
  X = torch.tensor(X, dtype = dtype, device = device)

  ## propagate forward
  model.eval()

  with torch.no_grad():
    out = model(X)

  X = X + out

  return X.cpu().numpy().squeeze().T
