# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:55:03 2025

@author: Riccardo
"""

import torch
from torch import nn
from tf_keras.datasets import mnist
import matplotlib.pyplot as plt
from patchify import patchify

# Importing random image from fashion mnist
(x_tr,y_tr),(x_test,y_test) = mnist.load_data()

# Extracting resolution
index = torch.randint(0,10,(1,)).item()
img = x_tr[index] 
(H,W) = (img.shape) # original resolution
C = 1 # grayscale


p = 7


# N = H*W/P**2


class vit(nn.Module):
     def __init__(self):
         super().__init__()
        
     def forward(self, img): 
        
         x = patchify(img,(p,p), step = p)           
         x = torch.tensor(x.reshape(-1,p,p))         # (N,P^2*C)
         x = x.flatten(1)
         
         # linear proj.
         proj = nn.Linear(N,d)
         z = proj(x.float())
         z = torch.cat([torch.rand(1,d),z])          # (N+1,D)
        
#         # add PE
#         z = PE(z)                     # (N+1,D)
        
#         # Transformer encoder
#         z_F = encoder(z)              # (N+1,D)
        
#         # MLP for pred.
#         categ = MLP(z_F)                
#         return "Ok"
# #-----------------------------------
# # Random image 
# img = torch.rand(H,W,C)

# x = patchifier(img)           # (N,P^2*C)
        
# z=lin_proj(x)                 # (N+1,D)
        

# z = PE(z)                     # (N+1,D)
        
#         # Transformer encoder
# z_F = encoder(z) 

# categ = MLP(z_F)

    
# model = vit()
# model(img)
        
x = patchify(img,(p,p), step = p)
x.shape
x = torch.tensor(x.reshape(-1,p,p))
x = x.flatten(1)
N = x.shape[1]
d = 128
proj = nn.Linear(N,d)
z = proj(x.float())
z = torch.cat([torch.rand(1,d),z])
