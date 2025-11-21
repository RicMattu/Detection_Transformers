# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:55:03 2025

@author: Riccardo Mattu
"""

import torch
from torch import nn
from tf_keras.datasets import mnist
import matplotlib.pyplot as plt
from patchify import patchify
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


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
        
         x = patchify(img,(p,p), step = p)          # array of int (H/p, W/p, p, p)  
         x = torch.tensor(x.reshape(-1,p,p))        # tensor (N,p,p)         #(N,P^2*C)
         x = x.flatten(1)                           # tensor (N,p**2)
         
         proj = nn.Linear(N,d)
         z = proj(x.float())                        # tensor (N, d) 
         z = torch.cat([torch.rand(1,d),z])         # tensor (N+1, d)
        
         p_enc_1d_model_sum = Summer(PositionalEncoding1D(d))
         z=z.unsqueeze(0)                           # tensor (1, N+1, d)
         z = p_enc_1d_model_sum(z)                  # tensor (1, N+1, d)
         
         encoder_l = nn.TransformerEncoderLayer(d_model=d, nhead=2)
         encoder   = nn.TransformerEncoder(encoder_l, num_layers=1)
         
         z_F = encoder(z)                           # tensor (1, N+1, d)
        
         # MLP for pred.
         MLP = nn.Sequential(
             nn.Linear(d,d),
             nn.GELU(),
             nn.Linear(d,d),
             )
         categ = MLP(z_F) 
         
         categ_token = z_F[:, 0, :]  # shape: (batch, d)
         proj_2 = nn.Sequential(
             nn.Linear(d,10),
             nn.Softmax())
         categ = proj_2(categ_token)               
         return categ
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
p_enc_1d_model_sum = Summer(PositionalEncoding1D(d))
z=z.unsqueeze(0)
z = p_enc_1d_model_sum(z)

encoder_l = nn.TransformerEncoderLayer(d_model=d, nhead=2)
encoder   = nn.TransformerEncoder(encoder_l, num_layers=1)
         
z_F = encoder(z)
MLP = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,d))
categ = MLP(z_F) 

categ_token = z_F[:, 0, :]  # shape: (batch, d)
proj_2 = nn.Sequential(
        nn.Linear(d,10),
        nn.Softmax())
categ = proj_2(categ_token)  # shape: (batch, num_classes)








