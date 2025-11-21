# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:14:43 2025

@author: User
"""
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D,Summer
import matplotlib.pyplot as plt

## 1D 

# Returns the position encoding only
p_enc_1d_model = PositionalEncoding1D(256)
x = torch.rand(1, 128, 256)
# Figure 1
plt.figure(figsize=(10,10))
fig1 = plt.imshow(x[0])
plt.gcf().colorbar(fig1)
plt.show()
#################################
penc_no_sum = p_enc_1d_model(x) # penc_no_sum.shape == (1, 6, 10)

# Figure 2
plt.figure(figsize=(10,10))
fig2 = plt.imshow(penc_no_sum[0])
plt.gcf().colorbar(fig2)
plt.show()


# # Return the inputs with the position encoding added
# p_enc_1d_model_sum = Summer(PositionalEncoding1D(10))


# penc_sum = p_enc_1d_model_sum(x)
# print(penc_no_sum + x == penc_sum) # True


# =============================================================================
# 2D
# =============================================================================
D = 256

# Istanzio il model PE 2D
p_enc_2d = PositionalEncoding2D(D)

# Simulo una "feature map"  (batch=1, channels=D, H=16, W=16)
x = torch.rand(1, D, 16, 16)

#################################
# Figure 1: l'input originale x
#################################
plt.figure(figsize=(10, 10))
fig1 = plt.imshow(x[0, 0].detach())   # visualizzo il primo canale
plt.colorbar(fig1)
plt.title("Input feature (channel 0)")
plt.show()

#################################
# Applico il positional encoding 2D
#################################
penc = p_enc_2d(x)   # shape = (1, D, H, W)

#################################
# Figure 2: visualizzo un canale del PE
#################################
plt.figure(figsize=(10, 10))
fig2 = plt.imshow(penc[0, 0].detach())   # primo canale
plt.colorbar(fig2)
plt.title("2D Positional Encoding (channel 0)")
plt.show()

#################################
# Figure 3: differenza tra x e PE(x)
#################################
plt.figure(figsize=(10, 10))
fig3 = plt.imshow((penc - x)[0, 0].detach())
plt.colorbar(fig3)
plt.title("Difference: PE(x) - x")
plt.show()