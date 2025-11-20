```python
from patchify import patchify
import torch
```

Carico un'immagine di prova dal mnist


```python
from tf_keras.datasets import mnist
(x_tr,y_tr),(x_test,y_test) = mnist.load_data()
```

    WARNING:tensorflow:From C:\Users\User\anaconda3\envs\py311\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
    
    


```python
img = x_tr[0]
print(img.shape) #(28,28) l'immagine è a in bianco e nero
```

    (28, 28)
    

Stampo l'immagine


```python
import matplotlib.pyplot as plt
plt.imshow(img)
plt.axis('off')
plt.show()
```


    
![png](output_5_0.png)
    


La sintassi di pathify è la seguente:  
```patchify(image_to_patch, patch_shape, step=1)```  
Supponiamo di voler creare 4 patches di dim. (14,14)


```python
p = 7
patches = patchify(img, (p,p), step=p)
print(patches.shape)
```

    (4, 4, 7, 7)
    

La dimensione dell'output ha la forma:
```(n_patches_verticali, n_patches_orizzontali, patch_height, patch_width)```  



```python
IMG = torch.zeros(patches.shape[0]*patches.shape[1],p,p)
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        IMG[i*patches.shape[1]+j]=torch.tensor(patches[i][j])

# k = 2
# 0 0 -> 0*2+0 = 0)
# 0 1 -> 0*2+1 = 1)
# 1 0 -> 1*2+0 = 2)
# 1 1 -> 1*2+1 = 3)
```

Tutto il blocco precedente può essere fatto in modo più sintetico usando reshape con questa sintassi:
```python
IMG = patches.reshape(-1, p, p)
```
infatti la dimensione in posizione -1 viene calcolata in automatico. 
```python
fig, axes = plt.subplots(patches.shape[0], patches.shape[1], figsize=(3*patches.shape[0], 3*patches.shape[1]))

for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        axes[i, j].imshow(IMG[i*patches.shape[1]+j], cmap='gray')
        axes[i, j].set_title('')

# Rimuovi i bordi/assi
for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()

```


    
![png](output_10_0.png)
    


## Immagini RGB
Con le immagini a colori il procedimento è lo stesso 


```python
from PIL import Image
from torchvision import transforms
```


```python
path = r"C:\Users\User\Desktop\Foto Via degli Dei\IMG20230714092427.jpg"
img = Image.open(path)
transform = transforms.ToTensor()
img = transform(img)
```


```python
import numpy as np
img = img.numpy()
print(img.shape) #(3, 3072, 4080)
img = np.transpose(img,(1,2,0))
print(f"Nuovo:",img.shape) #(3, 3072, 4080)
```

    (3, 3072, 4080)
    Nuovo: (3072, 4080, 3)
    


```python
p1 = int(img.shape[0]/4)
p2 = int(img.shape[1]/4)
patches = patchify(img, (p2,p1,3), step=(p2,p1,3))
print(patches.shape)
```

    (3, 5, 1, 1020, 768, 3)
    


```python
p1
```




    1020


