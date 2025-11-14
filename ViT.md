# 1) An Image is worth a 16 x 16 words
The first attempt of applying the attention mechanism to image inputs is not the DETR but the Vision Transformer
proposed by Dosovitskiy et al. [^fn1].  
Here we simply describe the main idea of the model.  
Its structure is depicted in the following figure:  
![ViT overview](https://github.com/google-research/vision_transformer/blob/main/vit_figure.png)

## ViT model
### 1) Patchification
Given an image $x \in \mathbb{R}^{H \times W \times C}$, where $(H,W)$ is the resolution and $C$ the number of channels; it is reshaped into a sequence of $N$ flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$ each of which of resolution of $(P,P)$, so that $N = HW/P^2$

### Transformer: patches -> patch embeddings -> 
The transformer receives as input a 1D sequence of N tokens
it uses a constant cotext vector D - dimensional with a trainable linear projection
prepended a learneble embedding

### Classification head (MLP)











[^fn1]: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. arXiv preprint [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
