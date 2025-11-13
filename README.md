# Detection_Transformers
This repo is a work in progress,  
Starting from my thesis, I will make a small compendium of attention mechanisms applied to CV.  


Please be patient!

<!--
## Timeline
<img src="images/isteresi.png" alt="Esempio di immagine" width="300"/>  
-->
## 1) An Image is worth a 16 x 16 words
The first attempt of applying the attention mechanism to image inputs is not the DETR but the Vision Transformer
proposed by Dosovitskiy et al. [^fn1].  
Here we simply describe the main idea of the model.  
Its structure is depicted in the following figure:  
![ViT overview](https://github.com/google-research/vision_transformer/blob/main/vit_figure.png)

### ViT model
Given an image $x \in \mathbb{R}^{H \times W \times C}$




[^fn1]: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. arXiv preprint [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
