# Pytorch Implementation of Augmenting Convolutional networks with attention-based aggregation

This is the unofficial PyTorch Implementation of "Augmenting Convolutional networks with attention-based aggregation"

reference: https://arxiv.org/pdf/2112.13692.pdf

## Prerequisites

+ PyTorch
+ PyTorch Lightning
+ timm
+ torchmetrics
+ torchvision
+ python3
+ CUDA

## Comments
- Due to computation limits, CIFAR100 dataset was used in contrast to ImageNet in the original paper. 
- Since the official code is not released yet, there may be differences in structures and hyperparameters.
  - Most of the hidden dimensions were chosen based on guesswork.   
- MADGRAD was used instead of LAMB optimizer. 
-   (I thought it would be inefficient to use LAMB for small batchsizes in my local machine) 
- LayerScale will be added soon


## Citations

```bibtex
@misc{touvron2021augmenting,
      title={Augmenting Convolutional networks with attention-based aggregation}, 
      author={Hugo Touvron and Matthieu Cord and Alaaeldin El-Nouby and Piotr Bojanowski and Armand Joulin and Gabriel Synnaeve and Hervé Jégou},
      year={2021},
      eprint={2112.13692},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
