# coding_sparseness
Scripts here are related to Liu et al., 2020, Hierarchical sparse coding of objects in deep convolutional neural networks (in review).

## setup framework

```
mkdir -p models
cd models
wget https://download.pytorch.org/models/vgg11-bbd30ac9.pth
mv vgg11-bbd30ac9.pth vgg11.pth
wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
mv alexnet-owt-4df8aa71.pth alexnet.pth
```


## DCNN activation extraction
Description: The DNNBrain toolbox was used to extract the DCNN activation. Check out [here](https://github.com/BNUCNL/dnnbrain/). 

```
DNNBRAIN_DATA=models  ipython activation_extraction.py
```

## Analyses for population sparseness
Code: /coding_sparseness.py
