# coding_sparseness
Scripts here are related to Liu, X., Zhen, Z., & Liu, J. (2020). Hierarchical sparse coding of objects in deep convolutional neural networks. Frontiers in computational neuroscience, 14, 110..

## setup framework
Description: The DNNBrain toolbox was used to extract the DCNN activation. Check out [here](https://github.com/BNUCNL/dnnbrain/). <br>

```
mkdir -p models
cd models
wget https://download.pytorch.org/models/vgg11-bbd30ac9.pth
mv vgg11-bbd30ac9.pth vgg11.pth
wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
mv alexnet-owt-4df8aa71.pth alexnet.pth
DNNBRAIN_DATA=models
```
