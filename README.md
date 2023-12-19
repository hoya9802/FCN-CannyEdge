# FCN Tuning

## Environment of Implementation

### Version of Python
 - conda create -n "env name" python==3.8

### Version of Pytorch
 - conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit==10.1 -c pytorch
<span style="color:red">**(Recommend using more than version 2.0 of Pytorch because of one-hot encoding and permute )compatibility issues!**</span>

### Installation of CUDA
 - conda install anaconda cudnn

## Dataset
### VOC dataset
1464 training set and train gt / 1449 test set and test gt / 21 classes
<br>
If you use canny_edge.py, you can obtain Canny edge datasets

## Implimentation
All of these tuned structures are based on FCN_8s
 - FCN_8s : Original FCN structure
 - FCRN_8s : ResNet34 + FCN Decoding structure (ResNet34 is slightly modified for Existing FCN Decoding process)
 - DFCRN_8s : ResNet34 + tuned FCN Decoding structure

