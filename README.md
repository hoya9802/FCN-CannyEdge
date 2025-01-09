# FCN Tuning

## Environment of Implementation

### Version of Python
```shell
conda create -n "env name" python==3.8
```
### Version of Pytorch
```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit==10.1 -c pytorch
```
**Warning : Recommend using more than version 2.0 of Pytorch because of one-hot encoding and permute compatibility issues!)**

### Installation of CUDA
```shell
conda install anaconda cudnn
```

## Dataset
### VOC dataset
1464 training set and train gt / 1449 test set and test gt / 21 classes
<br>
If you use [canny_edge.py](https://github.com/hoya9802/FCN_Tuning/blob/main/canny_edge.py), you can obtain Canny edge datasets

## Implimentation
All of these tuned structures are based on [FCN_8s](https://arxiv.org/pdf/1411.4038)
 - FCN_8s : Original FCN structure
 - FCRN_8s : ResNet34 + FCN Decoding structure (ResNet34 is slightly modified for Existing FCN Decoding process)
 - DFCRN_8s : ResNet34 + tuned FCN Decoding structure


## 1. Edge Amplification

**How the Human Brain Distinguishes Objects**<br>
The human brain unconsciously extracts edges by analyzing color and brightness differences to distinguish objects. It aggregates this information to classify the objects in an image projected onto the retina.

However, we found that traditional neural networks struggle to differentiate colors effectively in the early layers.

**Illustration of the Problem**<br>

![그림1](https://github.com/user-attachments/assets/5516ffc3-8262-45ed-9416-00cd39ae364d)

In the example images shown above, the two images are distinctly different. However, when a convolution operation is performed with a filter initialized with weights of 1, the resulting feature maps for both images are identical, producing a value of 255 x 9.

This indicates that the network relies entirely on the loss function to learn and improve its ability to distinguish objects during training.

Proposed Solution
To address the aforementioned issues, we utilized the Canny Edge Detection algorithm, which is known for its ability to effectively detect edges. We applied the Canny Edge algorithm to the input images and combined the resulting edge maps with the original images by taking the maximum value at each pixel. This preprocessing step was applied to all images.

![그림2](https://github.com/user-attachments/assets/6782ebf2-e39d-4a32-aea5-0a3d788cb5a9)

The image on the right shows a combination of the left image and its Canny edge map, created using the following formula:

Equation:

By applying convolution operations to the processed image, we observed that the values corresponding to the edges were amplified, highlighting the edge regions more effectively.

**Results**

<p align="center">
  <img src="https://github.com/user-attachments/assets/1ac66cc1-9682-464d-94d3-0a63e1761158" alt="이미지 설명" width="771">
</p>


As shown in the results, the original image remains largely intact, while the edge regions are distinctly amplified and highlighted in white. This demonstrates that our preprocessing method effectively enhances the edges without significantly altering the original image.

**Unexpected Outcomes**

<p align="center">
 <img width="796" alt="스크린샷 2025-01-09 오전 1 20 06" src="https://github.com/user-attachments/assets/25f54a13-2eea-48e6-8cb3-b7a6d0e77c14" />
</p>

Contrary to our expectations, the model's performance did not show significant improvement after incorporating edge-enhanced preprocessing. We hypothesized that providing both edge and color information during the feed-forward process would lead to faster saturation in the early stages of training. However, even after 50,000 epochs (see Figure 2), there was no substantial difference observed.

Additionally, the model's accuracy showed no notable improvement. In fact, there was a slight decrease in accuracy compared to the baseline (see Table 1).

Based on these findings, all subsequent experiments were conducted using the original images without Canny Edge preprocessing.

## 2. FCRN(Fully Convolutional ResNet)
**Limitations of the Baseline FCN Model**<br>

<p align="center">
 <img width="477" alt="스크린샷 2025-01-09 오전 11 46 10" src="https://github.com/user-attachments/assets/bc2adb73-be00-415d-9b02-562d711ab48c" />
</p>

In the original Fully Convolutional Network (FCN) paper, the FCN model was built upon the VGG16 architecture. Consequently, the early layers utilized 3x3 filters. While the VOC dataset contains 21 classes, the VGG16 model compresses the feature maps from 4096 dimensions to 21 in the final layer. This compression process inevitably results in significant information loss (see Figure 3).

**Advantages of Using ResNet34**

<p align="center">
 <img width="846" alt="스크린샷 2025-01-09 오후 12 11 15" src="https://github.com/user-attachments/assets/4871eaa8-bb16-4d68-a960-246938c76899" />
</p>

By using ResNet34, we gain several benefits. First, the network uses a 7x7 filter in the initial layer, allowing it to capture more coarse information compared to VGG16. Additionally, ResNet34 outperforms VGG16 in terms of classification performance. The model also reduces the dimensionality from 4096 to 21 in the final layer, whereas ResNet34 reduces it from 512 to 21, which helps preserve more information and reduces information loss (see Figure 4).

To ensure the number of channels added during skip connections aligns, we removed the first layer's Maxpooling operation. Furthermore, instead of using stride 2 to reduce the model size, we employed Maxpooling to reduce the size and only propagate the maximum values. This prevents the features from being diluted during the process.

**Results**

<p align="center">
 <img width="626" alt="스크린샷 2025-01-09 오후 12 14 47" src="https://github.com/user-attachments/assets/5a08e868-78cc-4694-b3d7-8c909aa3c2fd" />
</p>

The results show that FCRN_8s outperforms FCN_8s in predicting and displaying more accurate results for single objects. However, when it comes to multiple objects, FCRN_8s does not provide significantly better results, indicating the need for a different structural approach to handle multiple object scenarios more effectively.

## 3. DFCRN(Dense Fully Convolutional ResNet)

**FCN Skip Connection Strategy**

In the FCN model, skip connections occur once at 14x14 and again at 28x28 during the decoding process, after which they are merged. According to the original paper, skip connections at sizes larger than 28x28 did not significantly improve the model's performance, which is why the model was limited to processing up to 28x28.

To improve the model's performance, we sought to modify the existing skip connection structure.

**Modification of Transposed Convolution with Matrix Multiplication**

<p align="center">
 <img width="614" alt="스크린샷 2025-01-09 오후 12 16 31" src="https://github.com/user-attachments/assets/8f474078-92e5-4ebf-ac95-9fdd20d0a204" />
</p>

By replacing the traditional transposed convolution with matrix multiplication, we hypothesized that without skip connections, the model would naturally learn based on the relative positions of neighboring coordinates during training. As a result, we believed that both transposed convolution and bilinear interpolation would produce similar outcomes.

To address this, we proposed that the features from the encoding process, which are learned to fit the input data, should be used to enhance the surrounding coordinates during the decoding process. This approach would allow the model to create more detailed representations of the target positions, ultimately helping the model converge to optimal values and improving overall performance (see Figure 5).

**Modification of Skip Connection Strategy**

<p align="center">
 <img width="515" alt="스크린샷 2025-01-09 오후 12 57 53" src="https://github.com/user-attachments/assets/e6330607-633e-45ea-92d1-190780c6a774" />
</p>

The original model connected feature maps of the same size from the encoding and decoding processes, specifically linking the 14x14 and 28x28 feature maps directly. We rethought this approach by down-sampling the feature maps from the earlier layers of the network to match the sizes of the target feature maps (14x14 and 28x28). These down-sampled feature maps were then added to the skip connections (see Figure 6).

This modification was expected to allow the transposed convolutions in the decoding process to gather more diverse information, enabling more accurate interpolation and enhancing the overall performance of the model.

**Results**

<p align="center">
 <img width="622" alt="스크린샷 2025-01-09 오후 12 58 35" src="https://github.com/user-attachments/assets/39a4644d-903e-4f1e-a21f-93cbc05a3a94" />
</p>

As shown in the results, the modified model was able to capture more detailed information compared to the previous models. It also demonstrated better performance in handling multiple objects. Furthermore, we observed an increase in the mean Intersection over Union (mean IU) compared to previous models, indicating an improvement in overall model performance.

## 4. Discussion and Future Work

One of the major disappointments was the lack of significant performance improvement in edge amplification, as well as the absence of noticeable acceleration during the initial stages of the model’s learning. Several potential causes were identified for this:

1. Canny Edge as Ground Truth: We believe that the Canny Edge did not align well with the ground truth. While the Canny Edge method was effective in detecting finer details of edges, it resulted in capturing more edges than necessary for semantic segmentation, leading the model to learn from excessive and irrelevant edge information, which likely hindered performance improvement.

2. Color-Dependent Amplification: The degree of amplification varied greatly depending on the colors in the image. This inconsistency made it difficult to generalize the amplification process effectively across different images.

3. Edge Amplification Behavior: The amplification mechanism, which relied on maximizing the edge values, resulted in all values in the 3D color image reaching 255, making edge amplification ineffective in areas like white regions where pixels are already at their maximum value. This led to suboptimal amplification in certain regions.

For future work, we propose the following improvements:

1. Data Type Modification: The current uint8 data type, which has a maximum value of 256, limits the range of values and could lead to overflow issues. By switching to a float32 data type, overflow can be prevented. Additionally, applying a logarithmic transformation would allow stronger emphasis on edge regions while maintaining consistent amplification across the image. Afterward, converting the image back to uint8 would ensure that the edges are consistently amplified while also being robust to white regions.

2. Learning Canny Edge Parameters: The two key parameters for Canny Edge detection (max_val, min_val) could be learned as part of the training process. This adjustment may lead to better performance, as the model would have the ability to adapt these parameters to the specific image data.

## 5. Conclusion

Starting with Edge Amplification, I identified mathematical and structural issues in the original FCN and explored potential improvements. By implementing these improvements, I was able to achieve enhanced results, which was extremely rewarding. However, there are still many areas for further improvement. Due to time constraints and limitations in coding skills, I was unable to fully implement all the ideas I had conceptualized.

Despite this, working on this final project was a highly enjoyable experience. Revisiting several research papers, identifying areas where improvements were needed, and designing and validating these improvements from both mathematical and structural perspectives was an incredibly fulfilling process. It was a valuable learning experience that allowed me to recognize my shortcomings and grow in the field.

## 6. Reference

[FCN paper](https://arxiv.org/pdf/1411.4038.pdf)<br>
[ResNet paper](https://arxiv.org/pdf/1512.03385.pdf)<br>
[VGG16 paper](https://arxiv.org/pdf/1409.1556)

