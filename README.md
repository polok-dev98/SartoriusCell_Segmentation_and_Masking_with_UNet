# Sartorius Cell Instance Segmentation and Masking using UNet with attention machanism. Code done with Tensorflow framework.



Dataset download link : https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data?select=LIVECell_dataset_2021

# About the dataset: 
<code>train.csv</code> file provided to competitors contains the following columns 
- <code>id</code> - unique identifier
- <code>annotation</code> - run length encoded pixels for the identified neuronal cell
- <code>width</code> - image width
- <code>height</code> - image height
- <code>cell_type</code> - class of the cell
- <code>plate_time</code> - time the plate was created
- <code>sample_id</code> - id of the sample
- <code>sample_date</code> - date the sample was created
- <code>elapsed_timedelta</code> - time since the image was shot

<code>ids</code> present in the column <code>id</code> correspond to image files in <code>train</code> folder

<code>annotations</code> are run length encoded pixels in order to create a mask we will have to decode the annotations

# Steps: 

# Load Data

In dataset Masks are encoded in the annotation column by an algorithm called **Run Length Encoding**. RLE encodes a mask into a vector where vector index corresponds to flattened mask matrix index and the value at that index corresponds to length of the annotation.So perform RLE encodings.

# Preparing training and validation datasets

In order to train the neural network we have to load the data and optimize it for training

first train dataset has to be split into training and validation parts

**Validation dataset** is used to validate how well your model is doing before performing predictions on test dataset. Having a validation dataset helps to detect wheter the model is overtrained

Tensorflow provides <code>tf.data.Dataset</code> api which is very usefull when creating data pipelines for ML models, since it supports caching, batching, one can perform data preprocessing using <code>.map</code> method on <code>Dataset</code> object.

Having created functions that yield images and masks we can use <code>from_generator</code> method to create datasets from generators
# Modeling

**UNET** is a Conv net architecture proposed by Olaf Ronneberger, Philipp Fischer, Thomas Brox in their paper [U-Net: Convolutional Networks for Biomedical Image Segmentation
](https://arxiv.org/pdf/1505.04597.pdf). It has been very successful in performing semantic segmantation on many benchmarks. The architecture is composed by encoder and decoder networks with a bottleneck in between. Let's see a visualization from the authors.

<img src='https://miro.medium.com/max/680/1*TXfEPqTbFBPCbXYh2bstlA.png'/>

**The encoder** is composed of conv block each with two 3x3 conv layers followed by max pooling with pool size of 2, there is a total of 4 of this layers with number of filters 512, 256, 128, 64

**The bottlenck** is a simple conv block of two 3x3 conv layers with 1024 filters

**The decoder** consists of 4 upsampling conv block, each having tranposed conv layers with filters size of 2 and strides of 2, after upsampling skip connections are added, lastly two conv 3x3 layers are applied

Key idea of UNET are **skip connections**. Output of each encoder layer is added to corresponding decoder layer, this preserves the spatial structure of the input image, since upsampling in the decoder leaves unprecise expansions. Adding output from encoder layer helps with preserving a lot of information.

# Residual UNET 
**Residual UNET** simply is a UNET with residual conv blocks instead of regular conv blocks, the architecture looks like this

<img src='https://ichi.pro/assets/images/max/640/0*Q9iM4_vhdCYDlTsO.png'/>

It was proposed by Zhengxin Zhang, Qingjie Liu, Yunhong Wang in their paper [Road Extraction by Deep Residual U-Net](https://arxiv.org/abs/1711.10684).
<br/>
**What are residual conv blocks?** First let's see a simple visualization of a residual unit

<img src='https://miro.medium.com/max/1140/1*D0F3UitQ2l5Q0Ak-tjEdJg.png' />

In normal conv units tensors are directly propagated throught conv layers. This way of propagation has one big issue - **vanishing gradients**. Vanishing gradients problem occurs when training very deep networks, during backpropagation gradinets are propagated from deeper layers into shallower ones, sometimes the gradients can get smaller (close to 0) at each consecutive layer, this prevents the network from learning. Residual units first save input tensor then propagate them throught conv layers, then add saved input tensor to conv layers output, thus learning **identity mapping** and greatly helping with the vanishing gradients problem.

# Residual UNET with Attention
Attention was introduced to UNET in 2018's paper [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/pdf/1804.03999) by Ozan Oktay et al.

**What is attention in the context of computer vision?** Attention is very often used in NLP problems as a way to make a model focus more on for example a part of a sentence. In computer vision attention is a mechanism that allows your network to look only at certain parts of image. Such a part is called a **region of interest** (ROI). Looking at only parts of an image increases computational efficiency, while adding only a small amount of parameters. Below is a diagram from the paper, as you can see attention gate is aplied before concatenation skip connetions to decoder layer.

<img src='https://www.researchgate.net/publication/324472010/figure/fig1/AS:614439988494349@1523505317982/A-block-diagram-of-the-proposed-Attention-U-Net-segmentation-model-Input-image-is.png' />

**Why is attention needed for UNET?** Skip connections are main characteristic of UNET, they help to preserve spatial structure in the upsampling layers. One issue with skip connections is that since they come from shallower layers of the network they extract less complex feature maps, this means that many unuseful low-level features are concatenated to the decoder, attention learns which of those features are worth taking a look at and which are just noise. The end result is a more computationaly efficient network and slighlty better performance. Let's break down the attention gate architecutre. 

<img src='https://miro.medium.com/max/1838/1*Q1aMxFm1L6KJeia5wCmC5A.png' />

Attention gate takes as a input a skip connection and the output from the previous decoder layer. Matematically attention gate does the following operation

$$ q_{att}^{l} = \upsilon^{T}(\sigma_1(W^{T}_{x}x^{l}_{i} + W^{T}_{g}g_{i} + b_{g})) + b_{\upsilon} $$
$$ \alpha_{i}^{l} = \sigma_{2}(q_{att}^{l}(x_{i}^{l}, g_{i}; \Theta_{att})) $$

Where $\sigma_{2}(x_{i,c}) = \frac{1}{1+\exp(-x_{i, c})}$, $\Theta_{att}$ contains linear transformations $W_x, W_g$, which are computed using 1x1x1 convolutions for the input tensors. Let's see how we can implement this.

* Input g (previous decoder layer output) and x (skip connection)
* Convolve x with 1x1 filter and stride = 2, and g with 1x1 filter and stride = 1
* Add together x and g
* Apply ReLU activation function
* $\psi =$ 1x1x1 convolution 
* Apply sigmoid activation function
* Upsample sigmoid output to original input size (2x2)
* $att = multiply(upsample, x_{input})$ 
* 1x1 convolution with n_filters = n_input_x_filters and batch normalization

# Training
**Callbacks** are a way to introduce additional logic to the training loop, for example Tensorflow allows to create a callback that saves model's weigth at the end of each epoch, you can tweak this callback to save only best weights (weights when model's loss is minimized). Tensorflow docs - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback

The callbacks I am going to use:
- <code>ModelCheckpoint</code>
- <code>ReduceLROnPlateau</code>
- <code>EarlyStopping</code>

# Loss function

Since instance segmantation is binary classification problem, one might think binary crossentropy loss function is a perfect fit, however this is not the case. Binary crossentropy function makes training segmentation models difficult because it conisders only one pixel, it doesn't take into the account the whole image. How can we do better?

<h2>Dice Loss</h2>
Dice coeficient is a statistic from 1940s developed to be a measure of similarity between two samples. It was introduced to the field of computer vision in 2016 for 3d segmantation by Milletari et al.

$$D = \frac{2\sum\limits_{i = 1}^{n} p_{i}g_{i}}{\sum\limits_{i = 1}^{n} p_{i}^{2} + \sum\limits_{i = 1}^{n} g_{i}^{2}}$$

$p_{i}$ and $g_{i}$ are the values of corresponding pixels in reality numerator is intersection of two sets and denominator is the sum of areas of these two sets, dice coefficient values range from 0 to 1 where 1 would mean that the images are practically the same and 0 would mean that there is no similarity at all. Since optimizers in machine learning are trying to minimize the loss function dice loss is defined as
$$\ell = 1 - D$$

<h2>Intersection Over Union</h2>
IoU is this competitions evaluation metric, it is defined as
$$IoU(A, B) = \frac{\mid A \cap B \mid}{\mid A \cup B \mid}$$
Where $A$ and $B$ both are sets, simillariy to dice coeficient it takes values between 0 and 1 where 1 would mean that the two sets are identical and 0 - that the sets have nothing in common. IoU isn't used as a loss function mainly becouse it is not differentiable.

# Testing 
After training the model we use the model for generate predictions (Segmentations and masking)

