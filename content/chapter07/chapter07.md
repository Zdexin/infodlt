# 第7章 卷积神经网络的介绍
&emsp;&emsp;在数据时代，卷积神经网络（cnn）是一种特殊的深度学习体系结构，它使用卷积运算来提取输入图像的相关解释特征。卷积层作为前馈神经网络进行连接，同时使用卷积操作模拟人脑进行识别。单个的外层神经元接受外界刺激并做出反应。特别的，生物界的成像问题会成为一大难题。但在此单元，我们会学习到如何使用卷积神经网络来发现图像中的图案。
以下内容会出现在此章中：
1.	卷积运算
2.	推动发展
3.	卷积神经网络中的不同层
4.	卷积神经网络基础实例：MNIST数字分类
## 一.卷积运算
&emsp;&emsp;卷积计算在计算机视觉领域得到了广泛的应用，其性能超过了我们所使用的大多数传统的计算机视觉技术。卷积神经网络结合了著名的卷积运算和神经网络，因此得名卷积神经网络。因此，在深入学习卷积神经网络之前，我们将介绍卷积运算，看看它是如何工作的。<br>
&emsp;&emsp;卷积操作的主要目的是从图像中提取信息或特征。任何图像都可以被视为一个数值矩阵，该矩阵中的一组特定值将形成一个特征。卷积操作的目的是扫描该矩阵并尝试提取该图像的相关或解释性特征。例如，考虑一个5乘5的图像，其对应的强度或像素值显示为0和1：<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/1.png) <br>
&emsp;&emsp;并考虑以下3*3的矩阵：<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/2.jpg) <br>
我们可以使用3 x 3图像对5 x 5图像进行卷积，如下所示 :<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/3.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/4.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/5.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/6.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/7.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/8.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/9.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/10.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/11.jpg) <br>
上图可归纳如下。 为了使用3 x 3卷积内核对原始5 x 5图像进行卷积，我们需要执行
以下操作：
1.	使用橙色矩阵扫描原始绿色图像，每次只移动1个像素（步幅）
2.	对于橙色图像的每个位置，我们在橙色矩阵和绿色矩阵中的相应像素值之间进行逐元素乘法
3.	将这些逐元素乘法运算的结果加在一起以获得单个整数，该整数将在输出粉色矩阵中形成单个值<br>
&emsp;&emsp;从上图中可以看出，橙色3乘3矩阵仅在每次移动（步幅）中操作一次原始绿色图像的一部分，或者只出现一次。
 所以，让我们把以上操作运用到卷积神经网络中：
1.	橙色3 x 3矩阵称为kernel（卷积核）, feature detector,或则filter(过滤器)
2.	结果矩阵被称为feature map（特征图）<br>
&emsp;&emsp;因为我们基于原始输入图像中的卷积核和相应像素之间的元素乘法得到feature map，所以改变kernel或filter的值将每次给出不同的feature map。<br>
&emsp;&emsp;因此，我们有理由相信需要在卷积神经网络的训练过程中自己找出特征检测器的值，但在这里并不是这样的。 CNN在学习过程中找出这些数字。 因此，如果我们有更多filter，则意味着我们可以从图像中提取更多功能。<br>
&emsp;&emsp;在进入下一节之前，让我们介绍一些通常在CNN上下文中使用的术语：<br>
&emsp;&emsp;stride：我们之前简要地提到了这个术语。 通常，stride是我们在输入矩阵的像素上移filter weigh的像素数。 例如，步幅1意味着在对输入图像进行卷积时将filter移动一个像素，而步幅2意味着在对输入图像进行卷积时将filter移动两个像素。 我们的步幅越大，生成的特征映射越小。<br>
&emsp;&emsp;Zero-padding：如果我们想要包含输入图像的边框像素，那么我们的部分滤镜将位于输入图像之外。 Zero-padding是将边界周围填充0，直至满足要求。<br>
## 二.推动<br>
&emsp;&emsp;传统的计算机视觉技术用于执行大多数计算机视觉任务，例如检测目标和分割对象。这些传统的计算机视觉技术的性能很好，但它从未真正使用，例如自动驾驶汽车。2012年，Alex Krizhevsky 介绍了CNN，它通过将对象分类错误率从26％降低到到15％，在ImageNet竞赛中取得了突破。自此之后，CNN已经被广泛使用，并且已经发现了不同的变化。它甚至在ImageNet 竞赛中的正确率超过人类识别，如下图所示：<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/12.jpg) <br>
### 卷积神经网络的应用
&emsp;&emsp;自从CNN在计算机视觉甚至自然语言处理的不同领域取得突破以来，大多数公司已将这种深度学习解决方案集成到他们的计算机视觉回声系统中。 例如，谷歌使用这种架构作为其图像搜索引擎，Facebook使用它进行自动标记等等：
&emsp;&emsp;CNNs凭借其架构实现了这一突破，该架构直观地使用卷积操作从图像中提取特征。在接下来的介绍中，你会发现它与人类大脑的工作方式非常相似。<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/13.jpg) <br>
## 三.卷积神经网络的不同层
典型的CNN架构由多个执行不同任务的层组成，如上图所示。在本节中，我们会详细介绍他们，并且通过特别的方法将他们联系在一起，将会给计算机带来大的突破。
### 输入层
所有卷积网络结构都包含输入层。所有在这之后的卷积层和池化层都期望输入层有特殊的格式，输入应该是张量，具有以下格式：<br>
[batch_size, image_width, image_height, channels]<br>
batch_size是在应用随机梯度下降期间使用的原始训练的随机样本。<br>
image_width是卷积网络中输入图像的宽度。<br>
image_height是卷积网络中输入图像到网络的高度。<br>
channels是输入图像的颜色通道数。 这个数字可以是RPG图像中的3或者是二值图像中的1<br>
举个例子，考虑著名的MNIST数据集，我们要利用卷积神经网络来执行数字分类就要使用此数据集。<br>
如果数据集由单色28 x 28像素图像（如MNIST数据集）组成，则输入图层的所需形状如下：<br>
[batch_size, 28, 28, l].
要更改输入要素的形状，我们可以执行以下华政操作：<br>
input_layer = tf.reshape(features["x"], [–l, 28, 28, l])<br>
正如我们所见我们将batch_size的值规定为-1，这就意味着这个值应该根据features中的输入值来决定。通过此操作，我们能够通过batch size来微调卷积神经网络。
正如重塑操作的例子，假设我们将输入样本分成5份，我们的数组x将保存输入图像的3920个值。其中该数组的每个值对应于一个像素。这样的操作下，输入层将有以下格式：<br>
[5, 28, 28, l]<br>
### 卷积层
如前所述，卷积层的名称来自卷积运算。 进行这些卷积步骤的主要目的是从输入图像中提取特征，然后将它们提供给线性分类器。<br>
在自然图像中，特征值可以在图像中的任何位置。 例如，边缘可能位于图像的中间或角落，因此卷积步骤是能够在图像中的任何位置检测这些特征。<br>
在TensorFlow中定义卷积步骤非常容易。 例如，如果我们想要使用ReLU激活函数将大小为5乘5的20个filter应用于输入层，那么我们可以使用以下代码行来执行此操作：<br>
conv_layerl = tf.layers.conv2d( inputs=input_layer, filters=2O,<br>
kernel_size=[5, 5], padding="same", activation=tf.nn.relu)<br>
这个conv2d函数的第一个参数是我们在前面的代码中定义的输入层，它具有适当的形状，第二个参数是filters参数，它指定要应用于图像的过滤器的数量，其中过滤镜数量越多，从输入图像中提取的特征就越多。 第三个参数是kernel_size，它表示过滤器或特征检测器的大小。‘same’处是填充参数，以将零填充引入输入图像的边角像素。 最后一个参数指定应该用于卷积运算输出的激活函数。<br>
因此，在我们的MNIST示例中，输入张量将具有以下形状：<br>
[batch_size, 28, 28, l]<br>
经过卷积操作的形状如下：<br>
[batch_size, 28, 28, 20]<br>
输出张量与输入图像具有相同的尺寸，但现在我们有20个通道表示将20个滤镜应用于输入图像。<br>
### 非线性映射
&emsp;&emsp;在卷积步骤中，我们讨论把卷积层输出结果通过ReLU函数做非线性映射：<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/14.jpg) <br>
&emsp;&emsp;ReLU激活函数用零替换所有负像素值，整个卷积步骤过程和激活函数的目的是在输出图像中引出非线性结构，因为这对于训练过程是有用的，并且我们使用的数据通常是非线性的。要清楚地了解ReLU激活功能的好处，请查看下图，其中显示了卷积步骤的行输出及其处理之后的版本：<br>
### 池化层
&emsp;&emsp;我们学习过程中的重要步骤是池化，有时也会被称为缩减采样或子抽样。该步骤主要用于降低卷积步骤（特征图）的输出的维数。此池化步骤的优点是减少了要素图的大小，同时将重要信息保留在缩减版本中。<br>
&emsp;&emsp;下图显示了此步骤，通过使用2 x 2过滤器扫描图像，并在应用Max计算时使用步幅2。这种操作称为最大池化：
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/15.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/16.jpg) <br>
我们可以使用以下代码行将卷积步骤的输出连接到池化层：<br>
`pool_layerl = tf.layers.max_pooling2d(inputs=conv_layerl, pool_size=[2, 2], strides=2)`<br>
池化层接收来自卷积步骤的输入，其形状如下：<br>
[batch_size, image_width, image_height, channels]<br>
例如，在我们的数字分类任务中，池化层的输入将具有以下形状：<br>
[batch_size, 28, 28, 20]<br>
池化操作的输出将具有以下形状：<br>
[batch_size, l4, l4, 20]<br>
在这个例子中，我们将卷积步骤的输出大小减少了50％。 此步骤非常有用，因为它仅保留重要信息，并且还降低了模型的复杂性，从而避免了过度拟合。<br>
### 全连接层
在堆叠了一堆卷积和汇集步骤之后，我们接下来接入全连接层，我们将从输入图像获取的提取的高质量的特征提供给此全连接层，并凭这些高质量特征值进行数字分类<br>
例如，在数字分类任务的情况下，我们可以按照卷积和汇集步骤与具有1024个神经元和RELU激活来执行实际的分类的完全连接层。 此完全连接的图层接受以下格式的输入：<br>
[batch_size, features]<br>
因此，我们需要从第二次池化层重新整形或展平输入要素图以匹配此格式。<br>
`pooll_flat = tf.reshape(pool_layerl, [–l, l4 * l4 * 2O])`<br>
在这个函数中，我们用-1来表示batch size 的大小并且每个从池化层中输出的事例都会是宽度为14 高度为14，并且有20个通道。<br>
所以这个重塑操作的最终输出如下：<br>
[batch_size, 3l36]<br>
最后，我们可以使用TensorFlow的dense（）函数来定义具有所需数量的神经元（单位）和最终激活函数的全连接层：<br>
·dense_layer = tf.layers.dense(inputs=pooll_flat, units=lO24, activation=tf.nn.relu)·
### Logits layer
最后，我们需要logits层，来获取全连接层的输出，然后生成原始预测值。 例如，在数字识别的情况下，输出将是10个值的张量，其中每个值表示0-9的各个等级的分数。 所以，让我们以数字分类示例来定义这个logit层，我们只需要10个输出，并使用线性激活，这是TensorFlow的dense（）函数的默认值：<br>
`logits_layer = tf.layers.dense(inputs=dense_layer, units=l0)`<br>
logits图层的最终输出将是以下的张量：<br>
[batch_size, l0]<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/17.jpg) <br>
如前所述，logits层的模型将返回我们批处理的原始预测。 但我们需要将这些值转换为可解释的格式：
输入样本0-9的预测类。每个可能类的分数或概率。 例如，样本为0的概率为1，依此类推。<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/18.jpg) <br>
因此，我们预测的类将是10个概率中具有最高值的类。 我们可以使用argmax函数获取此值，如下所示：<br>
`tf.argmax(input=logits_layer, axis=l)`
## 卷积神经网络实例 - MNIST数字分类
在本节中，我们将使用MNIST数据集做一个实现CNN数字分类的完整示例。 我们将构建一个包含两个卷积层和全连接层的简单模型。<br>
让我们首先导入这个实现所需的库：<br>
```
%matplotlib inline
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix import math

```
接下来，我们将使用TensorFlow辅助函数来下载和预处理MNIST数据集，如下所示：<br>
```
from tensorflow.examples.tutorials.mnist import input_data mnist_data = input_data.read_data_sets('data/MNI3T/', one_hot=True)

Output:
3uccessfully downloaded train–images–idx3–ubyte.gz 99l2422 bytes. Extracting data/MNI3T/train–images–idx3–ubyte.gz
3uccessfully downloaded train–labels–idxl–ubyte.gz 2888l bytes. Extracting data/MNI3T/train–labels–idxl–ubyte.gz
3uccessfully downloaded tlOk–images–idx3–ubyte.gz l648877 bytes. Extracting data/MNI3T/tlOk–images–idx3–ubyte.gz
3uccessfully downloaded tlOk–labels–idxl–ubyte.gz 4542 bytes. Extracting data/MNI3T/tlOk–labels–idxl–ubyte.gz
```
数据集分为三个不相交的集合：训练，验证和测试。 那么，让我们输出每组中的图像数量：
```
print("– Number of images in the training set:\t\t(}".format(len(mnist_data.train.labels))) 
print("– Number of images in the test set:\t\t(}".format(len(mnist_data.test.labels))) 
print("– Number of images in the validation set:\t(}".format(len(mnist_data.validation.labels)))

–	Number of images in the training set: 55OOO
–	Number of images in the test set: lOOOO
–	Number of images in the validation set: 5OOO
```
图像的实际标签以one-hot编码格式存储，因此我们有一个包含10个零值的数组，除了该图像所代表的类的索引。 为了以后的使用，我们需要将数据集的类号作为整数：
`mnist_data.test.cls_integer = np.argmax(mnist_data.test.labels, axis=l)`
让我们定义一些已知的变量，以便稍后在我们的实现中使用：
```
# Default size for the input monocrome images of MNI3T image_size = 28

# Each image is stored as vector of this size. image_size_flat = image_size * image_size

# The shape of each image
image_shape = (image_size, image_size)

# All the images in the mnist dataset are stored as a monocrome with only l channel
num_channels = l

# Number of classes in the MNI3T dataset from O till 9 which is lO num_classes = lO
```
接下来，我们需要定义一个辅助函数来绘制数据集中的一些图像。 这个辅助函数将在九个子图的网格中绘制图像：
```
def plot_imgs(imgs, cls_actual, cls_predicted=None): 
assert len(imgs) == len(cls_actual) == 9
# create a figure with 9 subplots to plot the images. fig, axes = plt.subplots(3, 3) fig.subplots_adjust(hspace=O.3, wspace=O.3)
for i, ax in enumerate(axes.flat): 
    # plot the image at the ith index 
    ax.imshow(imgs[i].reshape(image_shape), cmap='binary')
    # labeling the images with the actual and predicted classes. 
    if cls_predicted is None:
        xlabel = "True: (O}".format(cls_actual[i]) 
    else:
        xlabel = "True: (O}, Pred: (l}".format(cls_actual[i], cls_predicted[i])
    # Remove ticks from the plot. 
    ax.set_xticks([]) 
    ax.set_yticks([])
    # 3how the classes as the label on the x–axis. 
    ax.set_xlabel(xlabel)
plt.show()

```
让我们从测试集中绘制一些图像并查看它的图像：
```
# Visualizing 9 images form the test set. imgs = mnist_data.test.images[0:9]

# getting the actual classes of these 9 images 
cls_actual = mnist_data.test.cls_integer[0:9]

#plotting the images
plot_imgs(imgs=imgs, cls_actual=cls_actual)
```
这是输出:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/chapter07_image/19.jpg) <br>
## 建立模型
现在，是时候构建模型的核心了。 计算图包括我们在本章前面提到的所有层。 我们首先定义一些函数，这些函数将用于定义特定形状的变量并随机初始化它们：
```
def new_weights(shape):
return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
return tf.Variable(tf.constant(0.05, shape=[length]))
```
现在，让我们根据一些输入层，输入通道，过滤器大小，过滤器数量以及是否使用池化参数来定义将负责创建新卷积层的函数：
```
def conv_layer(input, # the output of the previous layer.
input_channels, filter_size, filters,
use_pooling=True): # Use 2x2 max–pooling.

# preparing the accepted shape of the input Tensor.
shape = [filter_size, filter_size, input_channels, filters]

# Create weights which means filters with the given shape. filters_weights = new_weights(shape=shape)

# Create new biases, one for each filter. filters_biases = new_biases(length=filters)

# Calling the conve2d function as we explained above, were the strides parameter
# has four values the first one for the image number and the last l for the input image channel
# the middle ones represents how many pixels the filter should move with in the x and y axis
conv_layer = tf.nn.conv2d(input=input,
filter=filters_weights, strides=[l, l, l, l], padding='3AME')

# Adding the biase to the output of the conv_layer. conv_layer += filters_biases

# Use pooling to down–sample the image resolution? if use_pooling:
 

# reduce the output feature map by max_pool layer pool_layer = tf.nn.max_pool(value=conv_layer,
ksize=[l, 2, 2, l],
strides=[l, 2, 2, l], padding='3AME')

# feeding the output to a ReLU activation function. relu_layer = tf.nn.relu(pool_layer)

# return the final results after applying relu and the filter weights return relu_layer, filters_weights
```
如前所述，汇集层产生4D张量。 我们需要将这个4D张量展平为2D，以传送到完全连接层：
```
def flatten_layer(layer):
# Get the shape of layer. shape = layer.get_shape()

# We need to flatten the layer which has the shape of The shape
[num_images, image_height, image_width, num_channels]
# so that it has the shape of [batch_size, num_features] where 
number_features is image_height * image_width * num_channels

number_features = shape[l:4].num_elements()
# Reshaping that to be fed to the fully connected layer flatten_layer = tf.reshape(layer, [–l, number_features])
# Return both the flattened layer and the number of features. return flatten_layer, number_features
```
此函数创建一个完全连接的层，假设输入是2D张量：
```
def fc_layer(input, # the flatten output.
num_inputs, # Number of inputs from previous layer num_outputs, # Number of outputs
use_relu=True): # Use ReLU on the output to remove
negative values

# Creating the weights for the neurons of this fc_layer fc_weights = new_weights(shape=[num_inputs, num_outputs]) fc_biases = new_biases(length=num_outputs)

# Calculate the layer values by doing matrix multiplication of
# the input values and fc_weights, and then add the fc_bias–values. fc_layer = tf.matmul(input, fc_weights) + fc_biases
 

# if use RelU parameter is true if use_relu:
relu_layer = tf.nn.relu(fc_layer) return relu_layer

return fc_layer
```
在构建网络之前，让我们为输入图像定义一个占位符，其中第一个维度为None，表示任意数量的图像：
`input_values = tf.placeholder(tf.float32, shape=[None, image_size_flat], name='input_values')`
如前所述，卷积步骤要求输入图像为4D张量的形状。因此，我们需要将输入图像重塑为以下形状：
`[num_images，image_height，image_width，num_channels]`
所以，让我们重新整形输入值以匹配这种格式：
`input_image = tf.reshape（input_values，[ - l，image_size，image_size，num_channels]）`
接下来，我们需要为实际的类值定义另一个占位符，它将采用one hot编码格式：
`y_actual = tf.placeholder（tf.float32，shape = [None，num_classes]，name ='y_actual'）`
此外，我们需要定义一个占位符来保存实际类的整数值：
`y_actual_cls_integer = tf.argmax（y_actual，axis = l）`
那么，让我们从建立第一个CNN开始：
```
conv_layer_l，convl_weights = \ conv_layer（input = input_image，
input_channels = num_channels，filter_size = filter_size_l，filters = filters_l，use_pooling = True）
```
让我们检查将由第一个卷积层产生的输出张量的形状：
```
conv_layer_l
output: 
<tf.Tensor'Ruu：relu:o’ shape =（1，l4，l4，l6）dtype = float32>
```
接下来，我们将创建第二个卷积网络，并将第一个卷积网络的输出提供给它：
`conv_layer_2，conv2_weights = \ conv_layer（input = conv_layer_l，input_channels = filters_l，filter_size = filter_size_2，filters = filters_2，use_pooling = True）`
此外，我们需要仔细检查第二个卷积层的输出张量的形状。形状应该是（1，7,7,36），其中？标记表示任意数量的图像。<br>
接下来，我们需要展平4D张量以匹配完全连接层的预期格式，这是一个2D张量：
`flatten_layer，number_features = flatten_layer（conv_layer_2）`
我们需要仔细检查展平层的输出张量的形状：
```
flatten_layer输出：
<tf.Tensor'Reshape_l：0'形=（1，l764）dtype = float32>
```
接下来，我们将创建一个完全连接的图层，并将展平图层的输出提供给它。我们还将完全连接层的输出馈送到ReLU激活功能，然后将其馈送到第二个完全连接的层：
```
fc_layer_l = fc_layer（input = flatten_layer，
num_inputs = number_features，num_outputs = fc_num_neurons，use_relu = True）
```
让我们仔细检查第一个完全连接层的输出张量的形状：
```
fc_layer_l输出：
<tf.Tensor'Rueu_2：0'形状=（1，l28）dtype = float32>
```
接下来，我们需要添加另一个完全连接的层，它将获取第一个完全连接的层的输出，并为每个图像生成一个大小为10的数组，表示每个目标类的得分是正确的：
```
fc_layer_2 = fc_layer（input = fc_layer_l，
num_inputs = fc_num_neurons，num_outputs = num_classes，use_relu = False）
fc_layer_2输出：
<tf.Tensor'add_3：0'shape =（1，l0）dtype = float32>
```
接下来，我们将从第二个完全连接的层中标准化这些分数并将其提供给
softmax激活函数，将值压缩到0到1之间：
`y_predicted = tf.nn.softmax（fc_layer_2）`
然后，我们需要通过使用选择具有最高概率的目标类
TensorFlow的argmax功能：
`y_predicted_cls_integer = tf.argmax（y_predicted，axis = l）`
### 代价函数
接下来，我们需要定义我们的性能度量，即交叉熵。 如果预测的类是正确的，则交叉熵的值将为0：
`cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer_2,
labels=y_actual)`
接下来，我们需要将从上一步得到的所有交叉熵值求平均数，以便能够在测试集上获得单一的性能指标：
`model_cost = tf.reduce_mean(cross_entropy)`
现在，我们有一个需要优化/最小化的代价函数，因此我们将使用AdamOptimizer，这是一种优化方法，和梯度下降一样但更加优：
`model_optimizer = tf.train.AdamOptimizer(learning_rate=le–4).minimize(model_cost)`
### 绩效测评
为了显示输出，让我们定义一个变量来检查预测的类是否等于真实的类：
`model_correct_prediction = tf.equal(y_predicted_cls_integer, y_actual_cls_integer)`
通过转换布尔值来计算模型精度，然后对它们求平均值,以此对正确分类的值求和：
`model_accuracy = tf.reduce_mean(tf.cast(model_correct_prediction, tf.float32))`
### 训练集
让我们通过创建一个会话变量启动训练过程，该变量将负责执行我们之前定义的计算图：
`session = tf.3ession()`
此外，我们需要初始化到目前为止定义的变量：
`session.run(tf.global_variables_initializer())`
我们将分批提供图像以避免内存不足错误：
`train_batch_size = 64`
在开始训练过程之前，我们将定义一个辅助函数，通过迭代训练批来执行优化过程：
```
# number of optimization iterations performed so far total_iterations = O

def optimize(num_iterations):
# Update globally the total number of iterations performed so far.
 

global total_iterations

for i in range(total_iterations,
total_iterations + num_iterations):

 


and batch.
 
# Generating a random batch for the training process
# input_batch now contains a bunch of images from the training set
# y_actual_batch are the actual labels for the images in the input input_batch, y_actual_batch =
 
mnist_data.train.next_batch(train_batch_size)

# Putting the previous values in a dict format for Tensorflow to automatically assign them to the input
# placeholders that we defined above feed_dict = (input_values: input_batch,
y_actual: y_actual_batch}

# Next up, we run the model optimizer on this batch of images session.run(model_optimizer, feed_dict=feed_dict)

# Print the training status every lOO iterations. if i % lOO == O:
# measuring the accuracy over the training set. acc_training_set = session.run(model_accuracy,
feed_dict=feed_dict)
#Printing the accuracy over the training set print("Iteration: (O:>6}, Accuracy Over the training set:
(l:>6.l%}".format(i + l, acc_training_set))

# Update the number of iterations performed so far total_iterations += num_iterations
```
我们将定义一些辅助函数来帮助我们可视化模型的结果，并查看哪些图像被模型错误分类：
```
def plot_errors(cls_predicted, correct):
# cls_predicted is an array of the predicted class number of each image in the test set.


# Extracting the incorrect images. incorrect = (correct == False)
# Get the images from the test–set that have been
# incorrectly classified.
images = mnist_data.test.images[incorrect]
# Get the predicted classes for those incorrect images.
 

cls_pred = cls_predicted[incorrect]

# Get the actual classes for those incorrect images. cls_true = mnist_data.test.cls_integer[incorrect]
# Plot 9 of these images plot_imgs(imgs=imgs[0:9],
cls_actual=cls_actual[O:9], cls_predicted=cls_predicted[0:9])
```
我们还可以绘制预测结果与实际真实类别的混淆矩阵：
```
def plot_confusionMatrix(cls_predicted):

# cls_predicted is an array of the predicted class number of each image in the test set.

# Get the actual classes for the test–set. cls_actual = mnist_data.test.cls_integer

# Generate the confusion matrix using sklearn. conf_matrix = confusion_matrix(y_true=cls_actual, y_pred=cls_predicted)

# Print the matrix. print(conf_matrix)

# visualizing the confusion matrix. plt.matshow(conf_matrix)

plt.colorbar()
tick_marks = np.arange(num_classes) plt.xticks(tick_marks, range(num_classes)) plt.yticks(tick_marks, range(num_classes)) plt.xlabel('Predicted class') plt.ylabel('True class')

# 3howing the plot plt.show()
```
最后，我们将定义一个辅助函数来帮助我们测量训练模型在测试集上的准确性：
```
# measuring the accuracy of the trained model over the test set by splitting it into small batches
test_batch_size = 256

def test_accuracy(show_errors=False,
 

show_confusionMatrix=False):

#number of test images
number_test = len(mnist_data.test.images)

# define an array of zeros for the predicted classes of the test set which
# will be measured in mini batches and stored it. cls_predicted = np.zeros(shape=number_test, dtype=np.int)

# measuring the predicted classes for the testing batches.

# 3tarting by the batch at index O. i = O

while i < number_test:
# The ending index for the next batch to be processed is j. j = min(i + test_batch_size, number_test)

# Getting all the images form the test set between the start and end indices
input_images = mnist_data.test.images[i:j, :]

# Get the acutal labels for those images. actual_labels = mnist_data.test.labels[i:j, :]

# Create a feed–dict with the corresponding values for the input placeholder values
feed_dict = (input_values: input_images,
y_actual: actual_labels}

cls_predicted[i:j] = session.run(y_predicted_cls_integer, feed_dict=feed_dict)

# 3etting the start of the next batch to be the end of the one that we just processed j
i = j

# Get the actual class numbers of the test images. cls_actual = mnist_data.test.cls_integer

# Check if the model predictions are correct or not correct = (cls_actual == cls_predicted)

# 3umming up the correct examples correct_number_images = correct.sum()

# measuring the accuracy by dividing the correclty classified ones with
 

total number of images in the test set.
testset_accuracy = float(correct_number_images) / number_test

# showing the accuracy.
print("Accuracy on Test–3et: (O:.l%} ((l} / (2})".format(testset_accuracy, correct_number_images, number_test))

# showing some examples form the incorrect ones. if show_errors:
print("Example errors:") plot_errors(cls_predicted=cls_predicted, correct=correct)

# 3howing the confusion matrix of the test set predictions if show_confusionMatrix:
print("Confusion Matrix:") plot_confusionMatrix(cls_predicted=cls_predicted)
```
