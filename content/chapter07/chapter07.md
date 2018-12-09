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
[batch_size, 28, 28, 2O]<br>
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
[batch_size, 28, 28, 2O]<br>
池化操作的输出将具有以下形状：<br>
[batch_size, l4, l4, 2O]<br>
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

