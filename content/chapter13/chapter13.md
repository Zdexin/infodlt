
# &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;第13章 自动编码器-特征提取和去噪
&emsp;&emsp;自动编码器网络现在是广泛使用的深度学习架构之一。 它主要用于高效解码任务的无监督学习。 它还可以通过学习特定数据集的编码或表示用于来降低维数。 在本章中使用自动编码器，我们将展示如何通过构建具有相同尺寸但噪声较小的另一个数据集来对数据集进行去噪。 为了在实践中使用这个概念，我们将从MNIST数据集中提取重要特征，并试着看看如何通过这个显着增强性能。<br>
&emsp;&emsp;&emsp;本章将介绍以下主题：<br>
&emsp;&emsp;&emsp;&emsp;---自动编码器简介<br>
&emsp;&emsp;&emsp;&emsp;---自动编码器实例<br>
&emsp;&emsp;&emsp;&emsp;---自动编码器架构<br>
&emsp;&emsp;&emsp;&emsp;---压缩MNIST数据集<br>
&emsp;&emsp;&emsp;&emsp;---卷积自动编码器<br>
&emsp;&emsp;&emsp;&emsp;---去噪自动编码器<br>
&emsp;&emsp;&emsp;&emsp;---自动编码器的应用<br>
## 自动编码器简介<br>
&emsp;&emsp;自动编码器是另一种可用于许多有趣任务的深度学习架构，但它也可以被视为香草前馈神经网络的变体，其中输出具有与输入相同的维数。如图1所示，自动编码器的工作方式是将数据样本（x1，...，x6）提供给网络。它将尝试在L2层中学习此数据的较低表示，您可以将其称为以较低表示形式对数据集进行编码的方法。然后，网络的第二部分（可称为解码器）负责构造此表示的输出。您可以将网络从输入数据中学习的中间较低表示视为其压缩版本。<br>
&emsp;&emsp;与我们迄今为止看到的所有其他深度学习架构没有太大差别，自动编码器使用反向传播算法。自动编码器神经网络是一种适用的无监督学习算法反向传播，将目标值设置为等于输入：<br>
![image001](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter13/chapter13_image/image001.png)<br>
图1 通用自动编码器架构<br>
## 自动编码器示例<br>
&emsp;&emsp;在本章中，我们将演示使用MNIST数据集的自动编码器的不同变体的一些示例。作为具体示例，假设输入x是来自28×28图像（784像素）的像素强度值; 所以输入数据样本的数量是n = 784。 L2层中有s2 = 392个隐藏单位。 并且由于输出将与输入数据样本具有相同的维度，因此y z R784。输入层神经元数为784，中层神经元数为392；因此，网络将是更低的表示，它是输出的压缩版本。然后，网络将把输入a(L2)z R392的压缩较低表示馈送给网络的第二部分，后者将努力从这个压缩版本重建输入像素784。<br>
&emsp;&emsp;自动编码器依赖于这样的事实，即由图像像素表示的输入样本将以某种方式相关，然后它将使用这个事实来重建它们。因此，自动编码器有点类似于降维技术，因为它们还学习了输入数据的较低表示。<br>
&emsp;&emsp;&emsp;综上所述，一个典型的自动编码器将由三部分组成：<br>
&emsp;&emsp;&emsp;&emsp;1.编码器部分，其负责将输入压缩为较低表示<br>
&emsp;&emsp;&emsp;&emsp;2.代码，这是编码器的中间结果<br>
&emsp;&emsp;&emsp;&emsp;3.解码器，负责使用该代码重构原始输入<br>
下图显示了一个典型的自动编码器的三个主要组成部分：<br>
![image002](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter13/chapter13_image/image002.png)<br>
图2：编码器如何在图像上起作用<br>
正如我们提到的，自动编码器部分学习输入的压缩表示，然后提供给第三部分，第三部分试图重构输入。重构的输入将类似于输出，但是它不会完全与原始输出相同，因此自动编码器不能用于压缩任务。<br>
## 自动编码器架构<br>
&emsp;&emsp;正如我们所提到的，一个典型的自动编码器由三部分组成。让我们更详细地探讨这三个部分。为了激励你，我们不会在本章中重蹈覆辙。编解码器部分只是一个完全连接的神经网络，而代码部分是另一个神经网络，但它不是完全连接的。这部分代码的维数是可控的，我们可以把它当作超参数：<br>
![image003](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter13/chapter13_image/image003.png)<br>
图3：自动编码器的通用编码器解码器结构<br>
在深入使用自动编码器来压缩MNIST数据集之前，我们将列出可用于微调自动编码器模型的超参数集。主要有四个超参数：<br>
&emsp;&emsp;&emsp;1代码部分规模：这是中间层的单位数。这个层中的单元数量越少，得到的输入表示就越压缩。<br>
&emsp;&emsp;&emsp;2编码器和解码器中的层数：正如我们提到的，编码器和解码器只是一个完全连接的神经网络，我们可以通过增加更多的层来尽可能地加深。<br>
&emsp;&emsp;&emsp;3每层单位数：我们也可以使用不同数量的单位在每一层。编码器和解码器的形状非常类似于DeconvNets，其中编码器中的层数随着我们接近代码部分而减少，然后随着我们接近解码器的最后一层而开始增加。<br>
&emsp;&emsp;&emsp;4模型损失函数：我们可以使用不同的损失函数，如MSE或交叉熵。<br>
在定义这些超参数并给出它们的初始值之后，我们可以使用反向传播算法来训练网络。<br>
## 压缩MNIST数据集<br>
&emsp;&emsp;在本部分中，我们将构建一个简单的自动编码器，用于压缩MNIST数据集。因此，我们将把数据集的图像提供给编码器部分，编码器部分将尝试学习它们的较低压缩表示；然后我们将尝试在解码器部分再次构造输入图像。
## MNIST数据集<br>
&emsp;&emsp;我们将使用TensorFlow的帮助函数通过获取MNIST数据集开始实现。<br>
&emsp;&emsp;让我们为这个实现导入必要的包：<br>
```%matplotlib inline

import numpy as np import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist_dataset = input_data.read_data_sets('MNI3T_data', validation_size=O)

Output:
Extracting  MNI3T_data/train–images–idx3–ubyte.gz
Extracting  MNI3T_data/train–labels–idxl–ubyte.gz
Extracting  MNI3T_data/tlOk–images–idx3–ubyte.gz
Extracting  MNI3T_data/tlOk–labels–idxl–ubyte.gz
```
让我们从MNIST数据集的一些例子开始：
```#  Plotting one image from the training set.
image = mnist_dataset.train.images[2]
plt.imshow(image.reshape((28, 28)), cmap='Greys_r')

Output:
```


