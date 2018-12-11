
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
![image004](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter13/chapter13_image/image004.png)<br>
图4：来自MNIST数据集的示例图像<br>
```#  Plotting one image from the training set.
image = mnist_dataset.train.images[2]
plt.imshow(image.reshape((28, 28)), cmap='Greys_r')

Output:
```
![image005](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter13/chapter13_image/image005.png)<br>
图5：来自MNIST数据集的示例图像
## 建立模型<br>
&emsp;&emsp;为了建立编码器，我们需要计算出每个MNIST图像将有多少像素，以便我们能够计算出编码器的输入层的大小。来自MNIST数据集的每幅图像都是28乘28像素，因此我们将将这个矩阵重塑为28x 28=784像素值的向量。我们不必将MNIST的图像标准化，因为它们已经标准化了。<br>
&emsp;&emsp;让我们开始构建模型的三个组成部分。在这个实现中，我们将使用单个隐藏层的非常简单的体系结构，然后是ReLU激活，如下图所示：<br>
![image006](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter13/chapter13_image/image006.png)<br>
图6：MNIST实现的编码器-解码器体系结构<br>
&emsp;&emsp;让我们继续并根据前面的解释实现这个简单的编码器-解码器架构：<br>
```
# The size of the encoding layer or the hidden layer. encoding_layer_dim = 32

img_size = mnist_dataset.train.images.shape[l]

# defining placeholder variables of the input and target values inputs_values = tf.placeholder(tf.float32, (None, img_size), name="inputs_values")
targets_values = tf.placeholder(tf.float32, (None, img_size), name="targets_values")

# Defining an encoding layer which takes the input values and incode them. encoding_layer = tf.layers.dense(inputs_values, encoding_layer_dim, activation=tf.nn.relu)

# Defining the logit layer, which is a fully–connected layer but without any activation applied to its output
logits_layer = tf.layers.dense(encoding_layer, img_size, activation=None)

# Adding a sigmoid layer after the logit layer
decoding_layer = tf.sigmoid(logits_layer, name = "decoding_layer")

# use the sigmoid cross entropy as a loss function
model_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_layer, labels=targets_values)

# Averaging the loss values accross the input data model_cost = tf.reduce_mean(model_loss)

# Now we have a cost functiont that we need to optimize using Adam Optimizer
model_optimizier = tf.train.AdamOptimizer().minimize(model_cost)
```
现在我们已经定义了我们的模型，并且还使用了二进制交叉熵，因为图像、像素已经标准化了。<br>
## 模型训练<br>
&emsp;&emsp;在这一节中，我们将开始训练过程。我们将使用mnist_dataset对象的helper函数，以便从具有特定大小的数据集中获取随机批次；然后，我们将对这批图像运行优化器。让我们通过创建会话变量来开始本节，会话变量将负责执行前面定义的计算图：<br>
`# creating the sessionsess = tf.3ession()`<br>
下一步，让我们开始训练过程：<br>
```num_epochs = 2O
train_batch_size = 2OO

sess.run(tf.global_variables_initializer())
for e in range(num_epochs):
for ii in range(mnist_dataset.train.num_examples//train_batch_size):
input_batch = mnist_dataset.train.next_batch(train_batch_size)
feed_dict = (inputs_values: input_batch[O], targets_values:
input_batch[O]}
 input_batch_cost, _ = sess.run([model_cost, model_optimizier],
feed_dict=feed_dict)

print("Epoch: (}/(}...".format(e+l, num_epochs),
    "Training loss: (:.3f}".format(input_batch_cost))
Output:
Epoch: 2O/2O... Training loss: O.O9l
Epoch: 2O/2O... Training loss: O.O9l
Epoch: 2O/2O... Training loss: O.O93
Epoch: 2O/2O... Training loss: O.O93
Epoch: 2O/2O... Training loss: O.O95
Epoch: 2O/2O... Training loss: O.O95
Epoch: 2O/2O... Training loss: O.O89
Epoch: 2O/2O... Training loss: O.O95
Epoch: 2O/2O... Training loss: O.O95
Epoch: 2O/2O... Training loss: O.O96
Epoch: 2O/2O... Training loss: O.O94
Epoch: 2O/2O... Training loss: O.O93
Epoch: 2O/2O... Training loss: O.O94
Epoch: 2O/2O... Training loss: O.O93
Epoch: 2O/2O... Training loss: O.O95
Epoch: 2O/2O... Training loss: O.O94
Epoch: 2O/2O... Training loss: O.O96
Epoch: 2O/2O... Training loss: O.O92
Epoch: 2O/2O... Training loss: O.O93
Epoch: 2O/2O... Training loss: O.O9l
Epoch: 2O/2O... Training loss: O.O93
Epoch: 2O/2O... Training loss: O.O9l
Epoch: 2O/2O... Training loss: O.O95
Epoch: 2O/2O... Training loss: O.O94
Epoch: 2O/2O... Training loss: O.O9l
```
现在，我们已经训练该模型能够产生无噪声图像，这使得自动编码器适用于许多领域。<br>
&emsp;&emsp;在下一段代码中，我们不会将MNIST测试集的行图像提供给模型，因为我们需要首先向这些图像添加噪声，以查看经过训练的模型将如何能够生成无噪声的图像。这里，我将向测试图像添加噪声，并将它们传递到自动编码器。尽管有时很难分辨原始数字是什么，但它在消除噪音方面做得非常出色：<br>
```#Defining some figures
fig, axes = plt.subplots(nrows=2, ncols=lO, sharex=True, sharey=True,
figsize=(2O,4))

#Visualizing some images
input_images = mnist_dataset.test.images[:lO]
noisy_imgs = input_images + mnist_noise_factor
np.random.randn(*input_images.shape)

#Clipping and reshaping the noisy images
noisy_images = np.clip(noisy_images, O., l.).reshape((lO, 28, 28, l))
#Getting the reconstructed images
reconstructed_images = sess.run(decoding_layer, feed_dict=(inputs_values: noisy_images})

#Visualizing the input images and the noisy ones
for imgs, row in zip([noisy_images, reconstructed_images], axes):
for img, ax in zip(imgs, row):
ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=O.l)
Output:
```
![image007](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter13/chapter13_image/image007.png)<br>
图07_带有一些高斯噪声(顶行)的原始测试图像的示例及其基于训练好的去噪自动编码器的构造<br>
## 自动编码器的应用<br>
&emsp;&emsp;在前面的从低级表示构造图像的示例中，我们看到它与原始输入非常相似，并且我们还看到了CAN在去噪数据集时的好处。上述例子对于图像构造和数据集去噪非常有用。因此，可以将上述实现推广到您感兴趣的任何其他示例。<br>
&emsp;&emsp;此外，在本章中，我们已经看到了自动编码器的灵活性。建筑是什么，我们如何能做出不同的改变。我们甚至测试了它来解决从输入图像中去除噪声的难题。这种灵活性为auoencoder非常适合的更多应用程序打开了大门。<br>
## 图像彩色化<br>
&emsp;&emsp;“自然界的人——具体说的是一个多方面的版本——可以用面值计算。在下面的示例中，我们向模型提供没有任何颜色的输入图像，并且该图像的重建版本将通过自动编码器模型着色：<br>
![image008](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter13/chapter13_image/image008.png)<br>
图08：训练CAE以使图像着色<br>
![image009](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter13/chapter13_image/image009.png)<br>
图009：colorization文件架构<br>
现在我们的自动编码器已经训练好了，我们可以用它来给以前从未见过的图片上色！<br>
这种应用可以用来对早期照相机拍摄的非常古老的图像进行着色。<br>
## 更多应用<br>
&emsp;&emsp;另一个有趣的应用可以是产生更高分辨率的图像，或神经图像增强，如下图所示。这些数字显示了张理查的图像着色更逼真的版本：<br>
![image010](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter13/chapter13_image/image010.png)<br>
图15_彩色图像着色<br>
此图显示了自动编码器的另一个应用，用于进行图像增强：<br>
![image011](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter13/chapter13_image/image011.png)<br>
图11_神经增强bμAlexjc <br>
## 总结<br>
&emsp;&emsp;在本章中，我们介绍了一个全新的体系结构，可用于许多有趣的应用程序。自动编码器非常灵活，所以您可以在图像增强、着色或构造方面提出自己的问题。此外，自动编码器有更多的变化，称为变分自动编码器。它们也被用于非常有趣的应用，例如图像生成。<br>


学号|姓名|专业
-|-|-
201802110486|章志鹏|计算机应用技术
<br>

