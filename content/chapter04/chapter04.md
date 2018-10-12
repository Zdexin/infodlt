
# 第四章
## 使用和运行TensorFlow

&emsp;&emsp;在本章中，我们将概述最广泛使用的深度学习框架之一。TensorFlow拥有越来越多的社区支持，这使得它成为一个很好的选择。 用于构建复杂的深度学习应用程序。来自TensorFlow网站：<br>

&emsp;&emsp;“TensorFlow是一个使用数据流图进行数值计算的开源软件库，图中的节点表示数学操作，而图的边缘表示多维的操作。 Nal数据数组(张量)在它们之间通信。灵活的体系结构允许您使用单个API将计算部署到桌面、服务器或移动设备中的一个或多个CPU或GPU。TensorFlow最初是由谷歌机器智能研究组织内的谷歌大脑团队的研究人员和工程师开发的，目的是进行机器学习和深层神经网络的研究，但该系统的通用性足以适用于广泛的其他领域。”<br>
&emsp;&emsp;以下主题将在本章中讨论:<br>
&emsp;&emsp;&emsp;&emsp;---TensorFlow 安装<br>
&emsp;&emsp;&emsp;&emsp;---The TensorFlow环境 <br>
&emsp;&emsp;&emsp;&emsp;---计算图<br>
&emsp;&emsp;&emsp;&emsp;---TensorFlow数据类型, 变量,和占位符<br>
&emsp;&emsp;&emsp;&emsp;---从 TensorFlow获得输出 <br>
&emsp;&emsp;&emsp;&emsp;---TensorBoard—可视化学习<br>

### TensorFlow 安装
&emsp;&emsp;TensorFlow安装有两种模式：CPU和GPU。我们将从安装教程开始，在GPU模式下安装TensorFlow。

#### 用于Ubuntu 16.04的TensorFlow GPU安装
&emsp;&emsp;GPU模式安装TensorFlow需要最新的安装NVIDIA驱动程序，因为GPU版本的TensorFlow目前只支持CUDA。下一节将 通过一个逐步安装NVIDIA驱动程序和CUDA8的过程。<br>

#### 安装NVIDIA驱动程序和CUDA8
&emsp;&emsp;首先，您需要根据您的GPU安装正确的NVIDIA驱动程序。我有一个GeForce GTX 960m GPU，所以我会继续安装NVIDIA-375(如果你有不同的GPU，你可以使用NVIDI。) 一个搜索工具http：/www.nvidia.com/下载/index.aspx 帮助您找到正确的驱动程序版本。如果您想知道机器的GPU，可以在终端中发出以下命令：<br>

`lspci | grep –i nvidia`

&emsp;&emsp;您应该在终端中获得以下输出：<br>
![image1]()

&emsp;&emsp;接下来，我们需要添加一个NVIDIA驱动程序的专有存储库，以便能够使用APT-GET安装驱动程序：<br>

```sudo add–apt–repository ppa:graphics–drivers/ppa sudo apt–get update
sudo apt–get install nvidia–375
```

&emsp;&emsp;成功安装NVIDIA驱动程序后，重新启动机器。若要验证驱动程序安装是否正确，请在终端中发出以下命令：<br>

`cat /proc/driver/nvidia/version`


&emsp;&emsp;您应该在终端中获得以下输出:<br>
![image1]()

&emsp;&emsp;接下来，我们需要安装CUDA 8。打开以下CUDA下载链接：https：/developer.nvidia.com/Cuda-下载。选择您的操作系统、体系结构、发行版、版本，最后，在 Staller类型如下所示：<br>
![image1]()

&emsp;&emsp;安装程序文件大约是2GB。您需要发出以下安装说明：<br>
```sudo dpkg –i cuda–repo–ubuntul6O4–8–O–local–ga2_8.O.6l–l_amd64.deb sudo apt–get update
sudo apt–get install cuda
&emsp;&emsp;接下来，我们需要通过发出以下命令将库添加到.bashrc文件中：<br>
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >>
~/.bashrc

source ~/.bashrc

Next, you need to verify the installation of CUDA 8 by issuing the following command:
nvcc –V
```
&emsp;&emsp;您应该在终端中获得以下输出：<br>
![image1]()

&emsp;&emsp;最后，在本节中，我们需要安装cuDNN 6.0。NVIDIA CUDA深层神经网络库(CuDNN)是一个用于深层神经网络的GPU加速基元库.你可以从 mNVIDIA的网页。发出以下命令以提取和安装cuDNN：<br>

```cd ~/Downloads/ tar xvf cudnn*.tgz cd cuda
sudo cp */*.h /usr/local/cuda/include/ sudo cp */libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```
&emsp;&emsp;为了确保安装成功，您可以在终端中使用Nvidia-SMI工具。如果安装成功，此工具将为您提供监视信息。CHAS RAM和您的GPU的运行过程。<br>

#### 安装 TensorFlow
&emsp;&emsp;在为TensorFlow准备了GPU环境之后，我们现在可以在GPU模式下安装TensorFlow了。但是要通过TensorFlow安装过程，您可以首先安装一些有用的P ython包，将在下一章中帮助您，并使您的开发环境更容易。<br>
&emsp;&emsp;首先，我们可以通过发出以下命令来安装一些数据操作、分析和可视化库：<br>

```sudo apt–get update && apt–get install –y python–numpy python–scipy python– nose python–h5py python–skimage python–matplotlib python–pandas python– sklearn python–sympy

sudo apt–get clean && sudo apt–get autoremove sudo rm –rf /var/lib/apt/lists/*
```
&emsp;&emsp;接下来，您可以安装更多有用的库，例如虚拟环境、木星笔记本等等：<br>

```sudo apt–get update

sudo apt–get install git python–dev python3–dev python–numpy python3–numpy build–essential	python–pip python3–pip python–virtualenv swig python–wheel libcurl3–dev

sudo apt–get install –y libfreetype6–dev libpngl2–dev

pip3 install –U matplotlib ipython[all] jupyter pandas scikit–image
```
&emsp;&emsp;最后，我们可以通过发出以下命令，在GPU模式下开始安装TensorFlow:<br>
`pip3 install ––upgrade tensorflow–gpu`

&emsp;&emsp;您可以使用Python验证TensorFlow的成功安装：<br>
```python3
>>> import tensorflow as tf
>>> a = tf.constant(5)
>>> b = tf.constant(6)
>>> sess = tf.3ession()
>>> sess.run(a+b)
// this should print bunch of messages showing device status etc. // If everything goes well, you should see gpu listed in device
>>> sess.close()
```

&emsp;&emsp;您应该在终端中获得以下输出：<br>
![image1]()

#### 用于Ubuntu 16.04的TensorFlow CPU安装
&emsp;&emsp;在本节中，我们将安装CPU版本，它在安装之前不需要任何驱动程序。因此，让我们从安装一些有用的数据操作包和visu开始 dualization对偶（化），二元化，复线化：<br>

```sudo apt–get update && apt–get install –y python–numpy python–scipy python– nose python–h5py python–skimage python–matplotlib python–pandas python– sklearn python–sympy

sudo apt–get clean && sudo apt–get autoremove sudo rm –rf /var/lib/apt/lists/*
&emsp;&emsp;接下来，您可以安装更多有用的库，例如虚拟环境、木星笔记本等等：<br>

sudo apt–get update

sudo apt–get install git python–dev python3–dev python–numpy python3–numpy build–essential	python–pip python3–pip python–virtualenv swig python–wheel

libcurl3–dev

sudo apt–get install –y libfreetype6–dev libpngl2–dev

pip3 install –U matplotlib ipython[all] jupyter pandas scikit–image

&emsp;&emsp;最后，通过发出以下命令，可以在CPU模式下安装最新的TensorFlow：<br>

pip3 install ––upgrade tensorflow

&emsp;&emsp;您可以检查是否成功安装了TensorFlow，并运行了以下TensorFlow语句：<br>

python3
>>> import tensorflow as tf
>>> a = tf.constant(5)
>>> b = tf.constant(6)
>>> sess = tf.3ession()
>>> sess.run(a+b)
>> sess.close()
```
&emsp;&emsp;您应该在终端中获得以下输出：<br>
![image1]()

#### 用于MacOS X的TensorFlow CPU安装
&emsp;&emsp;在本节中，我们将使用Virtualenv为MacOSX安装TensorFlow。因此，让我们首先通过发出以下命令来安装pip工具：<br>
`sudo easy_install pip`

&emsp;&emsp;接下来，我们需要安装虚拟环境库：<br>
`sudo pip install ––upgrade virtualenv`

&emsp;&emsp;安装虚拟环境库后，我们需要创建一个容器或虚拟环境，该容器或虚拟环境将承载TensorFlow的安装和您希望安装wi的任何包。 影响底层主机系统的因素：<br>
```virtualenv ––system–site–packages targetDirectory # for Python 2.7

virtualenv ––system–site–packages –p python3 targetDirectory # for Python 3.n
```
&emsp;&emsp;这假设目标目录是~/TensorFlow。<br>

&emsp;&emsp;现在您已经创建了虚拟环境，您可以通过发出以下命令来访问它：<br>

`source ~/tensorflow/bin/activate`

&emsp;&emsp;发出此命令后，您将访问刚刚创建的虚拟机，并且可以安装只安装在此环境中且不会影响u的任何包。 你正在使用的隐藏系统或主机系统。<br>

&emsp;&emsp;为了退出环境，可以发出以下命令：<br>
`deactivate`

&emsp;&emsp;请注意，就目前而言，我们确实希望在虚拟环境中，所以现在打开它。一旦您玩完TensorFlow，就应该禁用它：<br>

`source bin/activate`

&emsp;&emsp;为了安装TensorFlow的CPU版本，您可以发出以下命令，这些命令还将安装TensorFlow所需的任何依赖库：<br>

`(tensorflow)$ pip install ––upgrade tensorflow	# for Python 2.7 (tensorflow)$ pip3 install ––upgrade tensorflow	# for Python 3.n`

#### 用于Windows的TensorFlow GPU/CPU安装
&emsp;&emsp;我们假设您的系统上已经安装了Python 3。要安装TensorFlow，请以管理员身份启动终端，如下所示。打开“开始”菜单，搜索cmd，然后右键并单击“以管理员身份运行”：<br>
![image1]()

&emsp;&emsp;打开命令窗口后，可以发出以下命令，以在GPU模式下安装TensorFlow：<br>
![image1]() 在发出下一个命令之前，需要安装pip或PIP 3(取决于您的Python版本)。<br>
```C:\> pip3 install ––upgrade tensorflow–gpu

Issue the following command to install TensorFlow in CPU mode:
C:\> pip3 install ––upgrade tensorflow
```
### The TensorFlow 环境
&emsp;&emsp;TensorFlow是Google的另一个深度学习框架，正如TensorFlow所暗示的那样，它来自于神经网络在多维数据数组或张量上执行的操作！实际上是张量的流动。<br>
&emsp;&emsp;但是首先，为什么我们要在这本书中使用一个深层次的学习框架呢？<br>
&emsp;&emsp;它扩展了机器学习代码：大部分关于深入学习和机器学习的研究都可以被应用/归因于这些深度学习框架。他们允许数据科学家 迭代速度非常快，使得深入学习和其他ML算法对从业人员来说更加容易获得。像google、facebook等大公司正在使用这样的深度学习框架来扩展到数十亿用户。<br>
&emsp;&emsp;它计算梯度：深度学习框架也可以自动计算梯度。如果你一步地进行梯度计算，你就会发现梯度计算不是小事一桩。 我也很难自己实现一个没有bug的版本。<br>
&emsp;&emsp;它将机器学习应用标准化用于共享：此外，预培训模型也可以在线获得，可以跨不同的深度学习框架使用，这些预培训模型可以帮助那些在GPU方面资源有限的人，以便 不用每次都从头开始。我们可以站在巨人的肩膀上，从那里开始。<br>
&emsp;&emsp;有许多深度学习框架，具有不同的优势、范式、抽象级别、编程语言等等。<br>
&emsp;&emsp;与GPU进行并行处理的接口：使用GPU进行计算是一个很吸引人的特性，因为GPU比cpu快得多，因为有大量的内核和并行性。<br> 

&emsp;&emsp;这就是为什么TensorFlow要想在深度学习中取得进展几乎是必要的，因为它可以帮助您的项目。<br>
&emsp;&emsp;那么，简单地说，什么是TensorFlow？<br>
&emsp;&emsp;TensorFlow是Google的一个深度学习框架，它是使用数据流图进行数值计算的开放源代码。<br>
&emsp;&emsp;它最初是由google大脑小组开发的，目的是促进他们的机器学习研究。<br>
&emsp;&emsp;TensorFlow是表示机器学习算法的接口，也是执行这些算法的实现。<br>
&emsp;&emsp;TensorFlow是如何工作的，基本范式是什么？<br>

### 计算图
&emsp;&emsp;关于TensorFlow的所有大思想中最大的思想是，数字计算是用计算图表示的。所以，任何TensorFlow的骨干图都将是一个计算图。<br>
&emsp;&emsp;--图节点是具有任意数量的输入和输出的操作。<br>
&emsp;&emsp;--节点之间的图边将是在这些操作之间流动的张量，而在实践中实现Tensor(张量)的最佳方法是n维矩阵。<br>

&emsp;&emsp;使用这种流图作为深度学习框架的主干的优点是，它允许您以小而简单的操作来构建复杂的模型。还有，这将使梯度计算非常简单，当我们在后面的一节中讨论这个问题时：<br>
![image1]()
&emsp;&emsp;另一种思考TensorFlow图的方法是，每个操作都是一个可以在此时计算的函数。<br>

### TensorFlow 数据类型, 变量, 和占位符
&emsp;&emsp;对计算图的理解将帮助我们从小的子图和运算的角度来考虑复杂的模型。<br>
&emsp;&emsp;让我们看一个只有一个隐藏层的神经网络的例子，以及它的计算图在TensorFlow中的样子：<br>
![image1]()
&emsp;&emsp;因此，我们有一些试图去计算的隐藏层，用RELU函数激活参数矩阵W乘以输入x再加上一个偏置项b。relu函数获取输出的最大值和0。<br>
&emsp;&emsp;下图显示了该图形在TensorFlow中的样子：<br>
![image1]()
&emsp;&emsp;在这个图中，我们有b和W的变量，我们有一个叫做占位符的x；我们在图中的每个操作都有节点。所以，让我们更详细地了解一下这些节点类型。<br>

#### 变量
&emsp;&emsp;变量是输出其当前值的有状态节点。在这个例子中，我们所说的变量是有状态的，这仅仅是b和W的意思，就是它们保留了它们当前的值。 通过多次执行，很容易将保存的值恢复到变量：<br>
![image1]()
&emsp;&emsp;此外，变量还有其他有用的特性；例如，可以在培训期间和之后将它们保存到您的磁盘上，这样可以实现我们前面提到的允许来自不同公司和组的人员保存、存储和将模型参数发送给其他人此外，变量是您想要优化的东西，以便将损失降到最低，我们将很快了解如何做到这一点。<br>

&emsp;&emsp;重要的是要知道图中的变量(如b和W)仍然是操作，因为根据定义，图中的所有节点都是操作。因此，当您评估这些在运行时保存b和W值的操作时，您将得到这些变量的值。<br>

&emsp;&emsp;我们可以使用TensorFlow的Variable()函数来定义一个变量并给它一些初始值：<br>

`var = tf.Variable(tf.random_normal((O,l)),name='random_values')`

&emsp;&emsp;这一行代码将定义一个2乘2的变量，并根据标准正态分布初始化它。您还可以给变量命名。<br>

#### 占位符
&emsp;&emsp;下一种类型的节点是占位符。占位符是在执行时输入值的节点:<br>
![image1]()
&emsp;&emsp;如果您的计算图表中有依赖于某些外部数据的输入，这些是值的占位符，我们将在培训期间将这些值添加到计算中。所以，对于占位符，我们不提供任何初始值。我们只分配张量的数据类型和形状，这样图表仍然知道要计算什么，即使它还没有任何存储的值。<br>

&emsp;&emsp;我们可以使用TensorFlow的占位符函数（ placeholder function）来创建一个占位符：<br>
```ph_varl = tf.placeholder(tf.float32,shape=(2,3)) ph_var2 = tf.placeholder(tf.float32,shape=(3,2)) result = tf.matmul(ph_varl,ph_var2)
```
&emsp;&emsp;这些代码行定义两个特定形状的占位符变量，然后定义将这两个值相乘的操作(请参阅下一节)。<br>

#### 数学运算
&emsp;&emsp;第三种类型的节点是数学运算，它们将是我们的矩阵乘法(MatMul)、加法(Add)和relu激活函数。所有这些都是TensorFlow图中的节点，它类似于NumPy操作：<br>
![image1]()
&emsp;&emsp;让我们看看这个图在代码中的样子。<br>

&emsp;&emsp;我们执行以下步骤来生成前面的图表：<br>

&emsp;&emsp;1.创建权重W和b，包括初始化。我们可以从均匀分布W~均匀(-1，1)中采样初始化权矩阵W，并将b初始化为0。<br>
&emsp;&emsp;2.创建输入占位符x，它的形状为m*784输入矩阵<br>
&emsp;&emsp;3.创建一个流程图。<br>

&emsp;&emsp;让我们继续按照以下步骤构建流程图：<br>
```# import TensorFlow package import tensorflow as tf
# build a TensorFlow variable b taking in initial zeros of size lOO
# ( a vector of lOO values)
b	= tf.Variable(tf.zeros((lOO,)))
# TensorFlow variable uniformly distributed values between –l and l
# of shape 784 by lOO
W = tf.Variable(tf.random_uniform((784, lOO),–l,l))
# TensorFlow placeholder for our input data that doesn't take in
# any initial values, it just takes a data type 32 bit floats as
# well as its shape
x = tf.placeholder(tf.float32, (lOO, 784))
# express h as Tensorflow ReLU of the TensorFlow matrix


#Multiplication of x and W and we add b h = tf.nn.relu(tf.matmul(x,W) + b )
```
&emsp;&emsp;正如您从前面的代码中看到的那样，我们实际上并不使用此代码片段操作任何数据。我们只是在我们的图形中构建符号，而您不能打印h并看到它的值，直到我们运行这个图表。因此，这个代码片段只是用来为我们的模型构建一个主干。如果您试图在前面的代码中打印W或b的值，您应该在Python中获得以下输出 :<br>
![image1]()
&emsp;&emsp;到目前为止，我们已经定义了我们的图形，现在，我们需要实际运行它。<br>


### 从TensorFlow得到输出
&emsp;&emsp;在上一节中，我们知道如何构建计算图，但是我们需要实际运行它并获得它的值。<br>

&emsp;&emsp;我们可以使用一个叫做session的东西来部署/运行这个图形，它只是一个绑定到一个特定的执行上下文，比如一个CPU或一个GPU。所以，我们要把我们构建的图形放到CPU或GPU上下文中。<br>

&emsp;&emsp;要运行这个图，我们需要定义一个名为sess的session对象，我们将调用函数Run，它包含两个参数：<br>
sess.run(fetches, feeds)

&emsp;&emsp;获取是返回节点输出的图节点的列表。这些是我们感兴趣的计算值的节点。<br>
&emsp;&emsp;前提要将是从图形节点到我们希望在模型中运行的实际值之间的字典映射。所以，这就是我们之前讨论过的占位符的填充位置。<br>

&emsp;&emsp;那么，让我们继续运行我们的图表：<br>
```# importing the numpy package for generating random variables for
# our placeholder x import numpy as np
# build a TensorFlow session object which takes a default execution
# environment which will be most likely a CPU sess = tf.3ession()
# calling the run function of the sess object to initialize all the
# variables. sess.run(tf.global_variables_initializer())
# calling the run function on the node that we are interested in,
# the h, and we feed in our second argument which is a dictionary
# for our placeholder x with the values that we are interested in. sess.run(h, (x: np.random.random((lOO,784))})
```
&emsp;&emsp;在通过Sess对象运行我们的图形之后，我们应该得到一个类似于下面的输出：<br>
![image1]()
&emsp;&emsp;如您所见，在上面代码片段的第二行中，我们初始化了变量，这是TensorFlow中的一个概念，称为懒惰计算。这意味着图形的计算只能在运行时进行，而TensorFlow中的运行时表示会话。因此，调用此函数，global_variables_initializer()，实际上将初始化你的图片中任何称为变量的内容，例如我们的例子中的W和b。<br>

&emsp;&emsp;我们还可以在WITH块中使用Session变量，以确保它在执行图形后将被关闭：<br>
```
ph_varl = tf.placeholder(tf.float32,shape=(2,3)) ph_var2 = tf.placeholder(tf.float32,shape=(3,2)) result = tf.matmul(ph_varl,ph_var2)
with tf.3ession() as sess: print(sess.run([result],feed_dict=(ph_varl:[[l.,3.,4.],[l.,3.,4.]],ph_var2:
[[l., 3.],[3.,l.],[.l,4.]]}))

Output: [array([[lO.4, 22. ],
[lO.4, 22. ]], dtype=float32)]
```

### TensorBoard – 可视化学习
&emsp;&emsp;使用TensorFlow进行的计算-比如训练一个庞大的深层神经网络-可能是复杂和混乱的，其相应的计算图也将是复杂的。为了方便理解、调试和优化TensorFlow程序，可以使用一套名为TensorBoard的可视化工具，它是一套可以在你的浏览器上运行的Web应用程序。 <br>

&emsp;&emsp;TensorBoard可以用于可视化您的TensorFlow图形，绘制有关图形执行的量化度量，并显示其他数据，如通过它的图像。当TensorBoard 完全配置，如下所示：<br>
![image1]()
&emsp;&emsp;为了了解TensorBoard是如何工作的，我们将构建一个计算图，它将充当MNIST数据集的分类器，MNIST数据集是手写图像的数据集。<br>

&emsp;&emsp;您不必了解这个模型的所有部分，但它将向您展示在TensorFlow中实现的机器学习模型的一般流程。<br>

&emsp;&emsp;因此，让我们从导入TensorFlow并使用TensorFlow助手函数加载所需的数据集开始；这些辅助函数将检查您是否已经下载了数据集，因此，它将为您下载：<br>
```
import tensorflow as tf

# Using TensorFlow helper function to get the MNI3T dataset from tensorflow.examples.tutorials.mnist import input_data
mnist_dataset = input_data.read_data_sets("/tmp/data/", one_hot=True)
```
&emsp;&emsp;输出:<br>
```Extracting /tmp/data/train–images–idx3–ubyte.gz Extracting /tmp/data/train–labels–idxl–ubyte.gz Extracting /tmp/data/tlOk–images–idx3–ubyte.gz Extracting /tmp/data/tlOk–labels–idxl–ubyte.gz
```
&emsp;&emsp;接下来，我们需要定义超参数(可以用来微调模型性能的参数)和模型的输入：<br>
```# hyperparameters of the the model (you don't have to understand the functionality of each parameter)
learning_rate = O.Ol num_training_epochs = 25 train_batch_size = lOO display_epoch = l
logs_path = '/tmp/tensorflow_tensorboard/'

# Define the computational graph input which will be a vector of the image pixels
# Images of MNI3T has dimensions of 28 by 28 which will multiply to 784 input_values = tf.placeholder(tf.float32, [None, 784], name='input_values')

# Define the target of the model which will be a classification problem of lO classes from O to 9
target_values = tf.placeholder(tf.float32, [None, lO], name='target_values')

# Define some variables for the weights and biases of the model weights = tf.Variable(tf.zeros([784, lO]), name='weights') biases = tf.Variable(tf.zeros([lO]), name='biases')
```
&emsp;&emsp;现在我们需要构建模型并定义一个成本函数，我们将对其进行优化：<br>
```# Create the computational graph and encapsulating different operations to different scopes
# which will make it easier for us to understand the visualizations of TensorBoard
with tf.name_scope('Model'):
# Defining the model
predicted_values = tf.nn.softmax(tf.matmul(input_values, weights) + biases)

with tf.name_scope('Loss'):
# Minimizing the model error using cross entropy criteria model_cost = tf.reduce_mean(–
tf.reduce_sum(target_values*tf.log(predicted_values), reduction_indices=l)) with tf.name_scope('3GD'):


# using Gradient Descent as an optimization method for the model cost above
model_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model_cost)

with tf.name_scope('Accuracy'):
#Calculating the accuracy
model_accuracy = tf.equal(tf.argmax(predicted_values, l), tf.argmax(target_values, l))
model_accuracy = tf.reduce_mean(tf.cast(model_accuracy, tf.float32))

# TensorFlow use the lazy evaluation strategy while defining the variables
# 3o actually till now none of the above variable got created or initialized
init = tf.global_variables_initializer()
```
&emsp;&emsp;我们将定义汇总变量，用于监视特定变量(如损失)上发生的更改，以及如何通过培训过程使其变得更好：<br>

```# Create a summary to monitor the model cost tensor tf.summary.scalar("model loss", model_cost)

# Create another summary to monitor the model accuracy tensor tf.summary.scalar("model accuracy", model_accuracy)

# Merging the summaries to single operation merged_summary_operation = tf.summary.merge_all()
```
&emsp;&emsp;最后，我们将通过定义一个会话变量来运行该模型，该变量将用于执行我们构建的计算图：<br>

```# kick off the training process with tf.3ession() as sess:

# Intialize the variables sess.run(init)

# operation to feed logs to TensorBoard summary_writer = tf.summary.FileWriter(logs_path,
graph=tf.get_default_graph())

# 3tarting the training cycle by feeding the model by batch at a time for train_epoch in range(num_training_epochs):

average_cost = O.
total_num_batch = int(mnist_dataset.train.num_examples/train_batch_size)


# iterate through all training batches for i in range(total_num_batch):
batch_xs, batch_ys = mnist_dataset.train.next_batch(train_batch_size)

# Run the optimizer with gradient descent and cost to get the loss
# and the merged summary operations for the TensorBoard
_, c, summary = sess.run([model_optimizer, model_cost, merged_summary_operation],
feed_dict=(input_values: batch_xs, target_values: batch_ys})

# write statistics to the log et every iteration summary_writer.add_summary(summary, train_epoch * total_num_batch + i)

# computing average loss average_cost += c / total_num_batch

# Display logs per epoch step
if (train_epoch+l) % display_epoch == O: print("Epoch:", '%O3d' % (train_epoch+l), "cost=",
"(:.9f}".format(average_cost)) print("Optimization Finished!")
# Testing the trained model on the test set and getting the accuracy compared to the actual labels of the test set
print("Accuracy:", model_accuracy.eval((input_values: mnist_dataset.test.images, target_values: mnist_dataset.test.labels}))

print("To view summaries in the Tensorboard, run the command line:\n" \ "––> tensorboard ––logdir=/tmp/tensorflow_tensorboard " \
"\nThen open http://O.O.O.O:6OO6/ into your web browser")
```
&emsp;&emsp;培训过程的结果应与此类似：<br>
```Epoch: OOl cost= l.l83lO9l28 Epoch: OO2 cost= O.6652lO275 Epoch: OO3 cost= O.552693334 Epoch: OO4 cost= O.498636444 Epoch: OO5 cost= O.4655l6675 Epoch: OO6 cost= O.4426l838l Epoch: OO7 cost= O.4255225l3 Epoch: OO8 cost= O.4l2l94222 Epoch: OO9 cost= O.4Ol4O8l34 Epoch: OlO cost= O.392437336 Epoch: Oll cost= O.3848l6745 Epoch: Ol2 cost= O.378l83398 Epoch: Ol3 cost= O.372455584 Epoch: Ol4 cost= O.367275238


Epoch: Ol5 cost= O.3627727ll Epoch: Ol6 cost= O.35859l895 Epoch: Ol7 cost= O.35489223l Epoch: Ol8 cost= O.35l45l424 Epoch: Ol9 cost= O.348337946 Epoch: O2O cost= O.345453O95 Epoch: O2l cost= O.342769O8O Epoch: O22 cost= O.34O236O65 Epoch: O23 cost= O.337953l5l Epoch: O24 cost= O.335739OOl Epoch: O25 cost= O.3337O28l8 Optimization Finished!
Accuracy: O.9l46
To view summaries in the Tensorboard, run the command line:
––> tensorboard ––logdir=/tmp/tensorflow_tensorboard Then open http://O.O.O.O:6OO6/ into your web browser
```
&emsp;&emsp;要查看TensorBoard中的汇总统计信息，我们将在终端中发出以下命令，从而在输出结束时遵循该消息：<br>

`tensorboard ––logdir=/tmp/tensorflow_tensorboard`

&emsp;&emsp;然后, 在你的浏览器上打开 http://O.O.O.O:6OO6/ 。<br>
&emsp;&emsp;当您打开TensorBoard时，您应该得到类似于以下屏幕截图的内容：<br>
![image1]()
&emsp;&emsp;这显示了我们正在监控的变量，比如模型的准确性和它是如何变得更高的，模型的丢失以及在整个训练过程中它是如何降低的。所以，你观察 我们在这里有一个正常的学习过程。但有时你会发现精度和模型损失是随机变化的，或者你想跟踪一些变量以及它们是如何变化的。 在整个过程中，TensorBoard将非常有用，帮助您发现任何随机性或错误。<br>

&emsp;&emsp;另外，如果切换到TensorBoard中的图形选项卡，您将看到我们在前面的代码中构建的计算图：<br>
![image1]()

### 总结
&emsp;&emsp;在本章中，我们介绍了Ubuntu和mac的安装过程，概述了TensorFlow编程模型，并解释了可以使用的不同类型的简单节点。 构建复杂的操作以及如何使用会话对象从TensorFlow获得输出。此外，我们还讨论了Tensorboard以及为什么它将有助于调试和分析复杂的深度学习应用程序。<br>
&emsp;&emsp;接下来，我们将对神经网络和多层神经网络背后的直觉进行基本解释。我们还将介绍一些TensorFlow的基本示例，并演示它是如何实现的，可用于回归和分类问题。<br>
