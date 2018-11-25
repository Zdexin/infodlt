# 6.深度前馈神经网络-实现数字分类
&emsp;&emsp;前馈神经网络(FNN)是一种特殊类型的神经网络，神经元之间的连接没有形成一个循环。因此，它不同于神经网络中的其他结构，我们将在本书后面学习(递归型神经网络)。FNN是一种广泛使用的结构，是第一个也是最简单的神经网络类型。<br>
&emsp;&emsp;在本章中，我们将介绍一个典型的FNN体系结构，我们将使用TensorFlow库来实现这一点。在介绍这些概念之后，我们将给出一个数字分类的实际例子。这个示例的问题是，给定一组包含手写数字的图像，如何将这些图像分类为10个不同的类(0-9)?<br>
&emsp;&emsp;本章将介绍以下主题:<br>
&emsp;&emsp;&emsp;&emsp;隐藏单元和体系结构设计 <br>
&emsp;&emsp;&emsp;&emsp;MNIST 数据集分析<br>
&emsp;&emsp;&emsp;&emsp;数字分类-模型构建和训练<br>
## 隐含层设计
&emsp;&emsp;在下一节中, 我们将回顾人工神经网络;他们可以在分类任务中做得很好, 比如对手写数字进行分类。<br>
&emsp;&emsp;假设我们有图1所示的网络:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/1.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图1“有一个隐含层的简单FNN”<br>
&emsp;&emsp;如前所述, 此网络中最左侧的层称为输入层, 而层内的神经元称为输入神经元。最右边或输出层包含输出神经元，或者在这种情况下包含单个输出神经元。中间层被称为隐含层, 因为这个层中的神经元既不是输入也不是输出。“隐含”这个词听起来可能有点神秘。第一次听到这个词时，我想它一定有很深的哲学或数学意义。没有别的意思。前一个网络只有一个隐含层，但有些网络有多个隐含层。例如，下面的四层网络有两个隐含层:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/2.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图2“具有更多隐含层的人工神经网络”<br>
&emsp;&emsp;其中组织输入、隐含和输出层的体系结构非常简单。例如，让我们通过一个实际的例子来看看一个特定的手写图像中是否有数字9。<br>
&emsp;&emsp;首先，我们将把输入图像的像素输入到输入层;例如，在MNIST数据集中，我们有单色图像。每一个都是28×28，所以我们需要在输入层中有28* 28= 784个神经元来接收这个输入图像。<br>
&emsp;&emsp;在输出层中, 我们只需要1个神经元, 这会产生一个概率 (或分数), 该图像是否有数字9。例如, 大于0.5 的输出值表示此图像具有数字 9, 如果小于 0.5, 则表示输入图像中没有数字9。<br>
&emsp;&emsp;因此，这些类型的网络，其中一个层的输出作为输入被输入到下一个层，被称为FNNs。这种分层的序列意味着里面没有循环。<br>
## MNIST数据集分析
&emsp;&emsp;在这部分，我们将用一个分类器来获取手写图像的信息。这种实现可以看作是神经网络中的Hello world!。<br>
&emsp;&emsp;MINST是一种广泛使用的用于测试机器学习技术的数据集。数据集包含一组手写数字，如下图所示:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/3.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图3 "mnist 数据集中的采样数字”<br>
&emsp;&emsp;因此, 数据集还包括手写图像及其相应的标签。<br>
&emsp;&emsp;在这一节中，我们将对这些图像进行基本模型的训练，我们的目标是在输入图像中分辨出哪个数字是手写的。<br>
&emsp;&emsp;另外，您会发现我们可以使用很少几行代码来完成这个分类任务，但是这个实现背后的思想是理解构建神经网络解决方案的基本细节。此外，我们将在此实现中介绍神经网络的主要概念。<br>
## MNIST数据集
&emsp;&emsp;MNIST 数据由Yann LeCun 的网站 (http://yann.lecun.com/exdb/mnist/) 上。幸运的是, 幸运的是，TensorFlow提供了一些帮助函数来下载数据集，所以让我们首先使用以下两行代码下载数据集:<br>
```python
from tensorflow.examples.tutorials.mnist import input_data 
mnist_dataset = input_data.read_data_sets("MNI3T_data/", one_hot=True)
```
&emsp;&emsp;MNIST 数据分为三部分: 培训数据的5.5万个训练数据 (minist.train)、1万个测试数据 (minist.test) 和5000点验证数据 (minist.validation)。在机器学习过程中，这种分流非常重要;我们必须有独立数据，我们不从中学习，才能确保我们所学到的东西实际上是一般化的!<br>
&emsp;&emsp;如前所述, 每个 MNIST 示例都有两个部分: 手写数字的图像及其对应的标签。训练集和测试集都包含图像及其相应的标签。例如, 训练图像是 mnist.train.images , 训练标签是 mnist.train.labels 。<br>
&emsp;&emsp;每个图像是28像素 x 28 像素。我们可以把它看做一个大的数字数组:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/4.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 4 "矩阵表示中的 MNIST 数字”<br>
&emsp;&emsp;为了将这个像素值矩阵提供给神经网络的输入层，我们需要将这个矩阵合并为一个有784个值的向量。数据集的最终形状是一串的784维向量空间。<br>
&emsp;&emsp;结果是 mnist.train.images 是一个形状为 (55OOO, 784) 的张量。第一个维度是图像列表的索引, 第二个维度是每个图像中每个像素的索引。张量中的每个条目都是特定图像中特定像素的0到1之间的像素强度:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/5.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 5 "MNIST 数据分析<br>
&emsp;&emsp;为了实现这个目的，我们将把标签编码为one-hot向量。一个one-hot向量是除此向量表示的数字的索引之外的所有都为零的向量。例如, 3 将是 [00、01、00、00、00]。因此, mnist.train.labels 是一个 (55OOO, 10) 的浮点数组:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/6.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 6 "MNIST 数据分析<br>
## 数字分类–模型构建和训练
&emsp;&emsp;现在，让我们继续构建我们的模型。我们的数据集中有10个类0-9目标是将任何输入图像分类到这些类中。我们将生成一个10个可能值的向量(因为我们有10个类)。它将表示从0到9的每一位数字作为输入图像的正确类的概率。<br>
&emsp;&emsp;例如，假设我们向模型提供特定的图像。模型可能70%确定这个图像是9 10%确定这个图像是8，以此类推。因此，我们将在这里使用softmax回归，它将产生0到1之间的值。<br>
&emsp;&emsp;softmax回归有两个步骤:首先，我们将输入在特定类中的证据加起来，然后将这些证据转换为概率。<br>
&emsp;&emsp;为了证明给定图像属于特定类别，我们对像素强度进行加权和。如果高强度的像素是不利于该类图像的证据，权重为负;如果是有利于该类图像的证据，权重为正。<br>
&emsp;&emsp;图7显示了每个类的权重的一个模型。红色表示负权重, 而蓝色表示正权重:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/7.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 7 "权重为每个 MNIST 类的一个学习模型"<br>
&emsp;&emsp;我们还加入了额外的偏见。基本上，我们想说的是有些东西更可能独立于输入。结果是, 给定一个输入![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(20).gif)的类![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(21).gif)的证据是: <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/1.jpg)<br>
&emsp;&emsp;其中:<br>
&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(22).gif)是权重<br>
&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(23).gif)是类 <br>
&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(24).gif)是对输入图像x中的像素求和的索引。<br>
&emsp;&emsp;然后，使用softmax函数将证据转换为我们预测的概率![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(25).gif):<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(26).gif):<br>
&emsp;&emsp;在在这里，softmax充当一个激活函数或链接函数，将线性函数的输出塑造成我们想要的形式，在本例中是10种情况下的概率分布(因为我们有10种可能的类，从0到9)。你可以把它看作是把证据转换成每个类的输入的概率。它的定义为:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(27).gif):<br>
&emsp;&emsp;如果你展开这个方程，你会得到:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/8.png)<br>
&emsp;emsp;但是把softmax看作第一种方法通常更有帮助:对输入进行幂运算，然后对其进行规范化。求幂意味着多一个单位的证据就会成倍地增加任何假设的权重。反过来说，如果证据少一个单位，就意味着一个假设得到的只是其早期权重的一小部分。没有一个假设的权重是零或负的。Softmax将这些权值规范化，使它们相加为1，形成一个有效的概率分布。<br>
&emsp;&emsp;你可以把我们的softmax回归想象成如下图所示，尽管有更多的![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(20).gif)。对于每个输出，我们计算![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(20).gif)的加权和，添加一个偏差，然后应用softmax:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/9.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 8 "softmax 回归的可视化"<br>
&emsp;&emsp;如果我们把它写成方程式, 我们得到: <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/10.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 9 "softmax 回归的等式表示法"<br>
&emsp;&emsp;我们可以使用向量表示法进行此过程。这意味着我们将把它变成一个矩阵乘法和向量加法。这对于计算效率和可读性非常有帮助:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/11.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 10 "softmax 回归方程的矢量化表示"<br>
&emsp;&emsp;更简洁地说，我们可以这样写: <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(28).gif)<br>
&emsp;&emsp;现在, 让我们把它转化为 TensorFlow 可以使用的东西。<br>
## 数据分析
&emsp;&emsp;让我们开始实现分类器，首先从导入这个实现所需的包开始:<br>
```python
    import tensorflow as tf
    import matplotlib.pyplot as plt import numpy as np
    import random as ran
```
&emsp;&emsp;接下来，我们将定义一些辅助函数，使我们能够从原始数据集下载我们的子集:<br>
```python
    #Define some helper functions
    # to assign the size of training and test data we will take from MNI3T dataset
    def train_size(size):
        print ('Total Training Images in Dataset = ' + 
    str(mnist_dataset.train.images.shape))
        print ('############################################')
        input_values_train = mnist_dataset.train.images[:size,:] 
        print ('input_values_train 3amples Loaded = ' +
    str(input_values_train.shape))
        target_values_train = mnist_dataset.train.labels[:size,:] 
        print ('target_values_train 3amples Loaded = ' +
    str(target_values_train.shape))
        return input_values_train, target_values_train

    def test_size(size):
        print ('Total Test 3amples in MNI3T Dataset = ' + str(mnist_dataset.test.images.shape))
        print ('############################################')
        input_values_test = mnist_dataset.test.images[:size,:] 
        print ('input_values_test 3amples Loaded = ' +
    str(input_values_test.shape))
        target_values_test = mnist_dataset.test.labels[:size,:] 
        print ('target_values_test 3amples Loaded = ' +
    str(target_values_test.shape))
        return input_values_test, target_values_test
```
&emsp;emsp;同时,我们将定义两个辅助函数显示数据集的具体数字,甚至显示图像的一个子集的平铺版本:<br>
```python
    #Define a couple of helper functions for digit images visualization 
    def visualize_digit(ind):
        print(target_values_train[ind])
        target = target_values_train[ind].argmax(axis=O) 
        true_image = input_values_train[ind].reshape([28,28]) 
        plt.title('3ample: %d Label: %d' % (ind, target))     
        plt.imshow(true_image, cmap=plt.get_cmap('gray_r')) 
        plt.show()

    def visualize_mult_imgs_flat(start, stop):
        imgs = input_values_train[start].reshape([l,784]) 
        for i in range(start+l,stop):
            imgs = np.concatenate((imgs, 
    input_values_train[i].reshape([l,784])))
        plt.imshow(imgs, cmap=plt.get_cmap('gray_r')) 
        plt.show()
```
&emsp;&emsp;现在，让我们进入正题，开始处理数据集。所以我们将定义我们想从原始数据集加载的培训和测试的例子。<br>
&emsp;&emsp;现在，我们将开始构建和培训我们的模型。首先，我们用希望加载多少训练和测试示例来定义变量。目前，我们将加载所有数据，但稍后我们将更改此值以节省资源:<br>
```python
    input_values_train, target_values_train = train_size(55000)

    Output:
    Total Training Images in Dataset = (55000, 784)
    ############################################
    input_values_train samples Loaded = (55000, 784) 
    target_values_train samples Loaded = (55000, 10)
```
&emsp;&emsp;现在，我们有一个55000个手写数字样本的训练集，每个样本是28×28像素的图像被压缩，成为一个784维的向量。我们也有相应的标签在one-hot编码格式。<br>
7emsp;&emsp;target_values_train 数据是所有 input_values_train 样本的关联标签。在下面的示例中, 数组以one-hot格式表示 7:<br>


Label|0|1|2|3|4|5|6|7|8|9
-|-|-|-|-|-|-|-|-|-|-
Array|[0,|0,|0,|0,|0,|0,|0,|1,|0,|0]


&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 11 "数字7的一个one-hot编码"<br>
&emsp;&emsp;让我们从数据集中可视化一个随机的图像，看看它是什么样子的，所以我们将使用前面的辅助函数来显示数据集中的一个随机数字:<br>
```python
    visualize_digit(ran.randint(0, input_values_train.shape[0])) 
```
&emsp;&emsp;Output:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/12.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 12 "display_digit输出数字的方法"<br>
&emsp;&emsp;我们还可以使用前面定义的辅助函数来可视化一组平面图像。平面中的每个值都表示像素强度，因此将像素可视化如下:<br>
```python
    visualize_mult_imgs_flat(0,400) 
```
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/13.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 13 "前400个训练示例"<br>
## 构建模型
&emsp;&emsp;到目前为止，我们还没有开始为这个分类器构建计算图。让我们从创建会话变量开始，该变量将负责执行我们将要构建的计算图:<br>
```python
    sess = tf.session()
```
&emsp;&emsp;接下来, 我们将定义模型的占位符, 用于将数据输送到计算图中:<br>
```python
    input_values = tf.placeholder(tf.float32, shape=[None, 784]
```
&emsp;&emsp;当我们在占位符的第一个维度中指定None时，就意味着占位符可以被提供尽可能多的示例。在本例中，我们的占位符可以被赋给任意数量的示例，其中每个示例都有784个值。<br>
&emsp;&emsp;现在, 我们需要定义另一个占位符来输送图像标签。此外, 我们稍后将使用此占位符将模型预测与图像的实际标签进行比较:<br>
```python
output_values = tf.placeholder(tf.float32, shape=[None, 10])
```
&emsp;&emsp;接下来, 我们将定义权重和偏差。这两个变量将是我们网络的可训练参数, 它们将是对不可见数据进行预测所需的唯一两个变量:<br>
```python
    weights = tf.Variable(tf.zeros([784,10])) biases = tf.Variable(tf.zeros([10]))
```
&emsp;&emsp;我喜欢把这些weight看成是每个数字的10个备忘单。这类似于老师用小抄来给多项选择题打分。 <br>
&emsp;&emsp;现在我们将定义softmax回归，这是我们的分类器函数。这个特殊的分类器被称为多项式逻辑回归，我们通过将这个数字的平型的数字乘以权重来做预测然后再加上偏差。<br>
```python
    softmax_layer = tf.nn.softmax(tf.matmul(input_values,weights) + biases)
```
&emsp;&emsp;首先, 让我们忽略 softmax，看softmax 函数内部是什么。matmul 是矩阵乘法的 TensorFlow函数。如果你知道矩阵乘法 (https://en.Wekipedia.org./wiki/Matrix_multiplication), 你会明白这个计算是正确的,并且:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/2.jpg)<br>
&emsp;&emsp;将会有大量的训练示例以f(m)f(n)矩阵为例：<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/14.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 14 "简单矩阵乘法"<br>
&emsp;&emsp;您可以通过评估 softmax_layer 来确认:<br>
```python
    print(softmax_layer) 
    Output:
    Tensor("softmax:0", shape=(？, 10), dtype=float32)
```
&emsp;&emsp;现在, 让我们来试验一下我们以前定义的计算图, 其中有三个来自训练集的样本, 看看它是如何工作的。要执行计算图, 我们需要使用之前定义的会话变量。我们需要使用 tf. global_variables_initializer () 初始化变量。<br>
&emsp;&emsp;让我们继续，只向计算图提供三个示例:<br>
```python
    input_values_train, target_values_train = train_size(3)
    sess.run(tf.global_variables_initializer())
    #If using TensorFlow prior to 0.12 use:
        #sess.run(tf.initialize_all_variables()) 
    print(sess.run(softmax_layer, feed_dict=(input_values: 
    input_values_train}))

    Output:

    [[	0.1	0.1	0.1	0.1	0.1	0.1	0.1	0.1	0.1	0.1]
    [	0.1	0.1	0.1	0.1	0.1	0.1	0.1	0.1	0.1	0.1]
    [	0.1	0.1	0.1	0.1	0.1	0.1	0.1	0.1	0.1	0.1]]
```
&emsp;&emsp;在这里，您可以看到提供给它的三个训练样本的模型预测。目前，模型对我们的任务一无所知因为我们还没有经过培训过程，所以它只输出了10%的概率作为输入样本的正确类。<br>
&emsp;&emsp;如前所述，softmax是一个激活函数，它将输出压缩到0到1之间，softmax的TensorFlow确保了单个输入示例的所有概率之和为1。<br>
&emsp;&emsp;让我们来实验一下 TensorFlow 的 softmax 函数:<br>
```python
    sess.run(tf.nn.softmax(tf.zeros([4])))
    sess.run(tf.nn.softmax(tf.constant([0.1, 0.005, 2])))
    
    Output:
    array([0.11634309, 0.10579926, 0.7778576 ], dtype=float32)
```
&emsp;&emsp;接下来，我们需要为这个模型定义损失函数，它将测量在为输入图像分配类时分类器的好坏。我们的模型的准确性是通过比较数据集中的实际值和我们从模型中得到的预测来计算的。<br>
&emsp;&emsp;我们的目标是减少实际值和预测值之间的任何错误分类。交叉熵定义为:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/3.jpg)<br>
&emsp;&emsp;其中:<br>
&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(29).gif)是我们预测的概率分布<br>
&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/CodeCogsEqn%20(30).gif)是真实分布 (带有数字标签的one-hot向量)<br>
&emsp;&emsp;从某种粗略的意义上说，交叉熵度量了我们在描述实际输入时的预测效率有多低。<br>
```python
    model_cross_entropy = tf.reduce_mean(–tf.reduce_sum(output_values * 
    tf.log(softmax_layer), reduction_indices=[1]))
```
&emsp;&emsp;此函数从 softmax_layer (其值范围从0到 1) 获取我们所有预测的日志, 并将它们按元素大小 (https://en.wikipedia.org/wiki/Hadamard_product_%28matrices%29) 乘以示例的真实值output_values。如果每个值的 log 函数接近零, 该负数值很大 (–np (0.01) = 4.6), 并且如果它接近1, 该负数值很小 (–np (0. 99) = 0.1):<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/15.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 15 "Y=log(x)"<br>
&emsp;&emsp;如果可以确定预测是不正确的，我们本质上是用一个非常大的数字惩罚分类器，如果可以确定是真确的，那么这个数将会非常小。下面是一个 softmax 预测的简单 Python 示例, 它确信数字是3:<br>
&emsp;&emsp;j = [0.03, 0.03, 0.01, 0.9, 0.01, 0.01, 0.0025,0.0025, 0.0025, 0.0025]<br>
&emsp;&emsp;让我们创建一个3的数组标签作为基本的事实来与我们的softmax功能相比<br>
&emsp;&emsp;K=[0,0,0,1,0,0,0,0,0,0]。<br>
&emsp;&emsp;你能猜出我们的损失函数给了我们什么吗?你能看出log (K)怎么用一个大的负数来惩罚一个错误的答案吗?请尝试以下操作以了解:<br>
```python
    –np.log(j)
    –np.multiply(np.log(j),k)
```
&emsp;&emsp;这将返回九零和值 0.1053;当他们都总结起来, 我们可以认为这是一个很好的预测。请注意, 当我们对实际的2进行相同的预测时会发生什么:<br>
&emsp;&emsp;k = [0,0,1,0,0,0,0,0,0,0]<br>
```python
    np.sum(–np.multiply(np.log(j),k))
```
&emsp;&emsp;现在, 我们的 cross_entropy 功能给我们 4.6051, 这显示了一个非常严重的错误预测, 它被严重地惩罚了，因为分类器非常自信它是3，而实际上它是2。<br>
&emsp;&emsp;接下来，我们开始训练分类器。为了训练它，我们必须为W和b建立适当的值，使我们的损失降到最低。<br>
&emsp;&emsp;下面是我们可以为训练分配自定义变量的地方。下面所有大写中的任何值都被设计成可以更改和打乱的。事实上，我推荐它!首先，使用这些值，然后注意当您使用的训练示例太少或学习率过高或过低时会发生什么:<br>
```python
    input_values_train, target_values_train = train_size(5500) 
    input_values_test, target_values_test = test_size(10000) 
    learning_rate = 0.1
    num_iterations = 2500
```
&emsp;&emsp;现在, 我们可以初始化所有变量, 以便我们的 TensorFlow 图可以使用它们:<br>
```python
    init = tf.global_variables_initializer()
    #If using TensorFlow prior to 0.12 use:
    #init = tf.initialize_all_variables()
    sess.run(init)
```
&emsp;&emsp;接下来, 我们需要使用梯度下降算法训练分类器。因此, 我们首先定义了我们的训练方法和一些变量来测量模型准确度。变量train将采用选择的学习率执行梯度下降优化器，以最小化模型损失函数 model_cross_entropy:<br>
```python
    train = 
    tf.train.GradientDescentOptimizer(learning_rate).minimize(model_cross_entro py)
    model_correct_prediction = tf.equal(tf.argmax(softmax_layer,1), 
    tf.argmax(output_values,1))
    model_accuracy = tf.reduce_mean(tf.cast(model_correct_prediction, 
    tf.float32))
```
## 模型训练
&emsp;&emsp;现在, 我们将定义一个循环，迭代num_iterations 次。对于每个循环, 它都进行了训练, 喂养， feed_dict 从 input_values_train 使用 target_values_train。为了计算准确性，它将使用不可见数据来对模型进行测试:<br>
```python
input_values_test :
    for i in range(num_iterations+1):
        sess.run(train, feed_dict=(input_values: input_values_train, 
    output_values: target_values_train})
        if i%100 == 0:
            print('Training 3tep:' + str(i) + ' Accuracy = ' + 
            str(sess.run(model_accuracy, feed_dict=(input_values: input_values_test, 
            output_values: target_values_test})) + ' Loss = ' + str(sess.run(model_cross_entropy, 
            (input_values: input_values_train, output_values: target_values_train})))
```
```python

Output:
Training step:0    Accuracy = 0.5988 Loss = 2.1881988 
Training step:100  Accuracy = 0.8647 Loss = 0.58029664 
Training step:200  Accuracy = 0.879  Loss = 0.45982164 
Training step:300  Accuracy = 0.8866 Loss = 0.40857208 
Training step:400  Accuracy = 0.8904 Loss = 0.37808096 
Training step:500  Accuracy = 0.8943 Loss = 0.35697535 
Training step:600  Accuracy = 0.8974 Loss = 0.34104997 
Training step:700  Accuracy = 0.8984 Loss = 0.32834956 
Training step:800  Accuracy = 0.9    Loss = 0.31782663 
Training step:900  Accuracy = 0.9005 Loss = 0.30886236 
Training step:1000 Accuracy = 0.9009 Loss = 0.3010645 
Training step:1100 Accuracy = 0.9023 Loss = 0.29417014 
Training step:1200 Accuracy = 0.9029 Loss = 0.28799513 
Training step:1300 Accuracy = 0.9033 Loss = 0.28240603 
Training step:1400 Accuracy = 0.9039 Loss = 0.27730304 
Training step:1500 Accuracy = 0.9048 Loss = 0.27260992 
Training step:1600 Accuracy = 0.9057 Loss = 0.26826677 
Training step:1700 Accuracy = 0.9062 Loss = 0.2642261 
Training step:1800 Accuracy = 0.9061 Loss = 0.26044932 
Training step:1900 Accuracy = 0.9063 Loss = 0.25690478 
Training step:2000 Accuracy = 0.9066 Loss = 0.2535662 
Training step:2100 Accuracy = 0.9072 Loss = 0.25041154 
Training step:2200 Accuracy = 0.9073 Loss = 0.24742197 
Training step:2300 Accuracy = 0.9071 Loss = 0.24458146 
Training step:2400 Accuracy = 0.9066 Loss = 0.24187621 
Training step:2500 Accuracy = 0.9067 Loss = 0.23929419
```
&emsp;&emsp;注意，在接近终点的时候损失仍然在减少，但是我们的准确率略有下降!这表明,我们仍然可以减少我们的损失,因此可以使我们的训练数据准确性最大化,但这可能无法帮助我们预测用于测试准确度的测试数据。这也被称为过度拟合(不是泛化)。使用默认设置，我们得到的准确率大约为91%。如果我想作弊以获得94%的准确率，我可以将测试示例设置为100。这显示了没有足够的测试例子能给你一种有偏差的准确性。<br>
&emsp;&emsp;请记住，这是计算分类器性能的一种非常不准确的方法。然而，我们出于学习和实验的目的，进行了这一操作。理想情况下，当使用大型数据集进行培训时，您可以一次使用小批量的培训数据进行培训，而不是一次全部培训。<br>
&emsp;&emsp;这是有趣的部分。现在我们已经计算了权重备忘单，我们可以用以下代码创建一个图:<br>
```python
    for i in range(10): plt.subplot(2, 5, i+1)
    weight = sess.run(weights)[:,i] plt.title(i)
    plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False) 
    frame.axes.get_yaxis().set_visible(False)
```
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/16.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 15 "我们的权重从0-9 的可视化"<br>
&emsp;&emsp;上图显示了0-9的模型权重，这是我们分类器最重要的方面。所有这些机器学习的工作都是为了找出最优的权重。一旦根据优化标准进行计算，你就有了备忘单，并且可以很容易地使用所学的权重找到答案。<br>
&emsp;&emsp; 学习的模型通过比较输入数字样本与红蓝权重的相似程度或不同程度来进行预测。红色越深，效果越好;白色表示中性，蓝色表示失误。<br>
&emsp;&emsp;现在,让我们使用备忘单,看看我们的模型执行:<br>
```python
    input_values_train, target_values_train = train_size(1) 
    visualize_digit(0)

    Output:
    Total Training Images in Dataset = (55000, 784)
    ############################################
    input_values_train samples Loaded = (l, 784) 
    target_values_train samples Loaded = (1, 10) 
    [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
```
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/12.png)<br>
&emsp;&emsp;让我们看看我们的softmax预测器:<br>
```python
    answer = sess.run(softmax_layer, feed_dict=(input_values: input_values_train})
    print(answer)
```
&emsp;&emsp;前面的代码将给我们一个10维的向量，每一列包含一个概率: <br>
&emsp;&emsp;[[2.1248012e–05 1.1646927e–05 <br>
&emsp;&emsp;8.9631692e–02 1.9201526e–02 <br>
&emsp;&emsp;8.2086492e–04 1.2516821e–05 <br>
&emsp;&emsp;3.8538201e–05 8.5374612e–01  <br>
&emsp;&emsp;6.9188857e–03 2.9596921e–02]]<br>
&emsp;&emsp;我们可以使用 argmax 函数找出最可能的数字, 以便正确分类输入图像:<br>
```python
answer.argmax()

Output:
7
```
&emsp;&emsp;现在，我们从网络中得到了一个正确的分类。让我们用我们的知识来定义一个辅助函数，它可以从数据集中选择一个随机的图像，并根据它测试模型:<br>
```python
    def display_result(ind):
        # Loading a training sample
        input_values_train = mnist_dataset.train.images[ind,:].reshape(1,784) 
        target_values_train = mnist_dataset.train.labels[ind,:]
        # getting the label as an integer instead of one–hot encoded vector 
        label = target_values_train.argmax()
        # Getting the prediction as an integer
        prediction = sess.run(softmax_layer, feed_dict=(input_values: 
    input_values_train}).argmax()
        plt.title('Prediction: %d Label: %d' % (prediction, label)) 
        plt.imshow(input_values_train.reshape([28,28]),
    cmap=plt.get_cmap('gray_r')) 
    plt.show()
```
&emsp;&emsp;现在试一下:<br>
```python
    display_result(ran.randint(0, 55000)) 
    Output:
```
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/17.png)<br>
&emsp;&emsp;我们又得到了一个正确的分类!<br>
## 总结
&emsp;&emsp;在本章中，我们对数字分类任务进行了基本的FNN实现。我们还回顾了在神经网络环境中使用的术语。<br>
&emsp;&emsp;接下来，我们将使用一些现代的最佳实践和技巧来构建一个复杂的数字分类模型，以提高模型的性能。<br>


学号|姓名|专业
-|-|-
201802110481|辛杨|计算机软件与理论

