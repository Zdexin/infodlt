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
&emsp;&emsp;我们还加入了额外的偏见。基本上，我们想说的是有些东西更可能独立于输入。结果是, 给定一个输入 x 的类 i 的证据是: <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/1.jpg)<br>
