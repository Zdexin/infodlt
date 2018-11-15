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
