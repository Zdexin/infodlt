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
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter06/1.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图1“有一个隐含层的简单FNN”<br>
&emsp;&emsp;如前所述, 此网络中最左侧的层称为输入层, 而层内的神经元称为输入神经元。最右边或输出层包含输出神经元，或者在这种情况下包含单个输出神经元。中间层被称为隐含层, 因为这个层中的神经元既不是输入也不是输出。“隐含”这个词听起来可能有点神秘。第一次听到这个词时，我想它一定有很深的哲学或数学意义。没有别的意思。前一个网络只有一个隐含层，但有些网络有多个隐藏层。例如，下面的四层网络有两个隐含层:<br>

