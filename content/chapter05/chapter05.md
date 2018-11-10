# TensorFlow 在运行中的一些基本的例子
&emsp;&emsp;在本章中, 我们将解释 TensorFlow 背后的主要计算概念, 即计算图模型, 并实现线性回归和逻辑回归。<br>
&emsp;&emsp;本章主要介绍以下主题:<br>
&emsp;&emsp;&emsp;&emsp;1.单个神经元的容量和激活功能<br>
&emsp;&emsp;&emsp;&emsp;2.激活函数<br>
&emsp;&emsp;&emsp;&emsp;3.前馈神经网络<br>
&emsp;&emsp;&emsp;&emsp;4.多层网络的需求<br>
&emsp;&emsp;&emsp;&emsp;5.TensorFlow 的术语回顾<br>
&emsp;&emsp;&emsp;&emsp;6.线性回归模型-构建与训练<br>
&emsp;&emsp;&emsp;&emsp;7.逻辑回归模型-构建和训练<br>
&emsp;&emsp;我们将首先说明单个神经元实际上能做什么以及它的模型，并且基于这点，将出现对多层网络的需求。接下来，我们将进一步阐述在TensorFlow中使用的以及可用的一些主要概念和工具，以及如何使用这些工具来构建简单的示例，如线性回归和逻辑回归。<br>
## 单个神经元的容量
&emsp;&emsp;神经网络是一种计算模型，其灵感主要来自于人脑的生物神经网络处理输入信息的方式。神经网络在机器学习研究(特别是深度学习)和工业应用方面取得了巨大突破，例如在计算机视觉、语音识别和文本处理方面取得了突破性成果。在这一章中，我们将试着去理解一种叫做多层感知器的神经网络。<br>
## 生物运作原理和连系
&emsp;&emsp;我们大脑的基本计算单元叫做神经元, 我们的神经系统中大约有860亿个神经元, 它大约与个![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(1).gif)突触相连。<br>
&emsp;&emsp;图1显示了一个生物神经元。图2显示了相应的数学模型。在生物神经元的绘制中, 每个神经元接收来自其树突的传入信号, 然后沿轴突产生输出信号, 轴突通过它分支上的突触连接到其他神经元。<br>
&emsp;&emsp;在神经元的相应数学计算模型中, 沿轴突传播的信号![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(3).gif)与系统中另一个神经元的树突![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(4).gif)进行乘法运算，这种运算是基于该突触处的权重![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn.gif)。这个想法是, 这个的主要目的是通过网络学习，得到突触的权重或者说它的强度![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(2).gif)，它们是控制一个特定神经元对另一个神经元的影响程度。<br>
&emsp;&emsp;此外，在图2的基本计算模型中，树突将信号传送到主细胞体，并将其全部相加。如果最终结果超过某个阈值，神经元就会在计算模型中被激活。<br>
&emsp;&emsp;另外，值得一提的是我们需要控制轴突输出的峰值，所以我们使用了一种叫做激活函数的东西。实际上，一个常见的激活函数选择是sigmoid函数，因为它需要一个实值输入(求和以后的信号强度)，并将其压缩为0到1之间。我们将在下面的部分中看到这些激活函数的详细信息:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/1.png)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 1 "大脑的计算单元"<br>
&emsp;&emsp;生物模型有相应的基本数学模型::<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/2.png)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 2 计算机中的神经元模型<br>
&emsp;&emsp;神经网络的基本计算单位是神经元，通常称为节点或单位。它从其他节点或外部源接收输入并计算输出。每个输入都有一个相关的权重(w)，它是根据相对于其他输入的重要性来分配的。该节点将函数![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(5).gif)(我们稍后定义)应用于其输入的加权和。因此，神经网络的基本计算单元一般称为神经元或者称为节点或单元。<br>
&emsp;&emsp;这个神经元从前一个神经元甚至外部源接收它的输入，然后它对这个输入做一些处理来产生所谓的激活。这个神经元的每个输入都与它自己的权重相关联，权重代表了这个神经元的强度、连接以及输入的重要性。<br>
&emsp;&emsp;因此,神经网络的的最终输出是由其权重![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(2).gif)加权的输入求和,然后神经元通过激活函数传递求和输出。<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/3.png)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 3 "单个神经元<br>
## 激活函数
&emsp;&emsp;神经元的输出如图3所示，并通过一个向输出引入非线性的激活函数进行计算。这个![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(5).gif)叫做激活函数。激活功能的主要目的是:<br>
&emsp;&emsp;&emsp;&emsp;1.在神经元的输出中引入非线性。这一点很重要, 因为大多数真实世界的数据是非线性的, 我们希望神经元学习这些<br>
&emsp;&emsp;&emsp;&emsp;&emsp;非线性表示。<br>
&emsp;&emsp;&emsp;&emsp;2.将输出压缩在一个特定范围内。<br>
&emsp;&emsp;每个激活函数 (或非线性) 都取一个数字, 并对其进行某种固定的数学运算。在实践中，您可能会遇到一些激活函数。<br>
&emsp;&emsp;接下来我们将简要介绍最常见的激活函数。<br>
### sigmoid
&emsp;&emsp;在历史上，sigmoid函数的激活功能在研究人员中广泛使用。该函数接受实值输入并将其压缩到0 - 1之间，如下图所示:<br>
