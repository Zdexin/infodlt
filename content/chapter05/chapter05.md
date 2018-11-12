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
&emsp;&emsp;生物模型有相应的基本数学模型:<br>
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
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/4.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 4 "sigmoid函数"<br>
### Tanh
&emsp;&emsp;Tanh 是另一个激活函数, 它容忍一些负值。Tanh 接受一个实值输入, 并将它们限制到 [-1、1]<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(6).gif)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/8.png)<br>
### ReLU
&emsp;&emsp;整流线性单元(ReLU)不能容忍负值，因为它接受实值输入并将其阈值设为零(将负值换成零):<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(1).gif)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/5.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 6 "Relu"激活函数<br>
&emsp;&emsp;**偏置的重要性:** 偏置的主要功能是为每个节点提供一个可训练的常量值(除了节点接收的正常输入之外)。请参阅(https://stackoverflow/com/quertions/2480650/role–of–bias–in–neural–networks) 以了解更多关于偏置在神经元中的作用。<br>
## 前馈神经网络
&emsp;&emsp;前馈神经网络是第一个也是最简单的人工神经网络。它包含多层排列的多个神经元(节点)。相邻层的节点之间有连接或边。所有这些连接都有相关的权重。<br>
&emsp;&emsp;前馈神经网络的一个例子如图7所示:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/6.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图 7 "前馈神经网络" 示例<br>
&emsp;&emsp;在前馈网络中，信息只以一个方向向前移动，从输入节点、通过隐藏节点(如果有的话)和输出节点。网络中没有循环 (前馈网络的这种特性不同于循环神经网络，在循环神经网络中节点之间的连接形成一个循环)。<br>
## 多层网络的需求
&emsp;&emsp;**多层感知器(MLP)** 包含一个或多个隐藏层(除了一个输入层和一个输出层)。单层感知器只能学习线性函数，而MLP也可以学习非线性函数。<br>
&emsp;&emsp;图7显示了一个隐藏层的 MLP。请注意, 所有连接都具有与之关联的权重, 但仅在图中显示三个权重 (![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(7).gif) 和![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(8).gif))。<br>
&emsp;&emsp;**输入层:** 输入层有三个节点。偏置节点的值为1。其他两个节点以X1和X2作为外部输入。如前所述，输入层不执行任何计算，因此输入层节点的输出分别为1、X1和X2，并被输入到**隐藏层**中。<br>
&emsp;&emsp;**隐藏层:隐藏层**也有三个节点，偏置节点的输出为1。**隐藏层**中其他两个节点的输出取决于**输入层**(1、X1和X2)的输出以及与连接(边)相关的权重。记住，![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(5).gif)指的是激活函数。然后将这些输出传送到输出层的节点。<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/9.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图"8"具有一个隐藏层的多层感知器<br>
&emsp;&emsp;**输出层:** 输出层有两个节点;它们从**隐藏层**获取输入，并对突出显示的隐藏节点执行类似的计算。这些计算结果的计算值(Y1和Y2)作为多层感知器的输出。<br>
&emsp;&emsp;给定一组特征![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(9).gif)和一个目标![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(10).gif)，多层感知器可以学习特征和目标之间的关系，进行分类或回归。<br>
&emsp;&emsp;让我们举一个例子来更好地理解多层感知器。假设我们有以下学生标记数据集：<br>
**表1-学生标记数据集示例：**<br>

学习时间|期中测试成绩|期末测试成绩
-|-|-
35|67|Pass
12|75|Fail
16|89|Pass
45|56|Pass
10|90|Fail
<br>
&emsp;&emsp;这两个输入列显示学生学习的小时数和学生获得的期中分数。最终结果列可以有两个值, 1 或 0, 表示学生是否在最后一学期通过。例如, 我们可以看到, 如果学生学习35小时, 并在期中获得了67分，他/她最终通过了最后一学期。<br>
&emsp;&emsp;现在, 假设我们想预测一个学生学习25小时, 在期中考试中得到70分是否会通过最后一学期:<br>
**表2-最后学期结果未知的样本学生:**

学习时间|期中测试成绩|期末测试成绩
-|-|-
36|70|?
<br>
&emsp;&emsp;这是一个二元分类问题，MLP可以从给定的例子(训练数据)中学习，并在给定一个新的数据点时做出有根据的预测。我们将很快看到MLP如何学习这种关系。<br>
## 培训我们的 MLP-反向算法
&emsp;&emsp多层感知器学习的过程称为反向算法。我建议阅读果壳网上Hemanth Kumar给出的答案和解释, (https://www.Quora.com/How–do–you–explain–back–propagation–algorithm–to–a–beginner–in–neural–network/answer/Hemanth–Kumar–Mantri) (稍后引述)。<br>
&emsp;&emsp;**误差的反向传播**，通常缩写为BackProp是人工神经网络(ANN)训练的几种方式之一。它是一种监督训练方案，也就是说，它从标记的训练数据中学习(有一个监督者，来指导它的学习)。<br>
&emsp;&emsp;简单来说，BackProp就像“**从错误中学习**”。每当人工神经系统出现错误时，监督者就会予以纠正。<br>
&emsp;&emsp;神经网络由不同层次的节点组成;输入层、中间隐藏层和输出层。相邻层节点之间的连接具有与其相关联的“权值”。学习的目标是为这些边分配正确的权重。给定一个输入向量，这些权重决定了输出向量是什么。<br>
&emsp;&emsp;在监督学习中, 训练集被标记。这意味着, 对于某些给定的输入, 我们知道预期输出 (标签)。<br>
&emsp;&emsp;反向算法：<br>
&emsp;&emsp;最初，所有的边缘权重都是随机分配的。对于训练数据集中的每一个输入，ANN都被激活，并且它的输出可以被观察到。将此输出与我们已经知道的期望输出进行比较，错误将“传播”回前一层。注意到这个错误，并相应地“调整”权重。重复此过程，直到输出错误低于预定的阈值。<br>
&emsp;&emsp;一旦上述算法终止，我们就有了一个“完成了学习的”ANN，我们认为它已经准备好处理“新的”输入。这个ANN据说从几个例子(标记数据)和它的错误(错误传播)中学到了东西。<br>
&emsp;&emsp;---hHemanth Kumar.<br> 
&emsp;&emsp;现在我们已经了解了反向传播的工作原理，让我们回到我们的学生标记数据集。<br>
&emsp;&emsp;图8中所示的MLP在输入层中有两个节点，它们占用所研究的输入小时和期中标记。它还有一个带有两个节点的隐藏层。输出层也有两个节点;上节点输出通过的概率，下节点输出失败的概率。<br>
&emsp;&emsp;在分类应用中, 我们广泛使用 softmax 函数(http://cs23ln.github/linear–classify/#softmax) 作为 MLP 输出层中的激活函数, 以确保输出为概率, 并且他们加起来等于1。softmax 函数接受一个任意实值的向量，并将其压缩为一个值介于0和1之间的向量，其总和为1因此, 在这个例子中:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/1.jpg)<br>

