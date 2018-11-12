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
**表2-最后学期结果未知的样本学生:


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
&emsp;&emsp;hHemanth Kumar.<br> 
&emsp;&emsp;现在我们已经了解了反向传播的工作原理，让我们回到我们的学生标记数据集。<br>
&emsp;&emsp;图8中所示的MLP在输入层中有两个节点，它们占用所研究的输入小时和期中标记。它还有一个带有两个节点的隐藏层。输出层也有两个节点;上节点输出通过的概率，下节点输出失败的概率。<br>
&emsp;&emsp;在分类应用中, 我们广泛使用softmax函数(http://cs23ln.github/linear–classify/#softmax )<br>作为MLP输出层中的激活函数,以确保输出为概率,并且他们加起来等于1。softmax函数接受一个任意实值的向量，并将其压缩为一个值介于0和1之间的向量，其总和为1因此, 在这个例子中:Probability(Pass)+Probablility(Fail)=1。<br>
### 步骤1-向前传播
&emsp;&emsp;网络中的所有权值都是随机初始化的。让我们考虑一个特定的隐藏层节点，并将其称为V。假设从输入到该节点的连接的权重是![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(12).gif)和![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(13).gif)(如图所示)。<br>
&emsp;&emsp;然后网络将第一个训练样本作为输入(我们知道，对于输入35和67，通过的概率是1) <br>
&emsp;&emsp;&emsp;&emsp;网络输入= [35, 67]<br>
&emsp;&emsp;&emsp;&emsp;期望的网络输出(目标) = [1, 0]<br>
&emsp;&emsp;然后考虑节点输出V，可计算如下(![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(5).gif)为sigmoid激活函数):<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(14).gif)<br>
&emsp;&emsp;同样计算隐藏层中其他节点的输出。隐藏层中两个节点的输出作为输出层中两个节点的输入。这使我们能够计算输出层中的两个节点的输出概率。<br>
&emsp;&emsp;假设输出层中两个节点的输出概率分别为0.4和0.6(由于权值是随机分配的，输出也是随机的)。我们可以看到，计算出的概率(0.4和0.6)与期望概率(分别为1和0)相差甚远，因此，网络的输出是不正确的。<br>
## 步骤2-反向传播和更新权重
&emsp;&emsp;我们计算输出节点的总误差，通过网络传播这些误差并使用反向传播计算梯度。然后，采用梯度下降法等优化方法对网络中的权值进行调整，以减少输出层的误差。<br>
&emsp;&emsp;假设与所考虑的节点相关联的新权值是![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(16).gif)。)和![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(17).gif)(在反向传播和调整权值之后)。<br>
&emsp;&emsp;如果我们现在将相同的示例作为输入，并输入到网络中，那么网络的性能应该会比初始运行时更好，因为现在已经对权重进行了优化，以最小化预测中的错误。与前面的[0.6，-0.4]相比，输出节点的错误现在减少到[0.2，-0.2]。这意味着我们的网络已经学会正确地分类我们的第一个训练样本。<br>
&emsp;&emsp;我们对数据集中的所有其他训练样本重复这个过程。然后,我们的网络已经学会了这些例子。<br>
&emsp;&emsp;如果我们现在想要预测一个学习了25小时，在期中有70分的学生是否能通过期末考试，我们要通过正向传播步骤，找到通过和失败的输出概率。<br>
&emsp;&emsp;在这里，我避免使用数学方程和对梯度下降法等概念的解释，而是尝试为算法开发一种直觉。有关反向传播算法的更多数学讨论，请参阅以下链接: <br>
&emsp;&emsp;(http://home.agh.edu.pl/%7Evlsi/AI/backp_t_en/backprop.html) <br>
## TensorFlow术语—回顾
&emsp;&emsp;在本节中，我们将概述TensorFlow库以及基本的TensorFlow应用程序的结构。TensorFlow是一个用于创建大型机器学习应用程序的开源库;它可以在各种各样的硬件上建模计算，从android设备到异构的多gpu系统。<br>
&emsp;&emsp;TensorFlow使用一种特殊的结构，以便在不同的设备(如cpu和gpu)上执行代码。计算被定义为一个图，每个图都由operations组成，也被称为ops，所以每当我们使用TensorFlow时，我们都会在一个图中定义一系列的操作。<br>
&emsp;&emsp;要运行这些操作，我们需要将计算图启动到会话控制中。会话控制将操作转换并将其传递给执行设备。<br>
&emsp;&emsp;例如，下面的图像表示TensorFlow中的图。![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(2).gif)，![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(18).gif)和![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(19).gif)是这个图边缘上的张量。矩阵是对张量W和x的运算;在此之后，调用Add，我们将前一个操作符的结果与![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/CodeCogsEqn%20(19).gif)相加，每个操作的结果张量与下一个操作交叉，直到最后，在那里可以得到想要的结果。<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter05/10.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图"9"TensorFlow计算图<br>
&emsp;&emsp;为了使用TensorFlow，我们需要导入库;我们给它命名为UG这样我们就可以通过写入UG点来访问一个模块然后是模块的名称:<br>
```python
    import tensorflow as tf
```
&emsp;&emsp;要创建第一个图，我们将首先使用不需要任何输入的源操作。这些源操作或源操作将把它们的信息传递给其他操作，这些操作将实际运行计算。<br>
&emsp;&emsp;让我们创建两个将输出数字的源操作。我们将它们定义为 A 和 B, 您可以在下面的代码段中看到:<br>
```python
    A=tf.constant([2])
    B=tf.constant([3])
```
&emsp;&emsp;然后, 我们将定义一个简单的计算操作tf.add(),用于求和两个元素。您也可以使用C=A+B,如下面的代码所示:<br>
```python
    C = tf.add(A,B)
    #C = A + B is also a way to define the sum of the terms
```
&emsp;&emsp;由于需要在会话的上下文中执行图, 因此我们需要创建一个会话对象:<br>
```python
    session = tf.session()
```
&emsp;&emsp;为了查看图表， 让我们运行会话以从以前定义的 C 操作获取结果:<br>
```python
   result = session.run(C) print(result)
   Output:
   [5]
```
&emsp;&emsp;你可能认为仅仅把两个数字加在一起就需要做很多工作，但是理解TensorFlow的基本结构是非常重要的。一旦你这样做了，你可以定义任何你想要的计算;同样，TensorFlow的结构允许它在不同的设备(CPU或GPU)甚至集群中处理计算。如果您想了解更多这方面的信息，可以运行该方法 tf. device ().<br>
&emsp;&emsp;同样，你也可以自由地对TensorFlow的结构进行实验，以便更好地了解它是如何工作的。如果您想要得到TensorFlow支持的所有数学操作的列表，您可以查看文档。<br>
&emsp;&emsp;现在, 您应该了解 TensorFlow 的结构以及如何创建基本应用程序。<br>
## 使用 TensorFlow 定义多维数组
现在, 我们将尝试使用 TensorFlow 定义这样的一个数组:<br>
```python
    salar_var = tf.constant([4]) 
    vector_var = tf.constant([5,4,2])
    matrix_var = tf.constant([[l,2,3],[2,2,4],[3,5,5]])
    tensor = tf.constant( [ [[l,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,l0],[9,l0,ll]] ] )
    with tf.session() as session: 
        result = session.run(salar_var)
        print "Scalar (l entry):\n %s \n" % result 
        result = session.run(vector_var)
        print "Vector (3 entries) :\n %s \n" % 
        result result = session.run(matrix_var)
        print "Matrix (3*3 entries):\n %s \n" % result
        result = session.run(tensor)
        print "Tensor (3*3*3 entries) :\n %s \n" % result
    Output:
    scalar (l entry): 
    [2]

    Vector (3 entries) : 
    [5 6 2]

    Matrix (3*3 entries): 
    [[l 2 3]
    [2 3 4]
    [3 4 5]]

    Tensor (3*3*3 entries) : 
    [[[ l	2	3]
    [	2	3	4]
    [	3	4	5]]
    [[	4	5	6]
    [	5	6	7]
    [	6	7	8]]
    [[	7	8	9]
    [	8	9	l0]
    [	9	l0	ll]]]
```
现在您已经了解了这些数据结构，我鼓励您使用以前的一些函数来了解它们的行为，根据它们的结构类型:
```python
    Matrix_one = tf.constant([[l,2,3],[2,3,4],[3,4,5]])
    Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]
    first_operation = tf.add(Matrix_one, Matrix_two) 
    second_operation = Matrix_one + Matrix_two
        with tf.session() as session:
        result = session.run(first_operation)
        print "Defined using tensorflow function :" 
        print(result)
        result = session.run(second_operation) 
        print "Defined using normal expressions :" 
        print(result)

    Output:
    Defined using tensorflow function : 
    [[3 4 5]
    [4 5 6]
    [5 6 7]]
 

    Defined using normal expressions : 
    [[3 4 5]
    [4 5 6]
    [5 6 7]]

```
&emsp;&emsp;有了常规的符号定义和Tensorflow函数，我们可以得到一个元素的乘法，也称为**Hadamard乘积**。但如果我们想要正则矩阵乘积呢? 我们需要使用另一个名为tf.matmul()的TensorFlow函数<br>
```python
    Matrix_one = tf.constant([[2,3],[3,4]])
    Matrix_two = tf.constant([[2,3],[3,4]]) first_operation = tf.matmul(Matrix_one, Matrix_two) with tf.3ession() as session:
    result = session.run(first_operation)
    print "Defined using tensorflow function :" 
    print(result)

    Output:
    Defined using tensorflow function : 
    [[l3 l8]
    [l8 25]]
```
&emsp;&emsp;我们也可以自己定义这个乘法，但是有一个函数已经这样做了，所以没有必要重新定义这个过程!<br>
