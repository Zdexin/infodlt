# 第10章 递归神经网络-语言建模
&emsp;&emsp; 循环神经网络（RNN）是一种广泛用于自然语言处理的深层学习结构。这组 结构使我们能够为当前预测提供相关信息，并且还具有处理任何输入序列中的长期依赖关系的特定结构。在本章中，我们将演示如何构造序列到如何构造序列模型，这将在NLP中的许多应用中都有用。我们将通过构建字符型语言模型来演示这些概念，并查看我们的模型如何生成与原始输入序列类似的语句。
本章将讨论以下主题：
#### •递归神经网络背后的感知
#### •LSTM（长短期记忆神经网络）网络
#### •语言模型的实现
## 递归神经网络背后的感知
&emsp;&emsp; 到目前为止，我们所处理的所有深度学习结构都没有机制来记忆它们以前收到的输入。例如，如果给前馈神经网络（FNN）输入一系列字符，例如HELLO，当输入到达E时，您会发现它没有保存任何信息即忘记它只读取H。这是基于序列学习的严重问题。而且由于它没有以前读过的字符的记忆，这种神经网络很难通过训练来预测下一个字符。这对于语言建模、机器翻译、语音识别等许多应用都没有意义。<br>
&emsp;&emsp; 由于这个特定的原因，我们将介绍RNNs（递归神经网络），一组深层学习体系结构，它们确实保存了信息并记住了它刚刚遇到的内容。让我们演示一下RNNS应该如何处理相同的输入序列。<br>
&emsp;&emsp; 字符，HELLO。当RNN信元/单元接收E作为输入时，它也接收较早接收到的字符H。将当前字符和过去字符作为输入提供给RNN单元为这些体系结构（即短期内存）提供了很大的作用；它还使得这些体系结构可用于预测/猜测H之后最有可能的字符（即L），在这个特定的序列中可能具体字母。<br>
&emsp;&emsp; 我们已经看到，以前的体系结构为它们的输入分配权重；RNNS遵循相同的优化过程，为它们的多个输入分配权重，这就是现在和过去。因此，在这种情况下，神经网络将给它们中的每一个输入分配两个不同的权重矩阵。为了做到这一点，我们将使用梯度下降和较重的反向传播（BPTT）方法。
## 递归神经网络的结构
&emsp;&emsp; 根据我们使用以前的深层学习结构的背景，你会发现为什么递归神经网络是特殊的。先前我们所了解的架构在输入或训练方面是不够灵活的。它们接受固定大小的序列/矢量/图像作为输入，并产生另一个固定大小结果的作为输出。RNN架构在某种程度上是不同的，因为它们允许将一个序列作为输入进行发馈传送，并将另一个序列作为输出，或者仅在输入/输出中具有序列，如图1所示。这种灵活性对于如语言建模和情感分析的多个应用非常有用：
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap1.JPG)<br>
图10.1：RNNs在输入或输出形状方面的柔性μ<br>
&emsp;&emsp; 这些架构背后的直觉是模仿人类处理信息的方式。在任何典型的谈话中，你对某人的话的理解完全取决于他之前说过的话，你甚至可以根据他刚才说的来预测他接下来要说什么。<br>
&emsp;&emsp; 在递归神经网络的情况下，应该遵循完全相同的过程。例如，假设你想把一个特定的词翻译成句子。不能使用传统的FNNS（反馈神经系统）来实现这一点，因为它们不能将之前单词的翻译作为我们想要翻译的当前单词的输入，并且这可能导致错误的翻译，因为该单词周围缺乏上下文相关联系的信息。<br>
递归神经网络确实能够保存关于过去的信息，并且它们具有某种循环规律，可以做到在任何给定点时将先前学习的信息用于当前预测：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap2.JPG)<br>
图10.2：递归神经网络体系结构，它的具有循环保存过去步骤的信息能力<br>
&emsp;&emsp; 在图2中，我们有一些称为A的神经网络，它接收输入X并产生和输出H。此外，它能够通过这个循环从过去的步骤中接收信息。<br>
&emsp;&emsp; 这个循环似乎不清楚，但是如果我们使用图2的展开版本，会发现它非常简单和直观，并且RNN只是相同网络（可以是普通FNN）的重复版本，如图3所示：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap3.JPG)<br>
图3：递归神经网络体系结构的展开版本<br>
&emsp;&emsp; 递归神经网络的这种直观体系结构及其在输入/输出形状方面的灵活性，使它们非常适合于基于序列的学习任务，例如机器翻译、语言建模、情感分析、图像字幕等。
## 递归神经系统的例子
&emsp;&emsp; 现在，我们直观地理解了递归神经网络是如何工作的，以及它在基于序列的不同有趣的示例中将如何适用，让我们仔细看看这些有趣的例子。
## 字符级语言模型
&emsp;&emsp; 语言建模是语音识别、机器翻译等应用中的一项重要任务。在这一节中，我们将尝试模仿递归神经网络的训练过程，并深入了解这些网络是如何工作的。我们将构建一个字符操作的语言模型。因此，我们将给我们的网络提供一大块文本，其目的是试图建立一个预测下一个字符的概率分布，给定前面的字符，生成类似于我们在训练过程中输入的文本。<br>
&emsp;&emsp; 例如，假设我们有一个只有四个字母的语言作为词汇-HELO。这个任务是训练一个递归神经网络上的特定输入序列的字符，如Hello。在这个特定的例子中，我们有四个训练样本：<br>
&emsp;&emsp; 1、在第一个输入字符H的前提下计算字符E的概率；<br>
&emsp;&emsp; 2、给定He的作为上文，计算字符L的概率;<br>
&emsp;&emsp; 3、根据HEL的上文计算字符L的概率；<br>
&emsp;&emsp; 4、最后，在给定上文为HELL的情况下，计算字符O的概率。<br>
&emsp;&emsp; 正如我们在前几章所了解到的，深度学习通常属于机器学习技术，它只接受实值数字作为输入。因此，我们需要某种方式转换或编码输入字符的数值形式。为此，我们将使用单热矢量编码，这是一种通过具有零向量来编码文本的方法，除了向量中的单个条目之外，向量中的单个条目，即我们试图建模的这种语言词汇表中的字符的索引（在本例中为helo）。在编码我们的训练样本之后，我们将一次将它们提供给循环神经网络类型的模型。每个给定字符循环神经类型的模型的输出值将是一个4维向量（向量的大小对应于词汇表的大小），它表示词汇表中的每个字符在给定输入字符之后成为下一个字符的概率。图4阐明了这一过程：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap4.JPG)<br>
&emsp;&emsp; 图4_循环神经网络模型的示例，其中以单热矢量编码字符作为输入，并且输出将分布在词汇表上，该词汇表表示当前字符之后最相似的μ字符<br>
&emsp;&emsp; 如图4所示，可以看到，我们将输入序列h中的第一个字符提供给模型，并且输出是表示对下一个字符的信任度的4维向量。因此，它具有输入h之后的下一个字符的置信度为1.0，e的下一个字符的置信度为2.2，对l的下一个字符的置信度为-3.0，最终对o的下一个字符的置信度为4.1。在这个特定的例子中，我们知道正确的下一个字符将是E，基于我们的训练序列Hello。因此，在训练这种循环神经型网络时，我们的主要目标是提高e作为下一个字符的置信度，并降低其他字符的置信度。为了进行这种优化，我们将使用梯度下降和反向传播算法来更新权重，并影响神经网络，以便我们的下一个字符（e，等等）为其他三个训练示例产生更高的置信度。<br>
&emsp;&emsp; 正如所看到的，循环神经网络的输出在接下来的词汇表的所有字符上产生一个置信度分布。我们可以把这个置信度分布转换成一个概率分布，因为概率需要加到1，下一个字符概率的增加将导致其他概率的降低，。对于这个特定的修改，我们可以使用标准的SoftMax分类器进行分类到每个输出向量。<br>
&emsp;&emsp; 为了从这类网络中生成文本，我们可以将初始字符馈送给模型，并获得下一个字符的概率分布，然后从这些字符中采样并将其作为输入反馈给模型。我们要生成具有所需长度的文本，就可以通过重复多次这个过程来获得所需长度的字符序列。<br>
## 使用莎士比亚数据的语言模型
&emsp;&emsp; 从前面的例子中，我们可以通过得到模型来生成文本。但是神经网络令我们惊讶得是，它不仅会生成文本，而且还会学习训练数据的样式和结构。我们可以通过训练循环神经网络类型的模型来演示这个有趣的过程，该模型针对特定类型的文本，这些文本具有特定的结构和风格，比如以下莎士比亚的作品。<br>
&emsp;&emsp; 让我们来看一下从训练网络生成的输出：<br>
Second Senator:<br>
They are away this miseries, produced upon my soul,<br>
Breaking and strongly should be buried, when I perish The earth and thoughts of many states.<br>
&emsp;&emsp; 虽然该神经网络只知道怎样一次生成一个字符，但它生成的文本和以及内容是具有情感意义的，它们实际上具有莎士比亚作品的结构和风格。<br>
## 梯度消失问题
&emsp;&emsp; 在训练这些RNN型结构的同时，我们使用梯度下降和时间的反向传播，这给许多基于序列的学习任务带来了一些成功。但是由于梯度的性质和使用了快速训练策略，所有梯度值将明显趋近于零或消失。<br>
&emsp;&emsp; 这个过程引入了梯度消失的问题，许多研究者会陷入其中。在本章的后面，我们将讨论研究人员如何处理这些问题，并产生普通的递归神经网络的变异来克服这个问题：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap5.JPG)<br>
图5：梯度消失问题
## 长期依赖问题
&emsp;&emsp; 研究人员所面临的另一个挑战性问题是人们在文本中可以找到的长期依赖性。例如，如果像我以前在法国生活一样，我学会了说话……顺序中的下一个显而易见的词是法语。<br>
&emsp;&emsp; 在这种情况下，普通的递归神经网络将能够处理它，因为它具有短期依赖性，如图6所示：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap6.JPG)<br>
图6：文本的短期依赖<br>
&emsp;&emsp; 另一个例子是，如果有人说我以前住在法国…然后他开始描述那里的生活，最后我学会了说法语。因此，为了让模型预测他/她在序列末尾所学的语言，模型需要一些关于早期单词live和法国的信息。如果模型无法跟踪文本中的长期依赖性，那么它将无法处理此类情况：
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap7.JPG)<br>
图7：文本中长期依赖性的挑战<br>
&emsp;&emsp; 为了处理文本中逐渐消失的梯度和长期依赖性，研究人员引入了一种称为长短期记忆网络（LSTM）的普通循环神经网络。
## 长短期记忆网络
&emsp;&emsp; 长短期记忆网络是循环神经网络的一种变形，用于帮助解决文本学习中的长期依赖关系。长短期记忆网络最初由Hochreiter &Schmidhuber(1997)引入，已经许多研究者对其进行了研究，并在许多领域产生了有趣的结果。<br>
&emsp;&emsp; 这些类型的体系结构将能够处理文本中的长期依赖问题，主要是因为它们的内部体系结构决定的。<br>
&emsp;&emsp; 长短期记忆网络与普通循环神经网络类似，因为随着时间推移，它有一个重复模块，但是这个重复模块的内部结构与普通循环神经网络不同。它包含了更多的被遗忘和已经更新信息的层次：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap8.JPG)<br>
图8：包含单个层的标准循环神经网络中的重复模块<br>
&emsp;&emsp; 如前所述，普通递归神经网络具有单个神经网络层，但是长短期记忆网络具有以特殊方式相互作用的四个不同层。这种特殊的交互使得长短期记忆网络在很多领域都能很好地工作，我们将在构建语言模型示例时看到：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap9.JPG)<br>
图9：包含四个相互作用层的长短期记忆网络中的重复模块<br>
&emsp;&emsp; 关数学细节以及四个层如何实现交互的更多细节，可以看一下这个有趣的教程：
`http://colah. github.io/posts/2Ol5–O8–Understanding–L3TMs/`
## 长短期记忆网络的工作原理是什么？
&emsp;&emsp; 在我们的普通的长短期记忆网络架构中，第一步是确定哪些信息不是必需的，并且通过丢弃这些信息来为更重要的信息留出更多的空间。为此，我们有一个称为遗忘栅极层的层，它查看以前的输出ht-1和当前输入xt，并决定要丢弃哪些信息。<br>
有&emsp;&emsp; 长短期记忆网络体系结构的下一步是决定哪些信息值得保存，并存储在单元格中。这是通过两个步骤完成的：
1、一个称为输入控制门的层，它决定单元的前一个状态的值需要更新。<br>
2、第二步是生成一组新的候选值，这些值将被添加到单元格中。<br>
&emsp;&emsp; 最后，我们需要决定长短期记忆网络单元将输出什么。此输出将基于我们的单元格状态，但将是经过过滤的版本。
## 语言模型的实现
&emsp;&emsp; 在本节中，我们将构建一个通过字符操作的语言模型。对于这个模型的实现，我们将使用安娜·卡列尼娜的小说作为实例，看看网络将如何学习实现文本的结构和风格：
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap10.JPG)<br>
图10：字符级循环神经网络的一般体系结构<br>
&emsp;&emsp; 这个网络结构是基于Andrej Karpathy's在循环神经网络系统实现的。<br>
&emsp;&emsp; 我们将构建一个基于安娜·卡列尼娜小说的人物的循环神经网络，它能够根据书中的文本生成新的文本。将找到包含在该实现的现实中的.txt文件。让我们先为这个字符级模型的实现导入必要的库：
```import numpy as np import tensorflow as tf
from collections import namedtuple
```
&emsp;&emsp; 首先，我们需要通过加载数据集并将其转换成整数来准备训练数据集。因此，我们将字符转换成整数编码，然后将它们编码为整数，这使得它作为模型的输入变量变得简单且易于使用：
```
#reading the Anna Karenina novel text file with open('Anna_Karenina.txt', 'r') as f:
textlines=f.read()
#Building the vocan and encoding the characters as integers language_vocab = set(textlines)
vocab_to_integer = (char: j for j, char in enumerate(language_vocab)} integer_to_vocab = dict(enumerate(language_vocab))
encoded_vocab = np.array([vocab_to_integer[char] for char in textlines], dtype=np.int32)
```
&emsp;&emsp; 那么，让我们来看看Anna Karenina文本中的前200个字符：
```
extlines[:2OO] Output:
"Chapter l\n\n\nHappy families are all alike; every unhappy family is unhappy in its own\nway.\n\nEverything was in confusion in the Oblonskys' house. The wife had\ndiscovered that the husband was carrying on"
```
&emsp;&emsp; 我们还把字符转换成一种方便的形式--整数。那么，让我们来看看字符的编码版本：
```
encoded_vocab[:2OO] Output:
array([7O, 34, 54, 29, 24, l9, 76, 45, 2, 79, 79, 79, 69, 54, 29, 29, 49,
45, 66, 54, 39, l5, 44, l5, l9, l2, 45, 54, 76, l9, 45, 54, 44, 44,
45, 54, 44, l5, 27, l9, 58, 45, l9, 3O, l9, 76, 49, 45, 59, 56, 34,
54,	29,	29,	49,	45,	66,	54,	39,	l5,	44,	49,	45,	l5,	l2,	45,	59,	56,
34,	54,	29,	29,	49,	45,	l5,	56,	45,	l5,	24,	l2,	45,	ll,	35,	56,	79,
35,	54,	49,	53,	79,	79,	36,	3O,	l9,	76,	49,	24,	34,	l5,	56,	l6,	45,
35,	54,	l2,	45,	l5,	56,	45,	3l,	ll,	56,	66,	59,	l2,	l5,	ll,	56,	45,
l5, 56, 45, 24, 34, l9, 45, l, 82, 44, ll, 56, l2, 27, 49, l2, 37,
45, 34, ll, 59, l2, l9, 53, 45, 2l, 34, l9, 45, 35, l5, 66, l9, 45,
34, 54, 64, 79, 64, l5, l2, 3l, ll, 3O, l9, 76, l9, 64, 45, 24, 34,
54, 24, 45, 24, 34, l9, 45, 34, 59, l2, 82, 54, 56, 64, 45, 35, 54,
l2, 45, 3l, 54, 76, 76, 49, l5, 56, l6, 45, ll, 56], dtype=int32)
```
&emsp;&emsp; 由于网络处理单个字符的时候，它类似于一个分类问题，在这个问题中我们试图从前一个文本中预测下一个字符。下面是我们的网络需要选择的类。因此，我们将一次向模型反馈传送一个字符，并且模型将通过对可能接下来出现的字符数(词汇)产生概率分布来预测下一个字符，该概率分布应该是需要从以下几个类中挑选出来的类：
```
len(language_vocab) 
Output:
83
```
&emsp;&emsp; 由于我们将使用随机梯度法下降来训练我们的模型，我们需要将我们的数据转换成训练多个批次。
## 生成小批量训练集
&emsp;&emsp; 在这一节中，我们将我们的数据分成小批次用于训练模型。因此，小批次训练集将由期望序列数的有序序列步骤组成。那么，让我们看看图11中的一个可视化示例：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap11.JPG)<br>
图11说明批次和序列的例子<br>
&emsp;&emsp; 所以，现在我们需要定义一个函数，它将迭代经过编码的文本并生成批处理训练集。在这个函数中，我们将使用一个非常好的Python机制，叫做yield。<br>
&emsp;&emsp; 一个典型的批次将具有N×M个字符，其中N是序列的数目，M是序列步长的数目。为了获得数据集中可能批的数量，我们可以简单地将数据的长度除以所需的批大小，在获得该数量的可能批之后，我们可以知道需要操作的每个批中应该有多少字符。<br>
&emsp;&emsp; 之后，我们需要将数据集分割成期望数量的序列（N）。我们可以使用数组重塑（尺寸）。我们知道我们需要N个序列（num_seqs被使用，遵循代码），让我们做第一个维度的大小。对于第二个维度，可以使用-1作为占位符；它将填充数组，并提供适当的数据。在此之后，应该有一个N×（M`*`K)的数组，其中K是批数。<br>
&emsp;&emsp; 现在我们有了这个数组，我们可以遍历它来获得训练批次，每个批次都有N×M字符。对于每一个后续批次，窗口通过num_steps移动。最后，我们还希望为我们的输入和输出数组创建模型输入。创建输出值的这一步骤非常简单；记住目标是在一个字符上移位的输入。通常会看到第一个输入字符作为最后一个目标字符，所以像这样：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap14.JPG)<br>
&emsp;&emsp; 其中X是输入批次，Y是目标批次。<br>
&emsp;&emsp; 这个窗口的方法是使用排列的方法来采取规模为num_steps的步骤，从0到arr..[l]，每个序列中的步骤之和就是总数。这样，从范围中得到的整数总是指向批的开始，并且每个窗口都是num_steps宽的：<br>
```def generate_character_batches(data, num_seq, num_steps): '''Create a function that returns batches of size
num_seq x num_steps from data.
'''
# Get the number of characters per batch and number of batches num_char_per_batch = num_seq * num_steps
num_batches = len(data)//num_char_per_batch
# Keep only enough characters to make full batches data = data[:num_batches * num_char_per_batch]
# Reshape the array into n_seqs rows data = data.reshape((num_seq, –l))
for i in range(O, data.shape[l], num_steps):
# The input variables
input_x = data[:, i:i+num_steps]
# The output variables which are shifted by one
output_y = np.zeros_like(input_x)
output_y[:, :–l], output_y[:, –l] = input_x[:, l:], input_x[:, O] yield input_x, output_y
```
&emsp;&emsp; 因此，让我们通过使用一个15个序列和50个序列步骤来演示这个函数：
```
generated_batches = generate_character_batches(encoded_vocab, l5, 5O) input_x, output_y = next(generated_batches)
print('input\n', input_x[:lO, :lO]) print('\ntarget\n', output_y[:lO, :lO]) 
Output:
input
[[7O 34 54 29 24 l9 76 45 2 79]
[45	l9	44	l5	l6	l5	82	44	l9	45]
[ll	45	44	l5	l6	34	24	38	34	l9]
[45	34	54	64	45	82	l9	l9	56	45]
[45	ll	56	l9	45	27	56	l9	35	79]
[49	l9	54	76	l2	45	44	54	l2	24]
[45	4l	l9	45	l6	ll	45	l5	56	24]
[ll	35	45	24	ll	45	39	54	27	l9]
[82	l9	66	ll	76	l9	45	8l	l9	56]
[l2	54	l6	l9	45	44	l5	27	l9	45]]
target
[[34 54 29 24 l9 76 45 2 79 79]
[l9	44	l5	l6	l5	82	44	l9	45	l6]
[45	44	l5	l6	34	24	38	34	l9	54]
[34	54	64	45	82	l9	l9	56	45	82]
[ll	56	l9	45	27	56	l9	35	79	35]
[l9	54	76	l2	45	44	54	l2	24	45]
[4l	l9	45	l6	ll	45	l5	56	24	ll]
[35	45	24	ll	45	39	54	27	l9	33]
[l9	66	ll	76	l9	45	8l	l9	56	24]
[54	l6	l9	45	44	l5	27	l9	45	24]]
```
&emsp;&emsp; 下一步，我们将期待建立这个例子的核心，就是是长短期记忆网络模型。<br>
## 构建模型
&emsp;&emsp; 在使用长短期记忆网络构建人物级模型之前，值得一提的是被称为堆叠长短期记忆网络的东西。堆叠的长短期记忆网络在不同的时间尺度上查看你的信息是非常有用的。
## 长短期记忆网络的堆叠
&emsp;&emsp; 大多数研究人员正在使用堆叠长短期记忆网络挑战序列预测问题。堆叠长短期记忆网络体系结构可以定义为由多个长短期记忆网络层组成的长短期记忆网络模型。前面的长短期记忆网络层提供了序列输出，而不是单个值输出到长短期记忆网络层，如下所述。<br。
&emsp;&emsp; 具体而言，它是每个输入时间步长的一个输出，而不是用于所有输入时间步长的一个输出时间步长：
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap12.JPG)<br>
图12：构建长短期记忆网络模型<br>
&emsp;&emsp; 因此，在这个例子中，我们将使用这种堆叠的长短期记忆网络体系结构，从而提供更好的性能。
## 模型结构
&emsp;&emsp; 这是我们构建网络的地方。我们将把它分解成部分，以便更容易对每一个比特进行推理。然后，我们可以将它们与整个网络连接起来：
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter10/chapter_10image/ap13.JPG)<br>
图13：字符级模型体系结构
## 输入
&emsp;&emsp; 图13是字符级模型架构，我们首先定义占位符作为模型输入。模型的输入将是训练数据和目标。我们还将使用一个称为keep_probability概率的参数，用于帮助模型避免过度拟合：
```
def build_model_inputs(batch_size, num_steps):
# Declare placeholders for the input and output variables inputs_x = tf.placeholder(tf.int32, [batch_size, num_steps],
name='inputs')
targets_y = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
# define the keep_probability for the dropout layer keep_probability = tf.placeholder(tf.float32, name='keep_prob') return inputs_x, targets_y, keep_probability
```
## 构建一个长短期记忆网络神经元
&emsp;&emsp; 在本节中，我们将编写一个用于创建长短期记忆网络单元的函数，该函数将在隐藏层中使用。这个神经元将是我们模型的基石。因此，我们将使用TensorFlow创建这个单元格。让我们看看如何使用TensorFlow构建一个基本长短期记忆网络的单元。我们调用下面的代码行来创建具有参数num_units的长短期记忆网络单元。表示隐藏层中的单元数目：
`lstm_cell = tf.contrib.rnn.BasicL3TMCell(num_units)`
&emsp;&emsp; 为了防止过拟合，我们可以使用称为放弃或者丢弃的方法，这是一种通过降低模型的复杂度来防止模型数据过拟合的机制：
`tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_probability)`
&emsp;&emsp; 正如我们前面提到的，我们将使用堆叠长短期记忆模型架构；它将帮助我们从不同的角度来查看数据，并且发现在实际上已经执行得很好。为了定义层叠的长短期记忆网络在张量流中，我们可以使用
```
tf.contrib.rnn.MultiRNNCell function 
(link: https://www.tensorflow.org/versions/ rl.O/api_docs/python/tf/contrib/rnn/MultiRNNCell):
tf.contrib.rnn.MultiRNNCell([cell]*num_layers)
```
&emsp;&emsp; 最初，对于第一个单元格，以前没有信息，我们需要将单元格状态初始化为零。我们可以使用以下函数来完成：
`initial_state = cell.zero_state(batch_size, tf.float32)`
&emsp;&emsp; 所以，让我们一起来创建我们的长短期记忆网络神经元：
```
def build_lstm_cell(size, num_layers, batch_size, keep_probability):
### Building the L3TM Cell using the tensorflow function lstm_cell = tf.contrib.rnn.BasicL3TMCell(size)
# Adding dropout to the layer to prevent overfitting drop_layer = tf.contrib.rnn.DropoutWrapper(lstm_cell,
output_keep_prob=keep_probability)
# Add muliple cells together and stack them up to oprovide a level of more understanding
stakced_cell = tf.contrib.rnn.MultiRNNCell([drop_layer] * num_layers) initial_cell_state = lstm_cell.zero_state(batch_size, tf.float32) return lstm_cell, initial_cell_state
```
## 循环神经网络
&emsp;&emsp; 接下来，我们需要创建输出层，该输出层负责读取各个长短期记忆网络单元的输出并将其传递给全连接层。该层具有SoftMax输出，用于在输入一个字符之后预测下一个字符的概率分布。<br>
&emsp;&emsp; 如大家所知，我们已经为具有大小N×M字符的网络生成了输入批次的数据集，其中N是该批次中的序列数，M是序列步骤数。<br>
&emsp;&emsp; 在创建模型时，我们还在隐藏层中使用了L隐藏单元。基于隐藏单元的批量大小和数量，网络的输出将是一个大小为N×M×L的3D张量，这时我们将长短期记忆网络单元称为M次，每个序列有步骤一个。每次调用长短期记忆网络神经元都会产生一个大小为L的输出。最后，我们需要按照需求去做N个序列。<br>
&emsp;&emsp; 因此，我们把这个N×M×L的输出传递到一个完全连接的层（对于所有输出具有相同权重的都是相同的），但是在这样做之前，我们将输出整形形成2D张量，它具有（M×N）×L的形状。这种整形将使我们在输出上操作时变得更容易。因为新形状将更方便；每行的值表示长短期记忆网络神经元的输出L，因此它是每个序列和步骤的一行。<br>
&emsp;&emsp; 在得到新的形状之后，我们可以通过与权重进行矩阵乘法将其连接到具有SoftMax的全连接层。在长短期记忆网络单元格中创建的权重和我们在这里将要创建的权重默认具有相同的名称，在这种情况下，TensorFlow将引发错误。为了避免这个错误，我们可以使用TensorFlow函数包装在可变范围内创建新的权重和偏差变量tf.variable_scope().<br>
&emsp;&emsp; 在解释输出的形状以及我们将如何重塑它之后，为了简化工作，让我们继续编写build model output函数：
```
ef build_model_output(output, input_size, output_size):
# Reshaping output of the model to become a bunch of rows, where each row correspond for each step in the seq
sequence_output = tf.concat(output, axis=l)
reshaped_output = tf.reshape(sequence_output, [–l, input_size])
# Connect the RNN outputs to a softmax layer with tf.variable_scope('softmax'):
softmax_w = tf.Variable(tf.truncated_normal((input_size, output_size), stddev=O.l))
softmax_b = tf.Variable(tf.zeros(output_size))
# the output is a set of rows of L3TM cell outputs, so the logits will be a set
# of rows of logit outputs, one for each step and sequence logits = tf.matmul(reshaped_output, softmax_w) + softmax_b
# Use softmax to get the probabilities for predicted characters model_out = tf.nn.softmax(logits, name='predictions') return model_out, logits
```
## 训练损失
&emsp;&emsp; 接下来是训练损失。我们得到了对数损失函数和目标，并计算了SoftMax交叉熵损失。首先，我们需要对目标进行单热编码，我们将它们作为字符编码。然后，我们重塑单热目标，因此它是一个具有大小（M×N）×C的2D张量，其中C是我们拥有的类/字符的数量。请记住，我们重塑了长短期记忆网络的输出，并运行它们，通过一个完全连接层和张量C。因此，我们的对数损失函数也将具有大小（M×N）×C。<br>
&emsp;&emsp; 然后，我们运行对数损失函数和目标通过tf.nn.softmax_cross_entropy_with_logits 并且得到了损失的意义。
```
def model_loss(logits, targets, lstm_size, num_classes):
# convert the targets to one–hot encoded and reshape them to match the logits, one row per batch_size per step
output_y_one_hot = tf.one_hot(targets, num_classes) output_y_reshaped = tf.reshape(output_y_one_hot, logits.get_shape())
#Use the cross entropy loss
model_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_y_reshaped)
model_loss = tf.reduce_mean(model_loss) return model_loss
```
## 优化器
&emsp;&emsp; 最后，我们需要使用一种优化方法来帮助我们从数据集中学习一些东西。正如我们所知，普通递归神经网络有梯度爆炸和梯度消失的问题长短期记忆网络只解决了一个问题，即梯度值的消失，但是即使在使用长短期记忆网络之后，一些梯度值也会爆炸并无限增长。为了解决这个问题，我们可以使用称为梯度裁剪的工具，这是一种将爆炸到特定阈值的梯度裁剪的技术。<br>
&emsp;&emsp; 因此，让我们通过使用Adam优化来定义我们的优化器，用于学习过程：
```
def build_model_optimizer(model_loss, learning_rate, grad_clip):
# define optimizer for training, using gradient clipping to avoid the exploding of the gradients
trainable_variables = tf.trainable_variables()
gradients, _ = tf.clip_by_global_norm(tf.gradients(model_loss, trainable_variables), grad_clip)
#Use Adam Optimizer
train_operation = tf.train.AdamOptimizer(learning_rate) model_optimizer = train_operation.apply_gradients(zip(gradients,
trainable_variables)) return model_optimizer
```
##构建网络
&emsp;&emsp; 现在，我们可以把所有的碎片放在一起，为网络构建一个类。为了通过长短期记忆网络单元实际运行数据，我们将使用`tf.nn.dynamic_rnn`。此函数将适当地传递长短期记忆网络单元上的隐藏状态和单元格状态。它为每个批次中的每个序列返回每个长短期记忆网络单元的输出。它还给了我们最终的长短期记忆网络状态。我们想把这个状态保存为最终状态，这样我们就可以把它传递到下一个小批量运行的第一个长短期记忆网络单元。对于tf.nn.dynamic_rnn，我们从创建长短期记忆网络中获得神经元和初始状态，以及我们的输入序列。另外，我们需要在输入循环神经网络之前对输入进行热编码：
```
class CharL3TM:
def   init  (self, num_classes, batch_size=64, num_steps=5O,
lstm_size=l28, num_layers=2, learning_rate=O.OOl, grad_clip=5, sampling=False):
# When we're using this network for generating text by sampling, we'll be providing the network with
# one character at a time, so providing an option for it. if sampling == True:
batch_size, num_steps = l, l else:
batch_size, num_steps = batch_size, num_steps
tf.reset_default_graph()
# Build the model inputs placeholders of the input and target variables
self.inputs, self.targets, self.keep_prob = build_model_inputs(batch_size, num_steps)
# Building the L3TM cell
lstm_cell, self.initial_state = build_lstm_cell(lstm_size, num_layers, batch_size, self.keep_prob)
### Run the data through the L3TM layers
# one_hot encode the input
input_x_one_hot = tf.one_hot(self.inputs, num_classes)
# Runing each sequence step through the L3TM architecture and finally collecting the outputs
outputs, state = tf.nn.dynamic_rnn(lstm_cell, input_x_one_hot, initial_state=self.initial_state)
self.final_state = state
# Get softmax predictions and logits
self.prediction, self.logits = build_model_output(outputs, lstm_size, num_classes)
# Loss and optimizer (with gradient clipping)
self.loss = model_loss(self.logits, self.targets, lstm_size, num_classes)
self.optimizer = build_model_optimizer(self.loss, learning_rate, grad_clip)
```
## 超参数模型
&emsp;&emsp; 与任何深度学习体系结构一样，有一些超参数可用于控制模型并微调它。下面是我们正在为这个体系结构使用的超参数集合：<br>
&emsp;&emsp; 批大小是一次通过网络的序列数。步骤的数目是网络被训练的序列中的字符数。较大的通常更好，神经网络学习更加长远。但要花更长的时间来训练。100在这里通常是个好数字。<br>
&emsp;&emsp; 长短期学习网络的大小是隐藏层中的单位数。<br>
&emsp;&emsp; 架构编号层使用的是隐藏LSTM层的数量。学习率是学习的典型学习率。<br>
&emsp;&emsp; 最后，我们称保留概率的新信息被放弃层使用，它帮助神经网络避免过拟合。因此，如果神经网络过度拟合，尝试减少这个超参数。
## 训练此模型
&emsp;&emsp; 现在，让我们通过向所构建的模型提供输入和输出来开始进行训练，然后使用优化器来训练网络。不要忘记，我们需要使用之前的状态，同时对当前状态进行预测。因此，我们需要将输出状态传递回网络作为输入，以便在下一个输入的预测期间可以使用它。<br>
&emsp;&emsp; 让我们为我们的超参数提供初始值（您可以根据您用来训练此体系结构的数据集来调整它们）：
```
batch_size = lOO	# 3equences per batch
num_steps = lOO	# Number of sequence steps per batch lstm_size = 5l2	# 3ize of hidden layers in L3TMs num_layers = 2	# Number of L3TM layers learning_rate = O.OOl	# Learning rate
keep_probability = O.5	# Dropout keep probability epochs = 5
# 3ave a checkpoint N iterations save_every_n = lOO
L3TM_model = CharL3TM(len(language_vocab), batch_size=batch_size, num_steps=num_steps,
lstm_size=lstm_size, num_layers=num_layers, learning_rate=learning_rate)
saver = tf.train.3aver(max_to_keep=lOO) with tf.3ession() as sess:
sess.run(tf.global_variables_initializer())
# Use the line below to load a checkpoint and resume training
#saver.restore(sess, 'checkpoints/ 	.ckpt') counter = O
for e in range(epochs):
# Train network
new_state = sess.run(L3TM_model.initial_state) loss = O
for x, y in generate_character_batches(encoded_vocab, batch_size, num_steps):
counter += l
start = time.time()
feed = (L3TM_model.inputs: x, L3TM_model.targets: y, L3TM_model.keep_prob: keep_probability, L3TM_model.initial_state: new_state}
batch_loss, new_state, _ = sess.run([L3TM_model.loss,
L3TM_model.final_state, L3TM_model.optimizer], feed_dict=feed)
end = time.time()
print('Epoch number: (}/(}... '.format(e+l, epochs),
'3tep: (}... '.format(counter),
'loss: (:.4f}... '.format(batch_loss),
'(:.3f} sec/batch'.format((end–start))) if (counter % save_every_n == O):
saver.save(sess, "checkpoints/i(}_l(}.ckpt".format(counter,
lstm_size))
saver.save(sess, "checkpoints/i(}_l(}.ckpt".format(counter, lstm_size))
```
&emsp;&emsp; 在训练模型的最后，你应该会得到类似于下面的错误：
```
Epoch	number:	5/5...	3tep:	978...	loss:	l.7l5l...	O.O5O	sec/batch
Epoch	number:	5/5...	3tep:	979...	loss:	l.7428...	O.O5l	sec/batch
Epoch	number:	5/5...	3tep:	98O...	loss:	l.7l5l...	O.O5O	sec/batch
Epoch	number:	5/5...	3tep:	98l...	loss:	l.7236...	O.O5O	sec/batch
Epoch	number:	5/5...	3tep:	982...	loss:	l.73l4...	O.O5l	sec/batch
Epoch	number:	5/5...	3tep:	983...	loss:	l.7369...	O.O5l	sec/batch
Epoch	number:	5/5...	3tep:	984...	loss:	l.7O75...	O.O65	sec/batch
Epoch	number:	5/5...	3tep:	985...	loss:	l.73O4...	O.O5l	sec/batch
Epoch	number:	5/5...	3tep:	986...	loss:	l.7l28...	O.O49	sec/batch
Epoch	number:	5/5...	3tep:	987...	loss:	l.7lO7...	O.O5l	sec/batch
Epoch	number:	5/5...	3tep:	988...	loss:	l.735l...	O.O5l	sec/batch
Epoch	number:	5/5...	3tep:	989...	loss:	l.726O...	O.O49	sec/batch
Epoch	number:	5/5...	3tep:	99O...	loss:	l.7l44...	O.O5l	sec/batch
```
## 保存检查点
&emsp;&emsp; 让我们加载检查点。有关保存和加载检查点的更多信息，您可以查看TensorFlow文档：
```
tf.train.get_checkpoint_state('checkpoints') Output:
model_checkpoint_path: "checkpoints/i99O_l5l2.ckpt"
all_model_checkpoint_paths: "checkpoints/ilOO_l5l2.ckpt" all_model_checkpoint_paths: "checkpoints/i2OO_l5l2.ckpt" all_model_checkpoint_paths: "checkpoints/i3OO_l5l2.ckpt" all_model_checkpoint_paths: "checkpoints/i4OO_l5l2.ckpt" all_model_checkpoint_paths: "checkpoints/i5OO_l5l2.ckpt" all_model_checkpoint_paths: "checkpoints/i6OO_l5l2.ckpt" all_model_checkpoint_paths: "checkpoints/i7OO_l5l2.ckpt" all_model_checkpoint_paths: "checkpoints/i8OO_l5l2.ckpt" all_model_checkpoint_paths: "checkpoints/i9OO_l5l2.ckpt" all_model_checkpoint_paths: "checkpoints/i99O_l5l2.ckpt"
```
## 生成文本
&emsp;&emsp; 我们有一个基于我们的输入数据集的训练模型。下一步是使用这个经过训练的模型来生成文本，并查看这个模型是如何学习输入数据的样式和结构的。要做到这一点，我们可以从一些初始字符开始，然后在下一步中输入新的预测字符作为输入。我们将重复这个过程，直到我们得到一个具有特定长度的文本。<br>
&emsp;&emsp; 在下面的代码中，我们还向函数添加了额外的语句，以便用一些初始文本启动神经网络，并从那里开始。<br>
&emsp;&emsp; 神经网络给我们每个单词的预测或概率。为了减少干扰并且只使用神经网络更有把握的字符，我们将只从输出中最可能的N个字符中选择一个新字符：
```
def choose_top_n_characters(preds, vocab_size, top_n_chars=4): p = np.squeeze(preds)
p[np.argsort(p)[:–top_n_chars]] = O p = p / np.sum(p)
c = np.random.choice(vocab_size, l, p=p)[O] 
return c
def sample_from_L3TM_output(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
samples = [char for char in prime]
L3TM_model = CharL3TM(len(language_vocab), lstm_size=lstm_size, sampling=True)
saver = tf.train.3aver() with tf.3ession() as sess:
saver.restore(sess, checkpoint)
new_state = sess.run(L3TM_model.initial_state) for char in prime:
x = np.zeros((l, l))
x[O,O] = vocab_to_integer[char] feed = (L3TM_model.inputs: x,
L3TM_model.keep_prob: l., L3TM_model.initial_state: new_state}
preds, new_state = sess.run([L3TM_model.prediction, L3TM_model.final_state],feed_dict=feed)
c = choose_top_n_characters(preds, len(language_vocab)) samples.append(integer_to_vocab[c])
for i in range(n_samples): x[O,O] = c 
feed = (L3TM_model.inputs: x, L3TM_model.keep_prob: l., L3TM_model.initial_state: new_state}
preds, new_state = sess.run([L3TM_model.prediction, L3TM_model.final_state], feed_dict=feed)
c = choose_top_n_characters(preds, len(language_vocab)) samples.append(integer_to_vocab[c])
return ''.join(samples)
```
&emsp;&emsp; 让我们启动最新的检查点来检查采样过程：
```
tf.train.latest_checkpoint('checkpoints') Output:
'checkpoints/i99O_l5l2.ckpt'
```
&emsp;&emsp; 现在，是时候使用这个最新的检查点：
```
checkpoint = tf.train.latest_checkpoint('checkpoints')
sampled_text = sample_from_L3TM_output(checkpoint, lOOO, lstm_size, len(language_vocab), prime="Far")
print(sampled_text)
Output:
```
<br>
```
INFO:tensorflow:Restoring parameters from checkpoints/i99O_l5l2.ckpt
Farcial the
confiring to the mone of the correm and thinds. 3he she saw the
streads of herself hand only astended of the carres to her his some of the princess of which he came him of
all that his white the dreasing of
thisking the princess and with she was she had
bettee a still and he was happined, with the pood on the mush to the peaters and seet it.

"The possess a streatich, the may were notine at his mate a misted and the
man of the mother at the same of the seem her felt. He had not here.

"I conest only be alw you thinking that the partion of their said."

"A much then you make all her somether. Hower their centing
 

about
this, and I won't give it in himself.
I had not come at any see it will that there she chile no one that him.

"The distiction with you all.... It was
a mone of the mind were starding to the simple to a mone. It to be to ser in the place," said Vronsky.
"And a plais in
his face, has alled in the consess on at they to gan in the sint at as that
he would not be and t
```
&emsp;&emsp; 你可以看到我们能够产生一些有意义的单词和一些毫无意义的单词。为了获得更多的结果，您以运行更多的模型，并尝试使用超参数。
## 总结
&emsp;&emsp; 我们了解了递归神经网络，它是如何工作的，以及为什么它们会成为一个大的处理算法。我们在一个字符循环神经网络语言模型上训练了有趣的数据集，并看到递归神经网络的去向。你可以有信心地期待在递归神经网络空间中有大量的创新，并且我相信它们将成为一个智能系统通用和关键的部分。


学号|姓名|专业
-|-|-
201802110485|李忠|计算机应用技术
