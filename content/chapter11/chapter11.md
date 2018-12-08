# &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;表示学习----实现词嵌入<br>
&emsp;&emsp;机器学习是一门以统计学和线性代数为基础的科学。由于反向传播的存在，在大多数机器学习或深度学习体系结构中，应用矩阵运算是非常常见的。这就是为什么深度学习，或者一般的机器学习，只接受实值量作为输入。但是这一事实与许多应用相矛盾，例如机器翻译、情感分析等等;因为它们的输入是文本。因此，为了在这些应用程序中能使用到深度学习，我们需要以深度学习能接受的形式来使用它。<br>
&emsp;&emsp;在本章中，我们将介绍表示学习的领域，这是一种从文本中获得实值表示的方法，与此同时也保留了实际文本的语义。例如，爱（love）的表达应该和爱慕（adore）的表达非常接近，因为他们用在非常相似的语境中。因此，本章将讨论以下主题:<br>
&emsp;&emsp;•	表示学习的介绍<br>
&emsp;&emsp;•	Word2Vec模型<br>
&emsp;&emsp;•	一个关于skip-gram体系结构的实例<br>
&emsp;&emsp;•	Word2Vec模型下skip-gram体系结构的实现<br>
## 表示学习的介绍<br>
&emsp;&emsp;到目前为止，我们使用的所有机器学习算法或体系结构都要求输入为实值，又或者是实值量的矩阵，这是机器学习中的一个常见主题。例如，在卷积神经网络中，我们必须将图像的原始像素值作为模型输入。在这一部分中，我们处理的是文本，因此我们需要以某种方式对文本进行编码并产生实值量，然后将其输入到机器学习的算法中。为了将输入文本编码为实值量，我们需要使用一种叫做自然语言处理(NLP)的中间科学。<br>
&emsp;&emsp;我们在这种管道模型中提到过，我们将文本输入到机器学习模型中，比如情感分析这种模型，这将会产生问题而无法工作，因为我们无法应用反向传播或其他任何操作,比如输入的点它只是一个字符串而已。因此，我们需要使用一种自然语言处理的机制，它将使我们能够构建一个文本的中间表示，它可以携带与文本相同的信息，并被输入到机器学习模型中。<br>
&emsp;&emsp;我们需要将输入文本中的每个单词或标记转换为实值向量。如果这些向量不包含原始输入的模式、信息、含义和语义，那么它们将是无用的。例如，在真实文本中，爱（love）和爱慕（adore）这两个词非常相似，意思几乎相同。我们需要这两个词的实值向量的结果彼此接近并处在同一个向量空间中。因此，这两个词的向量表示和另一个与它们不相似的词的表示会如下图：<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image001.png)<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图15.1„单词的向量表示<br>
&emsp;&emsp;有许多技术可以用于这项工作。这种技术被称为嵌入，即将文本嵌入到另一个实值向量空间中。<br>
&emsp;&emsp;之后我们会看到，这个向量空间实际上非常有趣，因为你会发现你可以把一个单词的向量从其他与它相似的单词中提取出来，或者在这个空间中做一些布局上的处理。<br>
## Word2Vec<br>
&emsp;&emsp;Word2Vec是自然语言处理领域应用最广泛的嵌入式技术之一。该模型通过观察输入词出现的上下文信息，从输入文本中创建实值向量。相似的词会在非常相似的语境中被提及，模型因此会知道这两个词应该放在相近的特定嵌入空间中。<br>
&emsp;&emsp;从下图的描述中，模型将习得love和adore这两个词有着非常相似的上下文，它们应该放在非常接近的向量空间中。而like这个词的语境可能和love这个词有点相似，但不会像单词adore那样接近love:<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image002.png)<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图15.2„表感情程度句子的样本<br>
&emsp;&emsp;Word2Vec模型也依赖于输入句子的语义特征;例如，单词adore和love主要用于积极的语境，通常放在名词短语或名词之前。因此，模型会知道这两个词有一些共同之处，它更有可能把这两个向量的向量表示放在相似的上下文中。因此，句子的结构会提供Word2Vec模型很多关于类似单词的信息。<br>
&emsp;&emsp;在实践中，人们向Word2Vec模型输入大量的文本。该模型将学习如何为相似的单词生成相似的向量，并且它将为输入文本中的每个唯一的单词执行此操作。所有这些单词的向量将被组合在一起，最终的输出将是一个嵌入矩阵，其中每一行表示一个特定单词的实值向量表示。因此，模型的最终输出将是训练语料库中所有唯一单词的嵌入矩阵。通常，好的嵌入矩阵可以包含数百万个实值向量。<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image003.png)<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图15.3„Word2Vec管道模型的例子<br>
&emsp;&emsp;Word2Vec建模使用窗口查看句子，然后根据上下文信息预测窗口中间词的向量;Word2Vec模型每次只扫描一个句子。与任何机器学习技术类似，我们需要为Word2Vec模型定义一个代价函数及其相应的优化准则，使模型能够为每个唯一的对象生成实值向量，并根据其上下文信息将向量相互关联。<br>
## 构建Word2Vec模型<br>
&emsp;&emsp;在本节中，我们将详细介绍如何构建Word2Vec模型。正如我们前面提到的，我们的最终目标是拥有一个经过训练的模型，该模型能够为输入文本数据生成实值向量表示，这也称为单词嵌入。<br>
在模型的训练过程中，我们将使用极大似然法(https:// en.wikipedia.org/wiki/Maximum_likelihood)，该方法可以在给予模型看到的前一个单词的情况下最大化输入句子中下一个单词wt的概率，我们称之为h。这种最大似然方法将用归一化指数函数表示为:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp; &emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image004.png)<br>
&emsp;&emsp;在这里，score函数计算一个值来表示目标词wt相对于语境h的兼容性。该模型将在训练时对输入序列进行训练，以最大化训练输入数据的可能性（对数似然法用于数学上的简化和使用对数的推导)：<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp; &emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image005.png)<br>
&emsp;&emsp;因此，ML方法将试图最大化上述方程，这最终会形成一个概率语言模型。但是这个计算代价是非常大的，我们需要使用在这个模型的相应当前语境h中用score函数来计算每一个在词汇表V单词w'中的单词的概率，这个过程在每一个训练步骤发生。<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image006.png)<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图15.4„概率语言模型的一般结构<br>
 &emsp;&emsp;由于构建概率语言模型的计算成本很高，人们倾向于使用不同的计算成本较低的技术，比如连续词袋模型 (CBOW) 和skip-gram模型。<br>
&emsp;&emsp;通过训练这些模型，建立了一种逻辑回归的二分类法，将相同语境下的真实目标词wt和h干扰词或虚词分离开来。下面的图表使用CBOW技术简化了这个想法:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image007.png)<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图15.5„skip-gram模型的总体架构<br>
 &emsp;&emsp;下一个图显示了构建Word2Vec模型时可以使用的两种架构:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image008.png)<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图15.6„不同的Word2Vec模型架构<br>
 &emsp;&emsp;更正式地说，这些技术的目标函数最大化了如下式子:<br>
&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image009.png)<br>
&emsp;&emsp; 其中:<br>
  ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image010.jpg)是基于模型在数据集D中理解语境h中的单词w的二元逻辑回归的概率，它是用0向量来计算的。这个向量表示已学习的嵌入。
&emsp;&emsp;  ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image011.png)是我们可以从一个有干扰的概率分布(如训练输入示例的单位图)中生成的虚词或干扰词。<br>
&emsp;&emsp; 综上所述，这些模型的目标是区分真实有用和无用的输入，因此，对于虚词和干扰词，模型为实词赋较高的概率值，为干扰词或虚词赋较少的概率值。
当模型将高概率值赋给实词，低概率值赋给干扰词时，此时目标函数最大化。<br>
 ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image012.png)
 从技术层面上讲，给实词赋高概率值的过程称为负采样(https://papers.nips)。(https://papers.nips.cc/paper/5O2l–distributed–representations–of–words–and–phrases–and–their–compositionality.pdf)，<br>使用这种损失函数有很好的数学动力:它提出的校正在极限下近似softmax函数的校正。但在计算上，它是很好的方法，因为计算损失函数现在只与我们选择的干扰词的数量(k)成比例，而不是词汇表中的所有单词(V)成比例。实际上，我们将使用非常类似的噪声对比估计(NCE) (https://papers.nips.cc/paper/5l65–learning–word–embeddings–efficiently–with–noise–contrastive–estimation.pdf) <br> 损失，对于这种损失，TensorFlow有一个方便的辅助函数，tf.nn.nce_loss().
## skip-gram体系结构的一个实例
 &emsp;&emsp;让我们通过一个实际的例子，看看skip-gram模型将如何在这种情况下工作:<br>
 &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;*the quick brown fox jumped over the lazy dog*<br>
 &emsp;&emsp;首先，我们需要建立一个词及其上下文的数据集。上下文的定义取决于我们，但上下文必须有意义。因此，我们将在目标单词周围设置一个窗口，从右边取一个单词，从左边取一个单词。
通过使用这种对于上下文操作的技术，我们最终会得到以下一组词及其对应的语境:<br>
 &emsp;&emsp; &emsp; &emsp;&emsp; &emsp;&emsp;&emsp;&emsp;([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...<br>
&emsp;&emsp;生成的单词及其对应的上下文将表示为形如(上下文、目标单词)这样的一对。skip- gram模型的思想与CBOW模型相反。在skip- gram模型中，我们将尝试基于目标词来预测单词的上下文。例如，考虑到第一对的时候，skip-gram模型将尝试从目标词敏捷的来预测那只和棕色等等。因此，我们可以重写我们的数据集如下:<br>
 &emsp;&emsp; &emsp; &emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;(quick, the), (quick, brown), (brown, quick), (brown, fox), ...<br>
 &emsp;&emsp; 现在，我们有了一组输入和输出。<br>
 &emsp;&emsp; 让我们尝试模拟特定步骤t的训练过程，因此，skip-gram模型将取第一个训练样本，其中输入为单词快速的，目标输出为单词那只。接下来，我们还需要构造干扰输入，因此我们将从输入数据的一元模型中随机选择。为了简单起见，干扰向量将只有一个。例如，我们可以选择睡觉这个词作为一个干扰的例子。<br>
 &emsp;&emsp; 现在，我们可以继续计算真正的一对词组和干扰词组之间的损失为:<br>
&emsp;&emsp;&emsp;&emsp; &emsp; &emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image013.png)<br>
&emsp;&emsp;在本例中，目标是更新0参数来改进之前的目标函数。通常，我们可以使用梯度。因此，我们将尝试计算目标函数参数0的梯度损失，它将被表示为 ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image014.jpg)。<br>
&emsp;&emsp;经过训练后，我们可以根据实值向量表示的降维结果对其进行显示。你会发现这个向量空间很有趣因为你可以用它做很多有趣的事情。例如，你可以在这个空间里通过说国王之于皇后就像男人之于女人。我们甚至可以通过从皇后向量中减去国王向量然后加上男人来得到女人向量;这样做得到的结果将非常接近女人的实际学习向量。你也可以在这个空间里学习如何布局。<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image015.png)<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图15.7„用t分布随机相邻嵌入（t-SNE）维度减少技术对两个维度的学习向量的规划<br>
 &emsp;&emsp;前面的示例提供了这些向量背后非常好的直观信息，以及它们如何对大多数NLP应用程序(如机器翻译或词性标记)有用。<br>
 ## Skip-gram 的Word2Vec实现
 &emsp;&emsp;在理解了skip-gram模型如何工作的数学方式之后，我们可以来实现skip-gram，它将单词编码为具有特定属性的实值向量(因此得名Word2Vec)。通过实现这个体系结构，您将了解另一种表示的学习过程是如何工作的。<br>
 &emsp;&emsp;文本是许多自然语言处理应用程序(如机器翻译、情感分析和文本转语音系统)的主要输入。因此，学习文本的实值表示将帮助我们对这些项目使用不同的深度学习技术。在这本书的前几章中，我们介绍了一种叫做独热编码的东西，它产生除该向量表示的字索引之外的0向量。所以，你可能好奇为什么我们不在这里使用它。因为这种方法效率很低，通常有一大堆不同的单词，可能大约有50,000个单词，使用独热编码就会产生一个49,999个为0的向量集，而只有一个向量集为1。
这样一个非常多余的输入将会导致巨大的计算浪费，因为我们会在神经网络的隐藏层中做矩阵乘法。<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image016.png)<br>
 &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp; 图15.8„在独热编码计算将导致巨大的浪费<br>
  &emsp;&emsp;正如我们前面提到的，使用独热编码的结果将得到非常多的稀疏向量，特别是当你想要编码大量不同的单词时。<br>
   &emsp;&emsp;如下图所示，当我们将除一项外的所有0的稀疏向量乘以一个权矩阵时，输出将仅为矩阵的一行，该行对应于稀疏向量的一个值:<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image017.png)<br>
 &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp; 图15.9„一个几乎全是0的独热向量与隐藏层权矩阵相乘的效果<br>
 &emsp;&emsp;为了避免这种巨大的计算浪费，我们将使用嵌入的方法，这是一个全连接层，其中带有一些嵌入权重。在这一层，我们跳过这个低效的乘法，从权值矩阵查找嵌入层的嵌入权值。
<br>
 &emsp;&emsp;因此，为了避免计算产生的浪费，取而代之的是我们将使用权重来查找这个权矩阵以便找到嵌入的权重。首先，需要构建此查找。为此，我们将把所有输入单词编码为整数，如下图所示，然后为了得到这个单词的相应值，我们将使用它的整数表示作为这个权矩阵中的行数。查找特定单词的相应嵌入值的过程称为嵌入查找。如前所述，嵌入层将只是一个全连接层，其中单位数表示嵌入维数。<br>
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; ![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter11/chapter11_image/image018.png)<br>
 &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;&emsp; &emsp;  &emsp;&emsp; &emsp;&emsp;&emsp;&emsp; 图15.10„标记化的查找表<br>
 &emsp;&emsp;你可以看到这个过程是非常直观和直接的;我们只需要遵循以下步骤:<br>

&emsp;&emsp;1.定义查找表，将其视为权矩阵<br>
&emsp;&emsp;2.将嵌入层定义为具有特定单元数的全连接的隐藏层(嵌入维数)<br>
&emsp;&emsp;3.使用权矩阵查找作为计算不必要的矩阵乘法的替代方法<br>
&emsp;&emsp;4最后，将查找表训练成任何权矩阵
<br>
&emsp;&emsp;如前所述，我们将在本节中构建一个skip-gram Word2Vec模型，这是一种学习词汇表示的有效方法，同时也保留了词汇所具有的语义信息。
因此，让我们继续使用skip-gram体系结构构建一个Word2Vec模型，该体系结构被证明优于其他体系结构。<br>
## 数据分析和预处理
&emsp;&emsp;在本节中，我们将定义一些辅助函数，使我们能够构建一个良好的Word2Vec模型。为了实现这个模型，我们将使用一个绿色版本的维基百科
 (http://mattmahoney.net/dc/textdata.html)。<br>
&emsp;&emsp;因此，让我们从导入所需的包开始:<br>
```#importing the required packages for this implementation import numpy as np
import tensorflow as tf
#Packages for downloading the dataset
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
#packages for data preprocessing 
import re
from collections import Counter 
import random
```
&emsp;&emsp;接下来，我们将定义一个类，如果之前没有下载数据集，它就用于下载数据集:
```# In this implementation we will use a cleaned up version of Wikipedia from Matt Mahoney.
# 3o we will define a helper class that will helps to download the dataset wiki_dataset_folder_path = 'wikipedia_data'
wiki_dataset_filename = 'text8.zip' wiki_dataset_name = 'Text8 Dataset'

class DLProgress(tqdm): last_block = O

def hook(self, block_num=l, block_size=l, total_size=None): self.total = total_size
self.update((block_num – self.last_block) * block_size) self.last_block = block_num
# Cheking if the file is not already downloaded if not isfile(wiki_dataset_filename):
with DLProgress(unit='B', unit_scale=True, miniters=l, desc=wiki_dataset_name) as pbar:
urlretrieve(
'http://mattmahoney.net/dc/text8.zip', wiki_dataset_filename,
pbar.hook)

# Checking if the data is already extracted if not extract it if not isdir(wiki_dataset_folder_path):
with zipfile.ZipFile(wiki_dataset_filename) as zip_ref: zip_ref.extractall(wiki_dataset_folder_path)
with open('wikipedia_data/text8') as f: cleaned_wikipedia_text = f.read()

Output:

Text8 Dataset: 3l.4MB [OO:39, 794kB/s]
```
&emsp;&emsp;我们可以看看这个数据集的前100个字符:
```cleaned_wikipedia_text[O:lOO]

' anarchism originated as a term of abuse first used against early working class radicals including t'
```
&emsp;&emsp;接下来，我们将对文本进行预处理，因此我们将定义一个辅助函数，它将帮助我们将特殊字符(如标点符号)替换为已知标记。另外，为了减少输入文本中的干扰，您可能需要删除文本中不经常出现的单词:
```def preprocess_text(input_text):

# Replace punctuation with some special tokens so we can use them in our model
input_text = input_text.lower()
input_text = input_text.replace('.', ' <PERIOD> ') input_text = input_text.replace(',', ' <COMMA> ') input_text = input_text.replace('"', ' <QUOTATION_MARK> ') input_text = input_text.replace(';', ' <3EMICOLON> ')
input_text = input_text.replace('!', ' <EXCLAMATION_MARK> ') input_text = input_text.replace('?', ' <QUE3TION_MARK> ') input_text = input_text.replace('(', ' <LEFT_PAREN> ') input_text = input_text.replace(')', ' <RIGHT_PAREN> ') input_text = input_text.replace('––', ' <HYPHEN3> ') input_text = input_text.replace('?', ' <QUE3TION_MARK> ') input_text = input_text.replace(':', ' <COLON> ')
text_words = input_text.split()
# neglecting all the words that have five occurrences of fewer text_word_counts = Counter(text_words)
trimmed_words = [word for word in text_words if text_word_counts[word]
> 5]

return trimmed_words
```
&emsp;&emsp;现在，让我们在输入文本时调用这个函数并查看输出:
```preprocessed_words = preprocess_text(cleaned_wikipedia_text) print(preprocessed_words[:3O])

Output:
['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first',
'used', 'against', 'early', 'working', 'class', 'radicals', 'including',
'the', 'diggers', 'of', 'the', 'english', 'revolution', 'and', 'the',
'sans', 'culottes', 'of', 'the', 'french', 'revolution', 'whilst']
```
&emsp;&emsp;让我们看看对于文本预处理版本，我们有多少单词和不同的单词:
```print("Total number of words in the text: (}".format(len(preprocessed_words))) print("Total number of unique words in the text: (}".format(len(set(preprocessed_words))))
 

Output:

Total number of words in the text: l668O599 Total number of unique words in the text: 6364l
```

