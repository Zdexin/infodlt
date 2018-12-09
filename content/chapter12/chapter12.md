# &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;神经情感分析
&emsp;&emsp;在这一章中，我们将讨论自然语言处理中最热门和最流行的应用之一，即情感分析。现在大多数人都是通过社交媒体平台来表达自己对某件事情的看法，而利用这海量的文字来记录客户对某件事情的满意度对于公司甚至政府来说都是非常重要的。<br>
&emsp;&emsp;在本章中，我们将使用递归式神经网络来建立情感分析的解决方案。本章将讨论以下主题:<br>
&emsp;&emsp;通用的情感分析架构<br>
&emsp;&emsp;情感分析-模型实现<br>
## 通用的情感分析架构
&emsp;&emsp;在本节中，我们将重点讨论可用于情感分析的通用深度学习体系结构。下图显示了构建情感分析模型所需的处理步骤。<br>
&emsp;&emsp;首先，我们要讲的是自然语言:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter12/chapter12_image/image022.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图1„情绪分析解决方案的通用管道模型甚至说是自然语言的解决方案<br>
&emsp;&emsp;我们将使用电影评论作为素材来构建这个情感分析应用程序。这个应用程序的目标是基于输入的原始文本来产生这个观点是否积极的判断。例如，如果原始文本类似于“这部电影很好”，那么我们需要模型为它产生积极的情绪。<br>
&emsp;&emsp;情感分析应用程序将带领我们完成许多处理步骤，这些步骤是在神经网络中使用自然人类语言所必需的，比如嵌入。<br>
&emsp;&emsp;在这种情况下，我们有一个原始文本，例如，这不是一个好电影!那么我们最终想要得到的是这到底是一种消极的情绪还是积极的情绪。<br>
&emsp;&emsp;这类应用程序有几个难点:<br>
&emsp;&emsp;其中之一就是序列可能有不同的长度。这是一个很短的例子，但是我们会看到另外一个超过500字文本的例子。<br>
&emsp;&emsp;另一个问题是，如果我们只看单个单词(比如good)，那就表明了积极的情绪。然而，它的前面是not，所以现在它表示的是一个负面情绪。这可能会使情况变得更加复杂，之后我们将看到一个示例。<br>
&emsp;&emsp;正如我们在前一章中学到的，神经网络不能处理原始文本，因此我们需要首先将其转换成所谓的标记。这些基本上都是整数值，所以我们遍历整个数据集，来计算每个单词被使用的次数。然后，我们制作一个词汇表，每个单词在这个词汇表中得到一个索引。比如这个单词有整数ID或标记的是11，这个单词标记的是6，而不是标记的21，等等。至此，我们已经将原始文本转换为一个名为token的整数列表。但神经网络仍然不能对这些数据进行操作，因为如果我们有10,000个词汇表，这些标记可以取0到9999之间的值，但它们之间可能根本不相关。所以很显然，数字998可能与数字999有完全不同的语义含义。<br>
&emsp;&emsp;因此，我们将使用我们在上一章学到的表示学习或嵌入的思想。该嵌入层将整数标记转换为实值向量，因此标记11成为向量[0.67,0.36，…，0.39]，如图1所示。对于下一个标记6也是一样。<br>
&emsp;&emsp;简单回顾一下我们在上一章所学习的内容:前面图中的嵌入层习得了标记和它们对应的实值向量之间的映射。同时，嵌入层习得单词的语义，使具有相似含义的单词在这个嵌入的空间中比较接近。<br>
&emsp;&emsp;从输入的原始文本中，我们得到一个二维矩阵，或者张量，现在可以输入到递归神经网络(递归神经网络)。这样就可以处理任意长度的序列，然后将该网络的输出值输入到一个具有sigmoid激活函数的全连接层或致密层。因此，输出值介于0和1之间，其中0的值表示负面情绪。但如果sigmoid函数的值既不是0也不是1呢?然后我们需要在中间引入一个临界值或阈值，这样如果这个值低于0.5，那么相应的输入就会被认为是负面情绪，超过这个临界值的值就会被认为是正面情绪。<br>
## 递归神经网络--情感分析语境
&emsp;&emsp;现在，让我们回顾一下递归神经网络的基本概念并在情感分析应用的背景下讨论它们。正如我们在递归神经网络章节中提到的，递归神经网络的基本构件是一个循环单元，如图所示:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter12/chapter12_image/image023.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图2„递归神经网络的一个抽象概念<br>
&emsp;&emsp;这个图是对循环单元内发生的事情的抽象表示。这里输入一个单词，比如“好”。当然，它必须转换成嵌入向量。然而，我们现在将忽略这一点。另外，这个单元有一种内存状态，根据这个内容的状态和输入，我们将更新这个状态并将新数据写入状态例如，假设我们以前在输入中见过“不”这个词;我们将其写入状态是为了当我们在下面的输入中看到“好”这个词的时候，能知道我们刚刚从状态那里看到了“不”这个词。现在，我们看到了“好”这个词。因此, 当我们看到“不好”这两个词的组合时，我们就必须写入状态,即这可能表明,整个输入的文本可能具有负面情绪。从旧状态到新状态的内容的映射是通过所谓的门来完成的，这些实现的方式在不同版本的循环单元中是不同的。它基本上是一个带激活函数的矩阵运算，但是我们马上就会看到，其中含有一个反向传播梯度的问题。因此，递归神经网络必须以一种特殊的方式来设计，这样梯度才不会扭曲太多。<br>
&emsp;&emsp;在一个循环单元中，我们有一个类似的生成输出的门，再一次，循环单元的输出取决于状态的当前内容和我们看到的输入。所以我们可以尝试做的是展开一个循环单元的处理过程:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter12/chapter12_image/image024.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图3„展开版的递归神经网络<br>
&emsp;&emsp;现在，我们得到的只是一个循环单元，但是流程图显示了在不同的时间阶段发生了什么。所以:<br>
&emsp;&emsp;在时间步骤1中，我们将单词this输入到循环单元，它的内部内存状态初始化为0。这是由TensorFlow在我们开始处理新数据序列时完成的。我们看到了this这个词循环单位状态是0。因此，我们使用内部门来更新内存状态，然后this在第2步中输入is这个词的时候被使用，现在，内存状态有一些内容。this这个词没有太多意义，所以状态可能还是0左右。<br>
&emsp;&emsp;is中也没有太多意义，所以状态可能还是0。<br>
&emsp;&emsp;在接下来的步骤中，我们看到的单词not，这意味着我们最终想要预测的，这是整个输入文本的情感。这是我们需要存储在内存中的东西，这样循环单元内的门会发现状态可能已经包含一些接近零的值。但现在它想存储我们刚刚看到的单词not，所以它在这种状态下保存了一些非零值。<br>
&emsp;&emsp;然后，我们继续下一个时间步骤，我们有单词a;这也没有太多的信息，所以它可能被忽略了。它只是在状态内复制。<br>
&emsp;&emsp;现在，我们有了“very”这个词，这表明，无论存在何种情绪，都可能是一种强烈的情绪，因此，循环单元现在知道，我们已经知道了单词not和very。它以某种方式将其存储在内存状态中。<br>
&emsp;&emsp;在接下来的时间里，我们看到了单词good，所以现在递归神经网络知道了not very good三个词，它想，哦，这可能是一个负面情绪!因此，它将该值存储在内部状态中。然后，在最后的步骤中，我们看到了movie这个单词，这和情绪没有实际联系，所以它可能会被忽略了。<br>
&emsp;&emsp;接下来，我们使用循环单元内的另一个门输出内存状态的内容，然后用sigmoid函数处理它(我们在这里不展示)。输出值在0到1之间。<br>
&emsp;&emsp;我们的想法是，我们想要从互联网电影数据库中成千上万的电影评论例子中训练这个神经网络，在那里，对于每个输入文本，我们给它的真正情感价值是积极或消极的。然后，我们希望TensorFlow找出循环单元内部的门存储的应该是什么，这样他们就能准确地从输入文本反映出正确的情感:<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter12/chapter12_image/image025.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图4„用于实现本章的架构<br>
&emsp;&emsp;我们将在此实现中使用的递归神经网络的体系结构是具有三个层的递归神经网络类型体系结构。在第一层，除了现在我们需要从每个时间步骤的循环单元中输出值之外，发生了我们刚刚解释的情况。然后，我们收集一个新的数据序列，它是第一个循环层的输出。接下来,我们可以将其输入第二层，因为递归环节需要输入数据序列(并且我们从第一层得到的输出和我们想要输入到第二循环层的输出是一些浮点值，它们的含义我们并不能真正理解)。但这在递归神经网络中是有意义的，可这不是我们人类能够理解的。然后，我们在第二个循环层中做类似的处理。<br>
&emsp;&emsp;首先，我们将这个循环单元的内部内存状态初始化为0;然后，我们从第一个循环层获取第一个输出并输入它。我们用这个循环单元中的门来处理它并更新状态，取第一层循环单元的输出作为第二个单词，并使用它作为输入和内部内存状态。我们继续这样做，直到我们处理完整个序列，然后我们收集第二循环层的所有输出。我们在第三个循环层中使用它们作为输入，在那里我们做类似的处理。但这里，我们只需要最后一步的输出，这是对到目前为止已经输入的所有内容的总结。然后我们将它输出到一个全连接层，我们在这里没有表现出来。最后是sigmoid激活函数，得到0到1之间的值，分别表示消极情绪和积极情绪。<br>
## 爆炸和消失的梯度-重述
&emsp;&emsp;正如我们在前一章提到的，有一种现象叫做梯度爆炸和梯度消失，这在递归神经网络中非常重要。让我们回到图1;该流程图解释了这种现象是什么。<br>
&emsp;&emsp;假设我们在这个数据集中有一个500字的文本，我们将使用它来实现我们的情感分析分类器。在每次步骤中，我们以递归的方式在循环单元中应用内部门;所以如果有500字，我们会用500次门来更新循环单元的内部记忆状态。<br>
&emsp;&emsp;我们知道，训练神经网络的方法是使用所谓的梯度反向传播，所以我们要用一些损失函数得到神经网络的输出，然后得到我们想要的输入文本的真实输出。然后，我们要最小化这个损失值，这样神经网络的实际输出就会是这个特定输入文本的期望输出。所以，我们需要求这个损失函数的梯度关于这些循环单位内的权值，这些权值是用于更新内部状态并输出最终值的门。<br>
&emsp;&emsp;现在，这个门被应用了500次，如果其中含有一个乘法，本质上我们得到的是一个指数函数。所以，如果你把一个值和它本身相乘500次，如果这个值略小于1，那么它会很快消失或者不存在。类似地，如果一个稍微大于1的值与自身相乘500次，它就会爆炸。<br>
&emsp;&emsp;能在500次乘法中幸存的值只有0和1。它们会保持不变，所以循环单位实际上比你在这里看到的要复杂得多。这是一个抽象的概念——我们想要映射内部内存状态和输入来更新内部内存状态并输出一些值，但是在现实中，我们需要非常小心地通过这些门反向传播梯度，这样我们就不会在很多很多步骤上得到指数乘法。我们也鼓励你看一些关于循环单元的数学定义的教程。<br>
## 情感分析-模型实现
&emsp;&emsp;我们已经看到了如何实现递归神经网络的LSTM变体的堆叠版本。更令人兴奋事情来了，我们将使用一个更高级别的API，称为Keras。<br>
## Keras
&emsp;&emsp;vKeras是一种高级神经网络API，用Python编写，能够在TensorFlow、CNTK或Theano上运行。它的开发重点是支持快速实验。能够以最小的时间间隔从一个想法落成到一个结果，是做好研究的关键。——Keras网站<br>
&emsp;&emsp;因此，Keras只是TensorFlow和其他深度学习框架的封装。它对于原型设计和快速构建非常有用，但另一方面，它使您对代码的控制减少。我们将有机会在Keras中实现这种情感分析模型，这样您就可以在TensorFlow和Keras中得到实际的实现。您可以使用Keras进行快速原型设计，并为生产准备系统使用TensorFlow。<br>
&emsp;&emsp;对你来说，更有趣的消息是你不必转换到一个完全不同的环境。现在，您可以在TensorFlow中访问Keras作为模块，并导入包，如下所示:<br>
```
from tensorflow.python.keras.models import 3equential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
```
&emsp;&emsp;那么，让我们继续使用我们现在可以在TensorFlow中称为更抽象的模块，它将帮助我们快速原型化深度学习解决方案。这是因为我们可以用几行代码编写完整的深度学习解决方案。<br>
## 数据分析和预处理
&emsp;&emsp;现在，让我们转向实际的实现，在那里我们需要加载数据。Keras实际上有一个功能，可以用于从IMDb加载这个情感数据集，但问题是它已经将所有单词映射到整数标记。这是使用自然人类语言洞察力神经网络的一个重要部分，我想向你们展示如何做到这一点。
另外，如果您想要使用这段代码来分析您在其他语言中可能拥有的任何数据，那么您需要自己完成这项工作，因此我们已经快速实现了一些用于下载此数据集的函数。
让我们从导入一系列必需的包开始:<br>
```
%matplotlib inline
import matplotlib.pyplot as plt import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from tensorflow.python.keras.models import 3equential
from tensorflow.python.keras.layers import Dense, GRU, Embedding from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
```
&emsp;&emsp;然后我们加载数据集:<br>
```
import imdb imdb.maybe_download_and_extract()

Output:
–	Download progress: l00.0%
Download finished. Extracting files. Done.

input_text_train, target_train = imdb.load_data(train=True) input_text_test, target_test = imdb.load_data(train=False)

print("3ize of the trainig set: ", len(input_text_train)) print("3ize of the testing set:	", len(input_text_test))

Output:
3ize of the trainig set: 25000
3ize of the testing set: 25000
```
&emsp;&emsp;正如你所看到的，它在训练集和测试集中有25000条文本。<br>
&emsp;&emsp;让我们看一个来自训练集的例子，看看它是什么样子的: <br>
```
#combine dataset
text_data = input_text_train + input_text_test input_text_train[l]

Output:
'This is a really heart–warming family movie. It has absolutely brilliant animal training and "acting" (if you can call it like that) as well (just think about the dog in "How the Grinch stole Christmas"... it was plain bad training). The Paulie story is extremely well done, well reproduced and in general the characters are really elaborated too. Not more to say except that this is a GREAT MOVIE!<br /><br />My ratings: story 8.5/10, acting 7.5/10, animals+fx 8.5/10, cinematography 8/10.<br /><br />My overall rating: 8/10 – BIG FAMILY MOVIE AND VERY WORTH WATCHING!'
target_train[1] Output:
1.0
```
&emsp;&emsp;这是一个相当短的文本，情感值是1.0，说明这是一个积极的情感，所以这是对这部电影的积极评价。<br>
&emsp;&emsp;现在，我们来看索引表，这是处理原始数据的第一步，因为神经网络不能处理文本数据。Keras已经实现了所谓的索引表，用于构建词汇表并将单词转换为整数。并且，我们说我们想要最多10000个单词，所以它只使用数据集中使用最广泛的10000个单词:<br>
```
num_top_words = l0000 
tokenizer_obj = Tokenizer(num_words=num_top_words)
```
&emsp;&emsp;现在，我们从数据集中提取所有的文本，我们调用这个函数以便让他适应文本:<br>
`tokenizer_obj.fit_on_texts(text_data)`<br>
&emsp;&emsp;标记索引表大约需要10秒，然后它就建立了词汇表。它看起来是这样的:<br>
```
tokenizer_obj.word_index

Output: ('britains': 33206,
'labcoats': l2l364, 'steeled': l02939, 'geddon': 6755l, "rossilini's": 9l757, 'recreational': 27654, 'suffices': 43205, 'hallelujah': 30337, 'mallika': 30343, 'kilogram': l22493, 'elphic': l04809, 'feebly': 328l8, 'unskillful': 9l728, "'mistress'": l222l8, "yesterday's": 25908, 'busco': 85664, 'goobacks': 85670, 'mcfeast': 7ll75, 'tamsin': 77763,
"petron's": 72628,
"'lion": 87485,
'sams': 5834l, 'unbidden': 60042, "principal's": 44902, 'minutiae': 3l453, 'smelled': 35009, 'history\x97but': 75538,
 

'vehemently': 28626, 'leering': l4905, 'kýnay': l07654, 'intendend': l0l260, 'chomping': 2l885, 'nietsze': 76308, 'browned': 83646, 'grosse': l7645, "''gaslight''": 747l3, 'forseeing': l03637, 'asteroids': 30997, 'peevish': 49633, "attic'": l20936, 'genres': 4026, 'breckinridge': l7499, 'wrist': l3996, "sopranos'": 50345, 'embarasing': 92679, "wednesday's": ll84l3, 'cervi': 39092, 'felicity': 2l570, "''horror''": 56254, 'alarms': l7764, "'01": 294l0,
'leper': 27793, 'once\x85': l0064l, 'iverson': 66834, 'triply': ll7589, 'industries': l9l76, 'brite': l6733, 'amateur': 2459,
"libby's": 46942, 'eeeeevil': l204l3, 'jbc33': 5llll, 'wyoming': l2030, 'waned': 30059, 'uchida': 63203, 'uttter': 93299, 'irector': l23847, 'outriders': 95l56, 'perd': ll8465,
.
.
.}
```
&emsp;&emsp;因此，每个单词现在都与一个整数相关联;因此，单词“the”等于数字1:<br>
```
tokenizer_obj.word_index['the']
Output:
l
```
&emsp;&emsp;这里，and约等于数字2：<br>
```
tokenizer_obj.word_index['and']
Output:
2
```
&emsp;&emsp;单词a约等于3：<br>
```
tokenizer_obj.word_index['a']
Output:
3
```
&emsp;&emsp;接下来，单词movie等于数字17：<br>
```
tokenizer_obj.word_index['movie']
Output:
l7
```
&emsp;&emsp;单词film代表数字19：<br>
```
tokenizer_obj.word_index['film']
Output: 
19
```
&emsp;&emsp;所有这一切意味着，the是在数据集中使用最多的单词，and是在数据集中使用第二多的单词。因此，每当我们想将单词映射到整数标记时，我们都会得到这些数字。<br>
&emsp;&emsp;让我们试着以数字743为例，这是romantic的数字表示：<br>
```
tokenizer_obj.word_index['romantic']
Output:
743
```
&emsp;&emsp;因此，每当我们在输入文本中看到单词romantic时，我们就把它映射到标记整数743。我们再次使用索引表将训练集中文本中的所有单词转换为整数标记:<br>
```
input_text_train[l] Output:
'This is a really heart–warming family movie. It has absolutely brilliant animal training and "acting" (if you can call it like that) as well (just think about the dog in "How the Grinch stole Christmas"... it was plain bad training). The Paulie story is extremely well done, well reproduced and in general the characters are really elaborated too. Not more to say except that this is a GREAT MOVIE!<br /><br />My ratings: story 8.5/10, acting 7.5/10, animals+fx 8.5/10, cinematography 8/10.<br /><br />My overall rating: 8/10 – BIG FAMILY MOVIE AND VERY WORTH WATCHING!
```
&emsp;&emsp;当我们将文本转换为整数标记时，它就变成一个整数数组:<br>
```
np.array(input_train_tokens[1])

Output:
array([ 11, 6, 3, 62, 488, 4679, 236, 17, 9, 45, 419,
513, 1717, 2425, 2, 113, 43, 22, 67, 654, 9, 37,
12, 14, 69, 39, 101, 42, 1, 826, 8, 85, 1,
6418, 3492, 1156, 9, 13, 1042, 74, 2425, 1, 6419, 64,
6, 568, 69, 221, 69, 2, 8, 825, 1, 102, 23,
62, 96, 21, 51, 5, 131, 556, 12, 11, 6, 3,
78, 17, 7, 7, 56, 2818, 64, 723, 447, 156, 113,
702, 447, 156, 1598, 3611, 723, 447, 156, 633, 723, 156,
7, 7, 56, 437, 670, 723, 156, 191, 236, 17, 2,
52, 278, 147])
```
&emsp;&emsp;单词this变成了数字11，单词is变成了数字59，等等。<br>
&emsp;&emsp;我们还需要对剩下的文本进行转换:<br>
```
input_test_tokens = tokenizer_obj.texts_to_sequences(input_text_test)
```
&emsp;&emsp;现在，还有另一个问题，因为标记序列的长度根据原始文本的长度不同而不同，但递归单元可以处理任意长度的序列。TensorFlow的工作方式是，批处理中的所有数据都需要具有相同的长度。<br>
&emsp;&emsp;因此，我们既需要可以确保整个数据集中的所有序列都具有相同的长度，也可以编写自定义数据生成器来确保单个批处理中的序列具有相同的长度。现在，确保数据集中的所有序列具有相同的长度要简单得多，但问题是存在一些异常值。我们有一些句子超过2200个单词。如果我们所有的短句都超过2200个单词，这将严重危害我们的内存。所以我们要折中一下;首先，我们需要计算所有的单词，或者每个输入序列中的标记数。我们能得到一个序列的平均字数大约是221:<br>
```
total_num_tokens = [len(tokens) for tokens in input_train_tokens + input_test_tokens]
total_num_tokens = np.array(total_num_tokens)

#Get the average number of tokens np.mean(total_num_tokens)

Output:
22l.277l6
```
&emsp;&emsp;我们发现最多字数会超过2,200个:<br>
```
np.max(total_num_tokens)
Output:
2208
```
&emsp;&emsp;现在，平均值和最大值之间有一个很大的区别，如果我们只是在数据集中给所有的句子最大值，那它们都有2208个标记，我们会浪费很多内存。如果您的数据集有数百万个文本序列，那么这将是一个棘手的问题。<br>
&emsp;&emsp;所以我们要做的是做一个折中，我们将填补所有的序列，并截断那些太长的序列，使它们只有544个单词。544个单词我们是这样计算的，我们取数据集中所有序列的平均字数然后加上两个标准差:<br>
```
max_num_tokens = np.mean(total_num_tokens) + 2 * np.std(total_num_tokens)
max_num_tokens = int(max_num_tokens)
max_num_tokens

Output:
544
```
&emsp;&emsp;我们能从中得到什么呢?这种方式覆盖了数据集中95%的文本，所以说明只有5%的文本比544个单词长:<br>
```np.sum(total_num_tokens < max_num_tokens) / len(total_num_tokens) Output:
0.94532
```
&emsp;&emsp;现在，我们在Keras中调用这些函数。它们要么填充过短的序列(因此它们只会添加零)，要么截断过长的序列(如果文本太长，基本上就是剪掉一些单词)。
这里有一件重要的事情:我们可以在前后模式下进行填充和截断。假设我们有一个整数标记序列我们想填充它，因为它太短了。我们可以:<br>
&emsp;&emsp;任何一种方法都要在开头加上所有这些零，这样我们就能在末尾得到整数记号。<br>
&emsp;&emsp;或者用相反的方式，这样我们就在开始有了所有的数据，然后在结束的地方有所有的0。但如果我们回头看看前面的递归神经网络流程图，记住它是一步一步地处理序列的，所以如果我们开始处理0，它可能没有任何意义，内部状态可能一直保持0。当它终于看到某个特定单词的整数标记时，它就会知道，好了，现在我们开始处理数据。<br>
&emsp;&emsp;然而，如果所有的0都在最后，我们就会处理开始所有的数据;然后在循环单元中会有一些内部状态。而现在，我们看到了很多0，这可能会破坏我们刚刚计算的内部状态。这就是为什么在一开始填充零可能是个好主意。<br>
&emsp;&emsp;但另一个问题是，当我们截断文本时，如果文本很长，我们就会截断它使它适应544个单词或者数字的范围。现在，假设我们在中间的某个地方发现了这句话，this very good movie 或者 this is not。你知道，我们只对很长的序列做这个操作，但是我们正确地分类这篇文章时有可能失去重要的信息。这是我们在截断输入文本时做出的折中方法。更好的方法是创建一个批处理，然后在该批处理中填充文本。所以，当我们看到一个非常非常长的序列时，我们会对其他序列进行填充，使其长度相同。但我们不需要将所有这些数据存储在内存中，因为大多数数据都被浪费了。<br>
&emsp;&emsp;让我们返回并转换整个数据集，使其被截断和填充;由此得到一个大的数据矩阵:<br>
```
seq_pad = 'pre'

input_train_pad = pad_sequences(input_train_tokens, maxlen=max_num_tokens, padding=seq_pad, truncating=seq_pad)

input_test_pad = pad_sequences(input_test_tokens, maxlen=max_num_tokens, padding=seq_pad, truncating=seq_pad)
```
&emsp;&emsp;我们检查这个矩阵的形状:<br>
input_train_pad.shape

Output: (25000, 544)
input_test_pad.shape Output:
(25000, 544)

&emsp;&emsp;那么，让我们来看看填充前后的标记示例:<br>
```
np.array(input_train_tokens[1])

Output:
array([ 11, 6, 3, 62, 488, 4679, 236, 17, 9, 45, 419,
513, 1717, 2425, 2, 113, 43, 22, 67, 654, 9, 37,
12, 14, 69, 39, 101, 42, 1, 826, 8, 85, 1,
6418, 3492, 1156, 9, 13, 1042, 74, 2425, 1, 6419, 64,
6, 568, 69, 221, 69, 2, 8, 825, 1, 102, 23,
62, 96, 21, 51, 5, 131, 556, 12, 11, 6, 3,
78, 17, 7, 7, 56, 2818, 64, 723, 447, 156, 113,
702, 447, 156, 1598, 3611, 723, 447, 156, 633, 723, 156,
7, 7, 56, 437, 670, 723, 156, 191, 236, 17, 2,
52, 278, 147])
```
&emsp;&emsp;填充完成后，这个示例如下:<br>
```
input_train_pad[1]

Output:
array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 11, 6, 3, 62, 488, 4679, 236, 17, 9,
45, 419, 513, 1717, 2425, 2, 113, 43, 22, 67, 654,
9, 37, 12, 14, 69, 39, 101, 42, 1, 826, 8,
85, 1, 6418, 3492, 1156, 9, 13, 1042, 74, 2425, 1,
6419, 64, 6, 568, 69, 221, 69, 2, 8, 825, 1,
102, 23, 62, 96, 21, 51, 5, 131, 556, 12, 11,
6, 3, 78, 17, 7, 7, 56, 2818, 64, 723, 447,
156, 113, 702, 447, 156, 1598, 3611, 723, 447, 156, 633,
723, 156, 7, 7, 56, 437, 670, 723, 156, 191, 236,
17, 2, 52, 278, 147], dtype=int32)
```
&emsp;&emsp;此外，我们还需要一个向后表示的功能，以便将整数标记表示回文本单词;我们只需要这个函数。这是一个非常简单的辅助函数，让我们继续实现它:<br>
