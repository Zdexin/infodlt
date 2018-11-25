
# 第一章     数据科学-鸟瞰
&emsp;&emsp;数据科学或机器学习是让机器能够在不被告知或编程的情况下从数据集中学习的过程。例如，很难编写一个程序可以将手写数字并根据所写的图像从0-9中作为输入图像。这同样适用于将收到的电子邮件归类为垃圾邮件或非垃圾邮件的任务。为了解决这些任务，数据科学家从数据科学或机器学习领域使用这些来教计算机如何自动识别数字，给它一些解释功能，可以区分一个数字和另一个数字。同为针对垃圾邮件/非垃圾邮件的问题，我们可以通过特定的学习算法来教计算机如何对邮件进行分类，而不是使用正则表达式和编写上百条规则来对收到的电子邮件进行分类垃圾邮件和非垃圾邮件。<br><br>
&emsp;&emsp;对于垃圾邮件过滤应用程序，您可以使用基于规则的方法对其进行编码，但它不足以用于在你的邮件服务器中。建立一个学习系统是一个理想的解决方案。
您可能每天都在使用数据科学的应用程序，通常是在不知情的情况下使用。例如，您的国家可能正在使用一个系统来检测您寄出的信件的邮政编码，若要自动将其转发到正确的区域。如果你在使用亚马逊，他们经常推荐你购买的东西，是因为他们通过了解你经常搜索或购买的东西来做到这一点。<br>
&emsp;&emsp;建立一个经过学习/训练的机器学习算法将需要一个历史数据样本库，它将学习如何区分不同的示例并提出这样的方法，从这些数据中获得了知识和趋势。在那之后，受过训练的算法可用于对未见数据进行预测。学习算法将使用原始历史数据，并尝试从这些数据中获得一些知识和趋势。<br><br>
&emsp;&emsp;在本章中，我们将对数据科学有一个鸟瞰的视角，它是如何作为一个黑匣子工作的，以及数据科学家每天面临的挑战。我们将涵盖以下内容题目：
<br>
&emsp;&emsp;&emsp;&emsp;1.	通过一个例子了解数据科学<br>
&emsp;&emsp;&emsp;&emsp;2.	设计数据科学算法的过程<br>
&emsp;&emsp;&emsp;&emsp;3.	开始学习<br>
&emsp;&emsp;&emsp;&emsp;4.	鱼类识别/检测模型的实现<br>
&emsp;&emsp;&emsp;&emsp;5.	不同学习类型<br>
&emsp;&emsp;&emsp;&emsp;6.	数据大小和行业需求<br>
<br><br>
## 通过一个例子了解数据科学
&emsp;&emsp;为了说明为特定数据构建学习算法的生命周期和挑战，让我们考虑一个实际的例子。自然保护协会正在与其他渔业公司和部门合作渔农处负责监察渔业活动，并为日后的渔业作出保护。因此，他们希望在未来使用相机来扩大这一监测过程。这些摄像机的部署将产生大量的数据，手工处理将十分繁琐和昂贵。因此，该部门希望开发一种自动测试的学习算法。对不同种类的鱼类进行总体检测和分类，加快视频审查过程。<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/%E5%9B%BE%E7%89%871.png) <br>
图1.1显示了一个由保护配置相机拍摄的图像样本,这些图像将用于构建系统。<br><br>

&emsp;&emsp;因此，我们在这个例子中的目的是分离不同的物种，如金枪鱼，月鱼，和更多的渔船捕获的鱼。作为一个说明性的例子，我们可以将问题限制为两类，即金枪鱼和月鱼。<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/%E5%9B%BE%E7%89%872.png) <br>
图1.2 金枪鱼(左)和月鱼(右)<br><br>
&emsp;&emsp;在限制我们的问题只包含两种类型的鱼之后，我们可以从我们收集的一些随机图像中抽取一个样本，并开始注意这两种类型之间的一些物理区别。考试请考虑以下物理差异：<br>
&emsp;&emsp;&emsp;&emsp;长度：你可以看到，与月鱼相比，金枪鱼长得很长。<br>
&emsp;&emsp;&emsp;&emsp;宽度：月鱼比金枪鱼宽。<br>
&emsp;&emsp;&emsp;&emsp;颜色：你可以看到，月鱼倾向于红色，而金枪鱼则倾向于蓝色和白色，依此类推。<br>
我们可以利用这些物理差异作为特征，帮助我们的学习算法(分类器)区分这两种类型的鱼。<br>
&emsp;&emsp;对象的解释特征是我们在日常生活中用来区分周围事物的东西。甚至婴儿也会利用这些解释功能来了解周围的环境。数据科学也是如此，以便建立一个能够区分不同对象(例如，鱼类类型)的学习模型，我们需要给它一些可供学习的解释特性(例如，鱼的长度)。为了使模型更加确定，减少混淆误差，可以在一定程度上增加对象的解释特征。<br>
&emsp;&emsp;鉴于这两种鱼类之间的物理差异，这两种不同的鱼类种群有不同的模型或描述。因此，我们分类任务的最终目标是让分类器学习这些不同的模型，然后给出这两种类型之一的图像作为输入。分类器将通过选择与此图像最匹配的模型(金枪鱼模型或月鱼模型)对其进行分类。<br>
&emsp;&emsp;在这种情况下，金枪鱼和月鱼的收集将作为我们分类器的知识库。最初，知识库(训练样本)将被标记/标记，对于每个图像，您将事先知道它是金枪鱼还是月鱼。因此，分类器将使用这些训练样本来建模不同类型的鱼，然后我们可以使用训练阶段的输出来自动标记未标记/未标记的鱼，即分类在训练阶段没有看到。这种未标记的数据通常称为看不见的数据。生命周期的培训阶段如下图所示：监督数据科学是从已知目标或输出的历史数据(如鱼类类型)中学习，然后使用这个学习模型来预测我们不知道目标/输出案例或数据样本。<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/1图片.png) <br>
图1.3 训练阶段运行周期<br><br>
&emsp;&emsp;让我们看看分类器的培训阶段将如何工作：<br>
&emsp;&emsp;&emsp;&emsp;预处理：在这一步中，我们将尝试利用相关的分割技术从图像中分割出鱼。<br>
&emsp;&emsp;&emsp;&emsp;特征提取：通过减去背景将鱼从图像中分割出来，然后测量每幅图像的物理差异(长度、宽度、颜色等)。最后，你会得到一些信息如图1.4所示。<br>
&emsp;&emsp;最后，我们将这些数据输入到分类器中，以便对不同的鱼类类型进行建模。正如我们所看到的，我们可以根据我们提出的物理差异(如长度、宽度和颜色)，在视觉上区分金枪鱼和月鱼。<br>
&emsp;&emsp;我们可以利用长度特征来区分这两种鱼。因此，我们可以通过观察它们的长度和它是否超过某个值(长度)来区分它们。<br>
&emsp;&emsp;因此，根据我们的培训样本，我们可以得出以下规则：<br>
```python
If length(fish)> length* then label(fish) =Tuna 
Otherwise label(fish) = Opah
```  
&emsp;&emsp;为了找到这个长度，我们可以根据训练样本进行长度测量。因此，假设我们得到这些长度测量，并得到如下直方图：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/2图片.png) <br>
图1.4 两种鱼类长度测量的直方图<br><br>
&emsp;&emsp;在这种情况下，我们可以根据长度特征导出一个规则，并区分金枪鱼和月鱼。在这个特殊的例子中，我们可以知道长度是7。这样我们就可以更新前面的规则：<br>
```python
If length(fish)> 7 then label(fish) =Tuna 
Otherwise label(fish) = Opah
```  
&emsp;&emsp;正如您可能注意到的，这不是一个有希望的结果，因为这两个直方图之间的重叠，因为长度特征不是一个完美的特点，仅用于区分这两种类型。因此，我们可以尝试合并更多的功能，如宽度，然后结合他们。因此，如果我们设法测量训练样本的宽度，我们可能会得到类似于跟随，接着:<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/图片3.png) <br>
图1.5 两种鱼类宽度测量的直方图<br><br>
&emsp;&emsp;正如您所看到的，依赖于一个特性不会给出准确的结果，输出模型会造成许多错误分类。所以，我们可以用某种方式将这两个特性结合起来，使其看上去很合理。<br>
&emsp;&emsp;因此，如果我们将这两个特性结合起来，我们可能会得到类似于下面的图形的东西：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/图片4.png) <br>
图1.6 两种鱼类的长度和宽度测量子集之间的组合<br><br>
&emsp;&emsp;结合长度和宽度特征的读数，我们将得到像前面的图表中的散点图。我们有红色的点代表金枪鱼，绿色的点代表opah鱼，我们可以建议这条黑线作为区分这两种鱼的规则或决策边界。例如，如果一条鱼的读数高于这个决定边界，那么它就是一条金枪鱼；否则，它将被预测为一条月鱼。<br>
&emsp;&emsp;我们可以设法增加规则的复杂性，以避免任何错误，并获得如下图形中的决策边界：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/图片5.png) <br>
图1.7 增加决策边界的复杂性以避免对培训数据的错误分类<br><br>
&emsp;&emsp;该模型的优点是我们在训练样本上得到了几乎0种错误分类。但实际上，这并不是使用数据科学的目的。数据科学的目标是建立一个模型，该模型能够对未见的数据进行良好的概括和执行。为了了解我们是否建立了一个推广的模型，我们将引入一个新的阶段，称为测试阶段，在这个阶段中，我们给训练的模型一个未标记的图像并希望模型可以指定正确的标签(Tuna和opah)。<br>
&emsp;&emsp;数据科学的最终目标是建立一个在生产中运行良好的模型，而不是训练集。所以，当你看到你的模型在训练中表现良好时，不要高兴，就像图1.7中的模型一样。大多数情况下，这种模型在识别图像中的鱼类类型时效果不佳。你的模型只有在训练集上才能正常工作，这一事件被称为“过度拟合”，而且大多数从业者都落入了这个陷阱。<br>
&emsp;&emsp;代替提出这样一个复杂的模型，您可以使用一个不那么复杂的模型，它将在测试阶段泛化。下图显示了如何使用不太复杂的模型来获得较少的错误分类错误，并对未见数据进行概括：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/图片6.png) <br>
图 1.8 使用不太复杂的模型，以便能够对测试样本(未见数据)进行归纳。<br><br>
## 数据科学算法的设计过程
&emsp;&emsp;不同的学习系统通常遵循相同的设计过程。它们从获取知识库开始，从数据中选择相关的解释性特征，通过一系列候选学习算法，同时关注每一个算法，最后是评估过程，衡量培训过程的成功程度。<br>
&emsp;&emsp;在本节中，我们将更详细地讨论所有这些不同的设计步骤：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/图片7.png) <br>
图1.11 “模型学习过程大纲”<br><br>
### 数据预处理
&emsp;&emsp;学习周期的这一部分代表了我们算法的知识库。因此，为了帮助学习算法对未知数据做出准确的决策，我们需要以最好的形式提供这个知识库。因此，我们的数据可能需要大量的清理和预处理(转换)。<br>
#### 数据清理
&emsp;&emsp;大多数数据集都需要这个步骤，在这个步骤中，您可以消除错误、噪声和冗余。我们需要我们的数据准确、完整、可靠和不偏不倚，因为使用不好的知识库可能会产生许多问题，例如：<br>
&emsp;&emsp;&emsp;&emsp;1.不准确和有偏见的结论<br>
&emsp;&emsp;&emsp;&emsp;2.增加误差<br>
&emsp;&emsp;&emsp;&emsp;3.降低了通用性，这是该模型能够很好地处理它以前没有训练过的未见数据<br>
##### 数据预处理
&emsp;&emsp;在这一步中，我们将对数据进行一些转换，以使其一致和具体。在预处理数据时，您可以考虑许多不同的转换：<br>
&emsp;&emsp;&emsp;&emsp;重命名：这意味着将分类值转换为数字，因为如果与某些学习方法一起使用，则分类值是危险的，而且数字也会在这些值之间强制排序。<br>
&emsp;&emsp;&emsp;&emsp;重新标度(正常化)：将连续值转换/包围到某个范围，典型地 [-1,1]或[0,1]<br>
&emsp;&emsp;&emsp;&emsp;新特点：从现有的特征中提炼出新的特征。例如，肥胖因素=体重/身高。<br>
### 特征选择
&emsp;&emsp;一个样本的解释特性(输入变量)的数量可能是巨大的，当您获得xi=(xi,xi^2,xi^3,...,xi^d)作为训练样本(观察/示例)时，d是非常大的。这方面的一个例子可以是文档分类任务，如您得到了10000个不同的单词，而输入变量将是不同单词出现的次数。<br>
&emsp;&emsp;大量的输入变量可能是有问题的，有时也是一个麻烦，因为我们有很多输入变量和很少的训练样本来帮助我们学习过程。为了避免有大量的输入变量(维数的麻烦)，数据科学家使用降维技术从输入变量中选择一个子集。例如，在文本分类任务中，他们可以执行以下操作：<br>
&emsp;&emsp;&emsp;&emsp;1.提取相关输入(例如，相互信息度量)<br>
&emsp;&emsp;&emsp;&emsp;2.主成分分析（PCA）<br>
&emsp;&emsp;&emsp;&emsp;3.分组(聚类)相似的单词(这使用相似性度量)<br>
### 模型选择
&emsp;&emsp;这一步是在使用任何降维技术来选择输入变量的适当子集之后执行的。选择输入变量的适当子集将使学习过程的其余部分非常简单。在这一步中，您正试图找到需要学习的正确模型。<br>
&emsp;&emsp;如果您有任何数据科学的经验，并将学习方法应用于不同领域和不同类型的数据，那么您将发现这一步骤很容易，因为它需要事先了解。现在你的数据看起来，什么样的假设符合你的数据的性质，并在此基础上，你选择了正确的学习方法。如果您没有任何先验知识，这也没关系，因为您可以通过猜测和尝试不同参数设置的不同学习方法来完成这一步，并选择给它一个比测试集的性能更好的方法。<br>
&emsp;&emsp;此外，初始数据分析和可视化将帮助您很好地猜测数据的分布形式和性质。<br>
### 学习过程
&emsp;&emsp;通过学习，我们指的是用于选择最佳模型参数的优化标准。这方面有各种优化标准：<br>
&emsp;&emsp;&emsp;&emsp;1.均方误差<br>
&emsp;&emsp;&emsp;&emsp;2.最大似然准则<br>
&emsp;&emsp;&emsp;&emsp;3. 最大后验概率<br>
&emsp;&emsp;优化问题可能很难解决，但模型和误差函数的正确选择会带来很大的影响。<br>
### 评估模型
&emsp;&emsp;在这一步中，我们尝试测量我们的模型对未见数据的泛化误差。由于我们只有特定的数据而事先不知道任何看不见的数据，所以我们可以随机地从数据中选择一个测试集，并且在训练过程中不要使用它，它的作用就像有效的看不见的数据。可以通过不同的方法来评估所选模型的性能：<br>
&emsp;&emsp;&emsp;&emsp;1.简单的持久化方法，将数据划分为训练集和测试集。<br>
&emsp;&emsp;&emsp;&emsp;2.其他基于交叉验证和随机次抽样的复杂方法<br>
&emsp;&emsp;我们在这一步的目标是比较不同模型在相同数据上的预测性能，并选择一个测试误差较好(较小)的模型，这将给我们带来更好的结果。对未见数据的泛化错误。通过使用统计方法测试结果的重要性，您也可以更确定泛化错误。<br>
## 学习
&emsp;&emsp;构建机器学习系统会带来一些挑战和问题，我们将在本节中讨论这些挑战和问题。这些问题中有许多是特定领域的，而另一些则不是。<br>
### 学习面临的挑战
&emsp;&emsp;以下概述了您在构建学习系统时通常将面临的挑战和问题。<br>
#### 特征提取-特征工程
&emsp;&emsp;特征提取是构建学习系统的关键步骤之一。如果您在这个挑战中做得很好，选择适当的/正确的特性数，那么剩下的学习内容过程将很容易。此外，特征提取与领域有关，需要事先了解哪些特征对特定任务可能很重要。例如，我们的鱼类识别系统的功能将不同于垃圾邮件检测或识别指纹。<br>
&emsp;&emsp;特征提取步骤从您拥有的原始数据开始。然后构建关于学习任务信息的派生变量/值(特性)，并为下一步的学习和评估(泛化)提供便利。<br>
&emsp;&emsp;一些任务将具有大量的特征和较少的训练样本(观察)，以促进后续的学习和概括过程。在这种情况下，数据科学家使用降维技术将大量的特征减少到更小的集合。<br>
#### 其他因素
&emsp;&emsp;在鱼的识别任务中，你可以看到鱼的长度、重量、鱼的颜色以及船的颜色可能不同，而且在图像中可能会出现阴影、低分辨率的图像和其他物体。所有这些问题都会影响到建议的解释性特征的重要性，这些特征应该是关于我们的鱼类分类任务的信息。<br>
&emsp;&emsp;在这种情况下，工作是有帮助的。例如，有人可能会想到检测船ID，并屏蔽船的某些部分，这些部分很可能不包含我们的系统检测到的任何鱼。这项工作将限制我们的搜索空间。<br>
#### 过拟合
&emsp;&emsp;在鱼类识别任务中，我们试图通过增加模型的复杂性和对训练样本的每一个实例进行完美的分类来提高模型的性能。正如我们稍后将看到的，这样的模型不适用于看不见的数据(例如我们将用于测试模型性能的数据)。如果训练的模型在训练样本上工作得很好，但在测试样本上却没有很好的表现，这就被称为过度拟合。<br>
&emsp;&emsp;如果您仔细阅读本章的后半部分，我们将建立一个学习系统，目的是将训练样本作为我们模型的知识库，以便从看不见的数据中学习和推广。训练模型的性能误差对训练数据不感兴趣；相反，我们感兴趣的是经过训练的模型在还没有进入训练阶段的测试样本上的性能(泛化)误差。<br>
#### 机器学习算法的选择
&emsp;&emsp;有时，您对用于特定任务的模型的执行不满意，并且需要另一个模型类。每种学习策略都有自己的假设。关于它将用作学习基础的信息。作为信息研究人员，您必须发现哪些怀疑最适合您的信息；通过这一点，您将有能力尝试一种模型并拒绝另一种模型。<br>
#### 先验知识准备
&emsp;&emsp;正如在模型选择和特征提取的概念中所讨论的，如果您事先了解以下内容，则可以处理这两个问题：<br>
&emsp;&emsp;&emsp;&emsp;1.适当的特征<br>
&emsp;&emsp;&emsp;&emsp;2.模型选择部分<br>
&emsp;&emsp;对鱼类识别系统中的解释性特征的先验了解使我们能够在不同类型的鱼中进行区分。我们可以通过努力想象我们的信息来进行宣传，并对不同鱼类分类的信息类型有一定的了解。在此基础上，可以选择APT模型族。<br>
#### 缺失值
&emsp;&emsp;缺少功能主要是因为缺少数据或选择“不告诉”选项。在学习过程中，我们如何处理这种情况呢？例如，假设我们发现特定鱼类类型的宽度由于某种原因而丢失。有许多方法可以处理这些缺失的功能。<br>
## 鱼类识别/检测模型的实现
&emsp;&emsp;为了特别介绍机器学习和深度学习的功能，我们将实现鱼的识别实例。不需要了解代码的内部细节。本节的重点是概述一个典型的机器学习管道。<br>
&emsp;&emsp;我们对这一任务的知识库将是一堆图像，每个图片都被标记为opah或tuna。对于这个实现，我们将使用一种在成像和计算机视觉领域取得突破的深度学习架构。这种结构称为卷积神经网络(CNNs)。它是一种深度学习结构，它利用图像处理的卷积运算从图像中提取特征，从而解释我们想要分类的对象。现在，你可以把它想象成一个魔盒，它将拍摄我们的图像，从中学习如何区分我们的两个类(opah和tuna)，然后我们将测试这个盒子的学习过程。用未贴标签的图片进行测试，看看它是否能分辨出图像中是哪种鱼。<br>
&emsp;&emsp;不同类型的学习将在后面的一节中讨论，因此稍后您将了解为什么我们的鱼识别任务属于监督学习类别。<br>
&emsp;&emsp;在本例中，我们将使用Keras。目前，您可以将Keras看作是一个API，它使构建和使用深度学习的方式比平常更容易。我们开始吧！我们在Keras网站上有：<br>
&emsp;&emsp;&emsp;&emsp;Keras是一个高级的神经网络API，用Python编写，能够运行在TensorFlow、CNTK或Theano之上。它是以快速试验为重点开发的。能从一个想法到另一个结果，尽可能少的延迟是做好研究的关键。<br>
### 知识库/数据集
&emsp;&emsp;正如我们前面提到的，我们需要一个历史数据基础，用来教学习算法关于它以后应该完成的任务。但是我们还需要另一个数据集来测试它在学习过程之后完成任务的能力。总之，在学习过程中，我们需要两种类型的数据集：<br>
&emsp;&emsp;&emsp;&emsp;1.第一个是知识库，其中有输入数据及其相应的标签，如鱼类图像及其相应的标签(opah或tuna)。这些数据将被输入学习算法中，从中学习，并试图发现模式/趋势，这将有助于对未标记的图像进行分类。<br>
&emsp;&emsp;&emsp;&emsp;2.第二种主要是测试模型的能力，将它从知识库中学到的东西应用到未标记的图像或未见数据上，看看它是否正常工作。<br>

&emsp;&emsp;正如你所看到的，我们只有作为学习方法的知识库的数据。我们手头的所有数据都有与之相关的正确输出。因此，我们需要以某种方式组成这些数据，这些数据没有任何与其相关的正确输出(我们将对其应用该模型)。<br>
&emsp;&emsp;在执行数据科学时，我们将执行以下操作：<br>
&emsp;&emsp;&emsp;&emsp;培训阶段：我们从知识库中提供我们的数据，并通过将输入的数据及其正确的输出输入模型中来训练我们的学习方法/模型。<br>
&emsp;&emsp;&emsp;&emsp;验证/测试阶段：在这一阶段，我们将衡量这个经过训练的模型做得有多好。我们还使用不同的模型属性技术来衡量我们训练过的模型的性能。（分类器的分类错误，IR模型的返回和精度，等等)。<br>

&emsp;&emsp;验证/测试阶段通常分为两个步骤：<br>
&emsp;&emsp;&emsp;&emsp;1.在第一步中，我们使用不同的学习方法/模型，并根据验证数据(验证步骤)选择性能好的学习方法/模型。<br>
&emsp;&emsp;&emsp;&emsp;2.然后根据测试集(测试步骤)对所选模型的精度进行测量和报告。<br>
&emsp;&emsp;现在，让我们看看我们如何得到这些数据，我们将应用这个模型，并看看它的训练有多好。<br>

&emsp;&emsp;由于没有正确的输出，我们没有任何训练样本，所以我们可以从我们将要使用的原始训练样本中合成一个样本。所以我们可以把我们的数据样本分成三个不同的部分（图1.9）<br>
&emsp;&emsp;&emsp;&emsp;1.训练集：这将作为我们的模型的知识库。通常，会是70%的原始数据样本。<br>
&emsp;&emsp;&emsp;&emsp;2.验证集：这将用于在一组模型中选择性能最好的模型。通常这将是10%的原始数据样本。<br>
&emsp;&emsp;&emsp;&emsp;3.测试集：这将用于测量和报告所选模型的准确性。通常，它与验证集一样大。<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/图片8.png) <br><br>
图 1.9 将数据拆分为训练、验证和测试集。<br><br>
&emsp;&emsp;如果您只使用一种学习方法，则可以取消验证集，并将数据重新拆分为仅训练和测试集。通常，数据科学家使用75/25或者70/30作为百分比。<br>
### 数据分析预处理
&emsp;&emsp;在这一部分中，我们将对输入图像进行分析和预处理，并将其以可接受的格式用于我们的学习算法，即这里的卷积神经网络。<br>
&emsp;&emsp;因此，让我们从导入此实现所需的包开始：<br>
```python
import numpy as np 
np.random.seed(2018)
import os 
import glob 
import cv2 
import datetime
import pandas as pd 
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.cross_validation import KFold 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D,ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping 
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import  __version__  as keras_version
```  
&emsp;&emsp;为了使用数据集中提供的图像，我们需要使它们具有相同的大小。OpenCV是一个很好的选择，从OpenCV网站：<br>
&emsp;&emsp;&emsp;&emsp;OpenCV(开放源码计算机视觉库)是在BSD许可下发布的，因此它对学术和商业都是免费的。它有C，Python和Java接口，支持Windows，Linux，MacOS，IOS和Android。OpenCV是为了提高计算效率而设计的，并且非常注重实时应用程序。该库采用优化的C/C语言编写，充分利用了多种c+的优点。矿石加工通过OpenCL启用，它可以利用底层异构计算平台的硬件加速。<br>
&emsp;&emsp;您可以通过使用python包管理器安装OpenCV，pip install OpenCV-python<br>
```python
# Parameters
# ––––––––––
# x : type
#	Description of parameter `x`. 
def rezize_image(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (32, 32), cv2.INTER_LINEAR) 
    return img_resized
```  
&emsp;&emsp;现在我们需要加载数据集的所有训练样本，并根据前面的函数调整每幅图像的大小。因此，我们将实现一个函数，它将从针对每种鱼类类型的不同文件夹：<br>
```python
def load_training_samples():
#用于保存培训输入和输出变量的变量
    train_input_variables = []
    train_input_variables_id = []
    train_label = []
    # 扫描鱼类型的每个文件夹中的所有图像
    print('Start Reading Train Images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        folder_index = folders.index(fld)
        print('Load folder (} (Index: (})'.format(fld, folder_index)) 
        imgs_path = os.path.join('..', 'input', 'train', fld, '*.jpg') 
        files = glob.glob(imgs_path)
        for file in files:
            file_base = os.path.basename(file)
            # 调整图像大小
            resized_img = rezize_image(file)
            # 将处理后的图像附加到分类器的输入/输出变量中
            train_input_variables.append(resized_img) 
            train_input_variables_id.append(file_base) 
            train_label.append(folder_index)
        return train_input_variables, train_input_variables_id, train_label
```  
&emsp;&emsp;正如我们所讨论的，我们有一个测试集，它将作为不可见的数据来测试我们的模型的泛化能力。因此，我们需要对图像进行同样的测试；加载它们并执行调整大小的处理：<br>
```python
#加载测试样本，用于测试模型的训练效果。
def load_testing_samples():
# 从测试文件夹中扫描图像
    imgs_path = os.path.join('..', 'input', 'test_stgl', '*.jpg')
    files = sorted(glob.glob(imgs_path))
    # 保存测试样本的变量 
    testing_samples = []
    testing_samples_id = []
    #处理图像并将它们附加到我们拥有的数组中。 
    for file in files:
        file_base = os.path.basename(file)
        # Image resizing
        resized_img = rezize_image(file) 
        testing_samples.append(resized_img) 
        testing_samples_id.append(file_base)
    return testing_samples, testing_samples_id
```  
&emsp;&emsp;现在，我们需要将前面的函数调用到另一个函数中，该函数将使用LOAD_TRANING_SAMPLES()函数，以便加载和调整训练样本的大小。此外，它还将添加几行代码，将培训数据转换为NumPy格式，重新构造该数据以适应我们的分类器，最后将其转换为浮动：<br>
```python
def load_normalize_training_samples():
    # 调用Load函数以加载和调整训练样本的大小
    training_samples, training_label, training_samples_id = load_training_samples()
    # 将加载和调整大小的数据转换为Numpy格式 
    training_samples = np.array(training_samples, dtype=np.uint8) 
    training_label = np.array(training_label, dtype=np.uint8)
    # 重塑训练样本
    training_samples = training_samples.transpose((0,3,1,2))
    # 将培训样本和培训标签转换为浮动格式
    training_samples = training_samples.astype('float32')
    training_samples = training_samples/255
    training_label = np_utils.to_categorical(training_label, 8) 
    return training_samples, training_label, training_samples_id
```  
&emsp;&emsp;我们还需要对测试做同样的工作：<br>
```python
def load_normalize_testing_samples():
    # 调用LOAD函数以加载和调整测试样本的大小
    testing_samples, testing_samples_id = load_testing_samples()
    # 将加载和调整大小的数据转换为Numpy格式
    testing_samples = np.array(testing_samples, dtype=np.uint8)
    # 重塑测试样本
    testing_samples = testing_samples.transpose((0,3,1,2))
    # 将测试样本转换为浮动格式
    testing_samples = testing_samples.astype('float32') 
    testing_samples = testing_samples / 255
    return testing_samples, testing_samples_id
```  
### 建立模型
&emsp;&emsp;现在是建立模型的时候了。正如我们所提到的，我们将使用一种称为CNN的深度学习架构作为这项鱼类识别任务的学习算法。同样，您不需要理解本章中的任何前面或即将出现的代码，因为我们只是在演示如何通过只使用几行代码，并借助Keras和TensorFlow作为一个深度学习平台来解决复杂的数据科学任务。<br>
&emsp;&emsp;还请注意，CNN和其他深度学习架构将在后面的章节中进行更详细的解释：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/图片9.png) <br>
图 1.10 CNN结构<br><br>
&emsp;&emsp;因此，让我们继续创建一个函数，它将负责创建CNN的体系结构，用于我们的鱼类识别任务：<br>
```python
# 创建CNN模型体系结构
def create_cnn_model_arch():
    pool_size = 2 # 我们将在整个过程中使用2x2池化层
    conv_depth_l = 32 
    conv_depth_2 = 64
    drop_prob = 0.5  
    hidden_size = 32 
    num_classes = 8 
    # Conv [32] –> Conv [32] –> Pool 
    cnn_model = Sequential()
    cnn_model.add(ZeroPadding2D((l, l), input_shape=(3, 32, 32), dim_ordering='th'))
    cnn_model.add(Convolution2D(conv_depth_l, kernel_size, kernel_size, activation='relu',
    dim_ordering='th'))
    cnn_model.add(ZeroPadding2D((l, l), dim_ordering='th')) 
    cnn_model.add(Convolution2D(conv_depth_l, kernel_size, kernel_size,activation='relu', dim_ordering='th'))
    cnn_model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(2, 2),dim_ordering='th'))
    # Conv [64] –> Conv [64] –> Pool 
    cnn_model.add(ZeroPadding2D((l, l), dim_ordering='th'))
    cnn_model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size, activation='relu',dim_ordering='th'))
    cnn_model.add(ZeroPadding2D((l, l), dim_ordering='th')) 
    cnn_model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size,activation='relu',dim_ordering='th')) 
    cnn_model.add(MaxPooling2D(pool_size=(pool_size, pool_size),strides=(2, 2),dim_ordering='th'))
    # Now flatten to lD, apply FC then ReLU (with dropout) and finally softmax(output layer)
    cnn_model.add(Flatten()) 
    cnn_model.add(Dense(hidden_size, activation='relu')) 
    cnn_model.add(Dropout(drop_prob)) 
    cnn_model.add(Dense(hidden_size, activation='relu')) 
    cnn_model.add(Dropout(drop_prob)) 
    cnn_model.add(Dense(num_classes, activation='softmax'))
    # 启动随机梯度下降优化器
    stochastic_gradient_descent = SGD(lr=le-2, decay=le-6, momentum=0.9,nesterov=True)
    cnn_model.compile(optimizer=stochastic_gradient_descent,
    # 使用随机梯度下降优化器
    loss='categorical_crossentropy')# 使用交叉熵损失函数
    return cnn_model
```  
&emsp;&emsp;在开始对模型进行训练之前，我们需要使用一种模型评估和验证方法来帮助我们评估我们的模型并查看它的泛化能力。为此，我们将使用一种叫做k折叠交叉验证的方法。同样，您不需要理解此方法或它是如何工作的，因为我们稍后将详细解释此方法。<br>
&emsp;&emsp;因此，让我们开始并创建一个函数，它将帮助我们评估和验证模型：<br>
```python
#以折叠交叉验证为验证方法的模型 
def create_model_with_kfold_cross_validation(nfolds=10):
    batch_size = 16 # 在每次迭代中，我们同时考虑32个训练示例。
    num_epochs = 30 # 我们在整个训练集上迭代2OO次。
    random_state =51 # 在同一平台上控制结果重复性的随机性
    # 在将训练样本输入到创建的CNN模型之前加载和规范化
    training_samples, training_samples_target, training_samples_id =load_normalize_training_samples() 
    yfull_train = dict()
    # 提供培训/测试指标，以分割培训样本中的数据
    # which is splitting data into lO consecutive folds with shuffling 
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True,random_state=random_state)
    fold_number = 0 # 折数初值
    sum_score = 0 #总分(每次迭代时将增加)
    trained_models = [] # 存储每个迭代的模型
    # 获取培训/测试样本
    #t培训/测试指数
    for train_index,test_index in kf: 
        cnn_model = create_cnn_model_arch()
        training_samples_X = training_samples[train_index] # 获取训练输入变量
        training_samples_Y = training_samples_target[train_index] # 获取培训输出/标签变量
        validation_samples_X = training_samples[test_index] # 获取验证输入变量
        validation_samples_Y = training_samples_target[test_index] # 获取验证输出/标签变量
        fold_number += 1
        print('Fold number {} from {}'.format(fold_number, nfolds)) 
        callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        # 拟合CNN模型，给出定义的设置
        cnn_model.fit(training_samples_X, training_samples_Y,batch_size=batch_size,
        nb_epoch=num_epochs, shuffle=True, verbose=2,
        validation_data=(validation_samples_X,
        validation_samples_Y), callbacks=callbacks)
        # 基于验证集的训练模型泛化能力度量
        predictions_of_validation_samples = cnn_model.predict(validation_samples_X.astype('float32'), batch_size=batch_size, verbose=2)
        current_model_score = log_loss(Y_valid, predictions_of_validation_samples)
        print('Current model score log_loss: ', current_model_score) 
        sum_score += current_model_score*len(test_index)
        # 存储有效预测
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_of_validation_samples[i]
            # 存储经过训练的模型
            trained_models.append(cnn_model)
            # 用当前模型计算的分数增量和得分值
        overall_score = sum_score/len(training_samples) 
        print("Log_loss train independent avg: ", overall_score)
        #在此阶段报告模型损失
        overall_settings_output_string = 'loss_' + str(overall_score) +'_folds_' + str(nfolds) + '_ep_' + str(num_epochs)
        return overall_settings_output_string, trained_models
```  
&emsp;&emsp;现在，在建立模型并使用k-折叠交叉验证方法对模型进行评估和验证后，我们需要在测试集上报告经过训练的模型的结果。为了做到这一点，我们也将使用k倍交叉验证，但这一次的测试，看看我们的训练模型有多好。<br>
&emsp;&emsp;因此，让我们定义一个函数，它将把经过训练的CNN模型作为输入，然后使用我们拥有的测试集来测试它们：<br>
```python
#测试模型的训练效果
def test_generality_crossValidation_over_test_set( overall_settings_output_string, cnn_models):
    batch_size = 16 # 在每次迭代中，我们同时考虑32个训练示例。
    fold_number = 0 # 折叠迭代器
    number_of_folds = len(cnn_models) # 根据训练步骤中使用的值创建折叠数
    yfull_test = [] # 变量来保存测试集的总体预测。
    #在测试集上执行实际的交叉验证测试过程 
    for j in range(number_of_folds):
        model = cnn_models[j] 
        fold_number += 1
        print('Fold number {} out of {}'.format(fold_number, number_of_folds))
        #加载和正规化测试样本
        testing_samples, testing_samples_id =load_normalize_testing_samples()
        #在当前测试折叠上调用当前模型
        test_prediction = model.predict(testing_samples,batch_size=batch_size, verbose=2) 
        yfull_test.append(test_prediction)
    test_result = merge_several_folds_mean(yfull_test, number_of_folds)
    overall_settings_output_string = 'loss_' +overall_settings_output_string \
    + '_folds_' + str(number_of_folds)
    format_results_for_types(test_result, testing_samples_id, overall_settings_output_string)
```  
#### 模型训练与测试
&emsp;&emsp;现在，我们准备开始模型训练阶段，调用创建_MODEL_WITY_KFLODE_交叉验证()的主要函数，使用10倍交叉验证来建立和训练CNN模型；然后，我们可以调用测试函数来度量模型泛化到测试集的能力：<br>
```python
#开始模型培训和测试
if __name__== '_main_':
    info_string, models = create_model_with_kfold_cross_validation() 
    test_generality_crossValidation_over_test_set(info_string, models)
```  
#### 鱼类识别
&emsp;&emsp;在解释了鱼类识别示例的主要构建块之后，我们已经准备好看到所有代码片段连接在一起，并看到我们如何成功地构建了一个如此复杂的系统。完整的代码放在书的附录部分。<br>
## 不同学习类型
&emsp;&emsp;根据Arthur_Samuel(https://en.wikipedia.org/wiki/Arthur_Samuel )的说法，数据科学赋予了计算机学习的能力，而无需显式编程。因此，任何在没有显式编程的情况下使用培训示例以便对未见数据做出决策的软件都被认为是学习的。数据科学或学习有三种不同的形式。<br>
图1.12显示了常用的数据科学/机器学习类型：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/图片10.png) <br>
### 有监督的学习
&emsp;&emsp;大多数数据科学家使用监督学习。监督学习是指您有一些解释性特性，称为输入变量(X)，并且您的标签是Associa使用训练样本，称为输出变量(Y)。任何监督学习算法的目的是学习从输入变量(X)到输出变量(Y)的映射函数：<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Y=f(X)<br>
&emsp;&emsp;因此，监督学习算法将尝试学习从输入变量(X)到输出变量(Y)的映射，以便以后可以用来预测看不见的样本。<br>
&emsp;&emsp;图1.13 显示了用于任何受监督的数据科学系统的典型工作流：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/图片11.png) <br>
图1.13 典型的监督学习工作流/管道。上面的部分展示了从将原始数据输入到特征提取模块开始的培训过程，在该模块中，我们将选择有意义的解释性特征来表示我们的数据。然后将提取的/选择的解释性特征与训练集相结合，将其反馈到学习算法中进行学习。然后我们做一些模型评估来调优。参数和学习算法，以获得最佳的数据样本。<br><br>
&emsp;&emsp;这种学习被称为监督学习，因为你得到了与它相关的每个训练样本的标签/输出。在这种情况下，我们可以说学习过程是由主管监督的。该算法对训练样本进行决策，并根据数据的正确标签进行校正。当监督学习算法达到可接受的精度时，学习过程就会停止。<br>
&emsp;&emsp;有监督的学习任务有两种不同的形式：回归和分类：<br>
&emsp;&emsp;&emsp;&emsp;分类：分类任务是当标签或输出变量是一个类别时，例如金枪鱼或opah或垃圾邮件和非垃圾邮件。<br>
&emsp;&emsp;&emsp;&emsp;回归：一个回归任务是当输出变量是一个实际值时，例如房价或高度<br>
### 无监督学习
&emsp;&emsp;无监督学习被认为是信息研究者使用的第二种最常见的学习方式。在这种类型的学习中，只给出了解释特征或输入变量(X)，没有任何相应的标签或输出变量。无监督学习算法的目标是接收信息中隐藏的结构和实例。这种学习被称为无监督学习，因为没有与训练样本相关的标记。因此，这是一个没有修正的学习过程，该算法将尝试自己找到基本结构。<br>
&emsp;&emsp;无监督学习可以进一步分为两种形式-聚类和关联任务：<br>
&emsp;&emsp;&emsp;&emsp;聚类：集群任务是您希望发现类似的培训示例组并将它们分组的地方，例如按主题对文	档进行分组。<br>
&emsp;&emsp;&emsp;&emsp;关联：关联规则学习任务是您希望在培训示例中发现一些描述关系的规则的地方（例如看X电影的人也倾向于看电影Y）<br>
&emsp;&emsp;图1.14 展示了一个无监督学习的小例子，在那里我们得到了分散的文档，并且我们试图将相似的文档组合在一起：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter01/图片12.png) <br>
图1.14“显示了如何无监督地使用相似度量，例如欧几里得距离，将相似的文档组合在一起，并为它们绘制决策边界。<br><br>
### 半监督学习
&emsp;&emsp;半监督学习是介于监督学习和非监督学习之间的一种学习方式，有输入变量(X)的训练实例，但只有一部分是使用输出变量(Y)标记标注的。<br>
&emsp;&emsp;这类学习的一个很好的例子是Flickr(https://www.flickr.com/ )，用户上传了很多图片，但只有一些图片被标记了)(比如日落、海洋和狗)和其余的都没有贴上标签。<br>
&emsp;&emsp;要解决属于这类学习的任务，您可以使用以下之一或它们的组合：<br>
&emsp;&emsp;&emsp;&emsp;监督学习：学习/训练学习算法，给出未标记数据的预测，然后反馈整个训练样本，从中学习并预测未见数据。<br>
&emsp;&emsp;&emsp;&emsp;无监督学习：使用无监督学习算法来学习解释性特征或输入变量的基本结构，就好像没有任何标记的训练样本一样。<br>
### 增强式学习
&emsp;&emsp;机器学习中的最后一种学习是增强式学习，在这种学习中没有主管，只有奖励信号。<br>
&emsp;&emsp;因此，增强式学习学习算法将尝试作出一个决策，然后一个奖励信号将在那里判断这个决定是正确的还是错误的。此外，这种监督反馈或奖励信号可能不会即时出现，但会延迟几步。例如，该算法现在将作出决定，但只有经过多个步骤后，奖励信号才能判断决策是好的还是坏的。<br>
## 数据和行业需求
&emsp;&emsp;数据是我们学习计算的信息库；如果没有信息，任何令人振奋和富有想象力的想法都是没有意义的。因此，如果你有一个有正确信息的信息科学应用程序，那么你就可以开始了。<br>
&emsp;&emsp;有能力调查和从你的信息中解脱出一种动机是显而易见的，尽管你的信息的结构，然而，由于巨大的信息正在成为当今时代的口号，所以我们需要信息科学的工具和进步，这些工具和技术可以在这样巨大的信息范围内进行扩展。一个明确无误的学习时间。如今，一切都在产生信息，具备适应信息的能力是一种考验。庞大的组织，例如谷歌、facebook、微软、ibm等，都会根据自己的适应性信息科学安排来制造自己的最终目标来应对巨大的问题。客户每天提供一次信息的数量。<br>
&emsp;&emsp;TensorFlow是一个机器智能/数据科学平台，于2016年11月9日由谷歌作为开源库发布。它是一个可伸缩的分析平台，使数据科学家能够在可见时间内用大量的数据构建复杂的系统，同时也使他们能够使用贪婪的学习方法。为了获得良好的性能，我们提供了大量的数据。<br>
## 总结
&emsp;&emsp;在本章中，我们建立了一个鱼类识别的学习系统；我们还看到了如何构建复杂的应用程序，比如鱼类识别，在TensorFlow和Keras的帮助下使用一些代码。这些编码示例并不是从您的角度来理解的，而是演示构建复杂系统的可见性，以及数据科学(尤其是深入学习)是如何使用的工具实现的。<br>
&emsp;&emsp;作为一名数据科学家，我们看到了在构建学习系统时在日常生活中可能遇到的挑战。<br>
&emsp;&emsp;我们还研究了构建学习系统的典型设计周期，并解释了该周期中涉及的每个组件的总体思想。<br>
&emsp;&emsp;最后，我们经历了不同的学习类型，每天都有大大小小的公司生成的各种数据，以及这些海量的数据是如何引发红色警报来构建可伸缩的工具的，从这些数据中分析和提取值。<br>
&emsp;&emsp;到目前为止，读者可能会对所有提到的信息感到不知所措，但我们在本章中所解释的大部分内容将在其他章节中讨论，包括数据科学挑战和鱼类识别的例子。本章的全部目的是对数据科学及其开发周期有一个全面的了解，而不需要对挑战和编码示例有任何深入的了解。本章中提到了编码示例，以消除数据科学领域大多数新手的恐惧，并向他们展示如何在几行代码中完成复杂的系统，如鱼类识别可以在一些代码中完成。<br>
&emsp;&emsp;接下来，我们将通过一个例子介绍数据科学的基本概念，开始我们的示例之旅。接下来的部分主要是通过介绍著名的泰坦尼克号的例子，为以后的高级章节做好准备。我们将讨论很多概念，包括不同的回归和分类学习方法，不同类型的性能错误，哪些是最需要关注的，以及更多关于处理一些数据科学挑战和处理不同形式的数据样本的信息。<br>

学号|姓名|专业
-|-|-
201802110484|杨杰|计算机应用技术
