# 第二章 行动中的数据建模--泰坦尼克号
&emsp;&emsp;线性模型是数据科学领域的基本学习算法。了解线性模型是如何工作的，在学习数据科学的过程中至关重要，因为它是基本的构建阻止了大多数复杂的学习算法，包括神经网络。<br>
&emsp;&emsp;在这一章中，我们将深入研究数据科学领域的一个著名问题，这就是泰坦尼克号的例子。这个例子的目的是引入线性模型进行分类，从数据处理和探索到模型评估，看到一个完整的机器学习系统流水线。我们将在本章中讨论以下主题：<br>
&emsp;&emsp;&emsp;&emsp;1.线性回归模型<br>
&emsp;&emsp;&emsp;&emsp;2.线性分类模型<br>
&emsp;&emsp;&emsp;&emsp;3.泰坦尼克号模型的建立和训练<br>
&emsp;&emsp;&emsp;&emsp;4.不同类型的错误<br>
## 线性回归模型
&emsp;&emsp;线性回归模型是最基本的回归模型，在预测数据分析中得到了广泛的应用。回归模型的总体思想是检查两件事：<br>
&emsp;&emsp;&emsp;&emsp;1.一组解释性特性/输入变量在预测输出变量方面做得很好吗？是使用考虑到因变量更改的可变性的特性的模型(输出变量)？<br>
&emsp;&emsp;&emsp;&emsp;2.哪些特征是因变量的重要特征？它们以何种方式影响因变量(由参数的大小和符号表示)？这些回归参数用于解释一个输出变量(因变量)和一个或多个输入特性(自变量)之间的关系。<br>
&emsp;&emsp;回归方程表示输入变量(自变量)对输出变量(因变量)的影响。这个方程的最简单形式，有一个输入变量和一个输出变量，由这个公式y=c+b*x定义。在这里，y=估计的相依分数，c=常数，b=回归参数/系数，和x=输入(独立)变量。<br>
### 原因
&emsp;&emsp;线性回归模型是许多学习算法的基石，但这并不是它们流行的唯一原因。以下是他们受欢迎的关键因素：<br>
&emsp;&emsp;&emsp;&emsp;广泛应用：线性回归技术是最古老的回归技术，在预测、财务分析等领域有着广泛的应用。<br>
&emsp;&emsp;&emsp;&emsp;运行速度快：线性回归算法非常简单，不包括太昂贵的数学计算。<br>
&emsp;&emsp;&emsp;&emsp;易于使用(不需要大量调优)：线性回归非常容易使用，而且大多数情况下，它是机器学	习或数据科学课程中学习的第一种学习方法，因为您不知道为了获得更好的性能，需要调整过多的超参数。<br>
&emsp;&emsp;&emsp;&emsp;高度可解释性：由于其简单易测各预测系数对的贡献，线性回归具有很高的可解释性；	您可以轻松地理解模型行为，并为非技术人员解释模型输出。如果系数为零，则相关的预测变量没有任何贡献。如果系数是而不是零，由于特定的预测变量的贡献可以很容易地确定。<br>
&emsp;&emsp;&emsp;&emsp;许多其他方法的基础：线性回归被认为是许多学习方法的基础，例如神经网络及其成长部分，深度学习。<br>
### 广告-一个金融例子
&emsp;&emsp;为了更好地理解线性回归模型，我们将通过一个实例广告。我们将尝试预测一些公司的销售额，考虑到一些与销售数量有关的因素。这些公司花在电视、广播和报纸上的广告。<br>
#### 依赖
&emsp;&emsp;为了用线性回归来建模我们的广告数据样本，我们将使用Stats模型库来获得线性模型的良好特性，但随后，我们将使用Scikit-Learning，它对于数据科学具有非常有用的功能。<br>
#### 向pandas输入数据
&emsp;&emsp;Python中有很多库可以用来读取、转换或写入数据。其中一个库是pandas(http://pandas.pydata.org/ )。pandas是一个开放源码库，具有很好的数据分析功能和工具以及非常容易使用的数据结构。<br>
&emsp;&emsp;你可以通过很多不同的方式很容易地得到pandas。要想得到pandas，最好的办法就是安装它。via conda(http://pandas.pydata.org/pandas–docs/stable/install.html#installing–pandas–with–anaconda)。<br>
&emsp;&emsp;&emsp;&emsp;“Conda是一个开放源码的软件包管理系统和环境管理系统，用于安装多个版本的软件包及其依赖项，并容易在它们之间切换。它在Linux、OSX和Windows上工作，是为Python程序创建的，但可以打包和分发任何软件。”-Conda网站。<br>
&emsp;&emsp;&emsp;&emsp;您可以通过安装Anaconda轻松获得Conda，这是一个开放的数据科学平台。<br>
&emsp;&emsp;那么，让我们来看看如何使用pandas来阅读广告数据样本。首先，我们需要导入pandas：<br>
```python
import pandas as pd
```  
&emsp;&emsp;接下来，我们可以使用panas.read_csv方法将我们的数据加载到一个名为DataFrame的易于使用的pandas数据结构中。要获得有关panas.read_csv及其参数的更多信息，请执行以下操作可参阅此方法的熊猫文档(https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html ):<br>
```python
# read advertising data samples into a DataFrame 
advertising_data =
pd.read_csv('http://www–bcf.usc.edu/~gareth/ISL/Advertising.csv',
index_col=0)
```  
&emsp;&emsp;传递给panas.read_csv方法的第一个参数是一个字符串值，表示文件路径。字符串可以是包含http、ftp、s3和文件的URL。传递的第二个参数是将用作数据行的标签/名称的列的索引。<br>
&emsp;&emsp;现在，我们有了DataDataFrame，它包含URL中提供的广告数据，每一行都被第一列标记。如前所述，熊猫提供了易于使用的数据结构，可以用作数据的容器。这些数据结构有一些与它们相关联的方法，您将使用这些方法来转换或操作您的数据。<br>
&emsp;&emsp;现在，让我们看一下广告数据的前五行：<br>
```python
# DataFrame.head 方法显示数据的前n行，n默认值为5     
advertising_data.head()
```
Output:
TV|radio|newspaper|sales
-|-|-
1|230.1|37.8|69.2|22.1
2|44.5|39.3|45.1|10.4
3|17.2|45.9|69.3|9.3
4|151.5|41.3|58.5|18.5
5|180.8|10.8|58.4|12.9
<br>


