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
# 将广告数据示例读入DataFrame
advertising_data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',
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
####

.|TV|Radio|Newspaper|Sales
-|-|-|-|-
1|230.1|37.8|69.2|22.1
2|44.5|39.3|45.1|10.4
3|17.2|45.9|69.3|9.3
4|151.5|41.3|58.5|18.5
5|180.8|10.8|58.4|12.9
<br>
<br>

#### 了解广告数据
&emsp;&emsp;这个问题属于监督学习类型，其中我们有解释特征(输入变量)和响应(输出变量)。<br>
&emsp;&emsp;&emsp;&emsp;特性/输入变量是什么？<br>
&emsp;&emsp;&emsp;&emsp;电视：在给定的市场上为单一产品在电视上花费的广告费用(以千美元计)<br>
&emsp;&emsp;&emsp;&emsp;收音机：用在收音机上的广告钱<br>
&emsp;&emsp;&emsp;&emsp;报纸：花在报纸上的广告钱<br>
&emsp;&emsp;响应/结果/输出变量是什么？<br>
&emsp;&emsp;&emsp;&emsp;销售：单个产品在给定市场上的销售(以千件为单位)<br>
&emsp;&emsp;我们还可以使用DataFrame方法形状来了解数据中的样本/观测数：<br>
```python
#sales 以千为单位
advertising_data.shape
```
Output: <br>
&emsp;&emsp;(20O, 4)<br>
&emsp;&emsp;因此，在广告数据中有200个观察结果。<br>
#### 数据分析和可视化
&emsp;&emsp;为了了解数据的底层形式，特征和响应之间的关系，以及更多的洞察力，我们可以使用不同类型的可视化。理解这种关系IP之间的广告数据特征和响应，我们将使用分散图。<br>
&emsp;&emsp;为了对数据进行不同类型的可视化，可以使用Matplotlib(https://matplotlib.org/ )，这是一个用于可视化的Python2D库。要获得Matplotlib，您可以将他们的安装说明放在：https://matplotlib.org/User/installing.html 上。<br>
&emsp;&emsp;让我们导入可视化库Matplotlib：<br>
```python
#为了对数据进行不同类型的可视化，使用Matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
```
&emsp;&emsp;现在，让我们使用一个散点图来可视化广告数据特性与响应变量之间的关系：<br>
```python
#使用一个散点图来可视化广告数据特性与响应变量之间的关系
fig, axs = plt.subplots(1, 3, sharey=True)
#将散点图添加到网格中
advertising_data.plot(kind='scatter', x='TV', y='sales', ax=axs[0], figsize=(16, 8))
advertising_data.plot(kind='scatter', x='radio', y='sales', ax=axs[1]) 
advertising_data.plot(kind='scatter', x='newspaper', y='sales', ax=axs[2])
```
Output:<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片1.png) <br>
图1. 了解广告数据特征与响应变量之间关系的散点图<br><br>
&emsp;&emsp;现在，我们需要看看广告将如何帮助提高销售。所以，我们需要问自己几个问题。值得问的问题就像广告和销售，哪种广告对销售贡献更大，以及每种类型的广告对销售的大致影响。我们将尝试用一个简单的线性模型来回答这样的问题。<br>
#### 简单回归模型
&emsp;&emsp;线性回归模型是一种学习算法，它使用解释性特征(或输入或predic)的组合来预测定量(也称为数值)响应
&emsp;&emsp;只有一个特征的简单线性回归模型采用以下形式：&emsp;&emsp;y = beta0 + beta1*x<br>
&emsp;&emsp;在这里:<br>
&emsp;&emsp;&emsp;&emsp;1.	y是预测的数值(响应)<sales<br>
&emsp;&emsp;&emsp;&emsp;2.	X是特征值<br>
&emsp;&emsp;&emsp;&emsp;3.	β0被称为拦截<br>
&emsp;&emsp;&emsp;&emsp;4.	β1是特征x<tv广告的系数。<br>
&emsp;&emsp;Beta 0和Beta 1都被认为是模型系数。为了在广告示例中建立一个能够预测销售价值的模型，我们需要学习这些系数，因为beta 1。将是特征x对响应y的学习效果。例如，如果beta 1=0.04，这意味着在电视广告上额外花费100美元与四个小部件的销售增长有关。因此，我们需要继续研究，看看我们如何才能了解这些系数。<br>
##### 模型学习
&emsp;&emsp;&emsp;&emsp;为了估计模型的系数，我们需要用一条与实际销售相似的回归线来拟合数据。为了得到一条最适合数据的回归线，我们将使用一个叫做最小二乘的标准。因此，我们需要找到一条将预测值和观察值(实际值)之间的差异最小化的线。换句话说，我们需要找到一条使平方和最小化的回归线。图2说明了这一点：
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片2.png) <br>
图2 “用一条回归线拟合数据点(电视广告样本)，使残差的平方最小(预测值和观测值之间的差值)” <br> <br>
&emsp;&emsp;以下是图2中存在的元素：<br>
&emsp;&emsp;&emsp;&emsp;黑点表示x(电视广告)和y(销售)的实际值或观察值。<br>
&emsp;&emsp;&emsp;&emsp;蓝线表示最小二乘线(回归线)。<br>
&emsp;&emsp;&emsp;&emsp;红线表示残差，这是预测值和观察值(实际值)之间的差异。<br>
&emsp;&emsp;因此，这就是我们的系数与最小二乘线(回归线)的关系：<br>
&emsp;&emsp;&emsp;&emsp;Beta 0是截距，它是x=0时y的值。<br>
&emsp;&emsp;&emsp;&emsp;β1是斜率，它表示y的变化除以x的变化。<br>
图3给出了这方面的图形解释：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片3.png) <br>
图3 “最小二乘线与模型系数的关系” <br> <br>
现在，让我们开始使用StatsModel学习这些系数：
```python
#在一行代码中创建一个合适的模型(表示最小二乘线)
lm = smf.ols(formula='sales ~ TV', data=advertising_data).fit()
# 显示训练后的模型系数
lm.params
```
Output：<br>
&emsp;&emsp;Intercept 7.O32594 &emsp;&emsp;  TV O.O47537 <br>
&emsp;&emsp;dtype: float64 <br>
正如我们所提到的，线性回归模型的优点之一是它们易于解释，所以让我们继续解释这个模型。<br>
#### 模型解释
&emsp;&emsp;让我们看看如何解释模型的系数，例如TV广告系数(Beta 1)：<br>
&emsp;&emsp;&emsp;&emsp;输入/功能(TV广告)支出的单位增加与销售(响应)中的O.O 47537单位增长相关联。换句话说，在电视广告上额外花费100美元增加销售4.7537件。<br>
&emsp;&emsp;从电视广告数据中建立一个学习模型的目的是预测看不见的数据的销售情况。那么，让我们看看如何使用所学的模型来预测销售价值(但我们没有基于电视广告的给定价值。)<br>
#### 使用预测模型
&emsp;&emsp;假设我们有看不见的电视广告支出数据，我们想知道它们对公司销售的相应影响。因此，我们需要使用所学的模型来为我们做到这一点。假设我们想从50000美元的电视广告知道销售额。<br>
&emsp;&emsp;让我们使用我们所学的模型系数来进行这样的计算：&emsp;&emsp;y = 7.032594 + 0.047537 x 50<br>
```python
7.032594 + 0.047537*50000
```
Output:<br>
&emsp;&emsp;9,4O9.444<br>
&emsp;&emsp;我们也可以使用Statsmodel为我们做预测。首先，我们需要在熊猫DataFrame中提供电视广告值，因为StatsModel界面期望它：<br>
```python
7.032594 + 0.047537*50000
#创建Pandas DataFrame以匹配StatsModel接口期望
new_TVAdSpending = pd.DataFrame({'TV':[50000]})
new_TVAdSpending.head()
```
Output:<br>

.|TV
-|-
0|50000
<br>

&emsp;&emsp;现在，我们可以继续使用预测函数来预测销售价值：<br>
```python
#使用模型对新值进行预测。
preds = lm.predict(new_TVAdSpending)
```
Output:<br>
array([ 9.4O942557])<br>
&emsp;&emsp;让我们看看所学的最小二乘线是什么样子的。为了画出这条线，我们需要两个点，每个点用这对表示：(x，x预测值)。<br>
&emsp;&emsp;&emsp;&emsp;那么，让我们取电视广告功能的最小值和最大值：<br>
Output:

.|TV
-|-
0|0.7
1|296.4
<br>

&emsp;&emsp;让我们为这两个值得到相应的预测：<br>
```python
# x最小值和最大值的预测
predictions = lm.predict(X_min_max) 
predictions
```
Output:<br>
&emsp;&emsp;array([7.O658692,2l.l2245377])<br>



