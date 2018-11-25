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
&emsp;&emsp;(200, 4)<br>
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
&emsp;&emsp;Intercept 7.032594 &emsp;&emsp;  TV 0.047537 <br>
&emsp;&emsp;dtype: float64 <br>
正如我们所提到的，线性回归模型的优点之一是它们易于解释，所以让我们继续解释这个模型。<br>
#### 模型解释
&emsp;&emsp;让我们看看如何解释模型的系数，例如TV广告系数(Beta 1)：<br>
&emsp;&emsp;&emsp;&emsp;输入/功能(TV广告)支出的单位增加与销售(响应)中的0.047537单位增长相关联。换句话说，在电视广告上额外花费100美元增加销售4.7537件。<br>
&emsp;&emsp;从电视广告数据中建立一个学习模型的目的是预测看不见的数据的销售情况。那么，让我们看看如何使用所学的模型来预测销售价值(但我们没有基于电视广告的给定价值。)<br>
#### 使用预测模型
&emsp;&emsp;假设我们有看不见的电视广告支出数据，我们想知道它们对公司销售的相应影响。因此，我们需要使用所学的模型来为我们做到这一点。假设我们想从50000美元的电视广告知道销售额。<br>
让我们使用我们所学的模型系数来进行这样的计算：&emsp;&emsp;y = 7.032594 + 0.047537 x 50<br>
```python
7.032594 + 0.047537*50000
```
Output:<br>
&emsp;&emsp;9,409.444<br>
我们也可以使用Statsmodel为我们做预测。首先，我们需要在熊猫DataFrame中提供电视广告值，因为StatsModel界面期望它：<br>
```python
#创建Pandas DataFrame以匹配StatsModel接口期望
new_TVAdSpending = pd.DataFrame({'TV':[50000]})
new_TVAdSpending.head()
```
Output:<br>

.|TV
-|-
0|50000
<br>

现在，我们可以继续使用预测函数来预测销售价值：<br>
```python
#使用模型对新值进行预测。
preds = lm.predict(new_TVAdSpending)
```
Output:<br>
array([ 9.40942557])<br><br>
让我们看看所学的最小二乘线是什么样子的。为了画出这条线，我们需要两个点，每个点用这对表示：(x，x预测值)。<br>
&emsp;&emsp;那么，让我们取电视广告功能的最小值和最大值：<br>
```python
# x最小值和最大值的预测
predictions = lm.predict(X_min_max) 
predictions
```
Output:

.|TV
-|-
0|0.7
1|296.4
<br>

让我们为这两个值得到相应的预测：<br>
```python
# x最小值和最大值的预测
predictions = lm.predict(X_min_max) 
predictions
```
Output:<br>
&emsp;&emsp;array([7.0658692, 21.12245377])<br>
现在，让我们绘制实际数据，然后用最小二乘线拟合它：
```python
#绘制实际观测数据
advertising_data.plot(kind='scatter', x='TV', y='sales')
#绘制最小二乘线
plt.plot(new_TVAdSpending, preds, c='red', linewidth=2)
```
Output:<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片4.png) <br>
图4 “实际数据和最小二乘线的图”<br><br>
本例的扩展和进一步的解释将在下一章中解释。<br>
## 线性分类模型
&emsp;&emsp;在这一部分中，我们将进行Logistic回归，这是广泛使用的分类算法之一。<br>
&emsp;&emsp;什么是Logistic回归？Logistic回归的简单定义是，它是一种包含线性判别的分类算法。<br>
&emsp;&emsp;我们将在两个方面澄清这一定义：<br>
    &emsp;&emsp;&emsp;&emsp;1.与线性回归不同，Logistic回归不尝试估计/预测给定一组特征或输入变量的数值变量的值。相反，逻辑回归算法的输出是给定样本/观测属于特定类的概率。简单地说，让我们假设我们有一个二进制分类器阳离子问题。在这种类型的问题中，输出变量中只有两个类，例如，病或未病。因此，某个样本属于病类的概率是P0，而某个样本属于非病类的概率是P1=1-P0。因此，Logistic回归算法的输出总是在0到1之间。<br>
    &emsp;&emsp;&emsp;&emsp; 2.您可能知道，有很多学习算法用于回归或分类，每个学习算法对数据样本都有自己的假设。选择适合您的数据的学习算法的能力将随着实践和对该主题的良好理解而逐渐出现。因此，逻辑回归算法的中心假设是，我们的输入/特征空间可以被线性曲面分割成两个区域(每类一个)，如果我们只有两个特征或一个平面，如果我们有三个，以此类推。这个边界的位置和方向将由你的数据决定。如果您的数据满足此约束，则将它们分割为与具有线性SUR的每个类相对应的区域。面对现实，你的数据被说成是线性可分的。下图说明了这一假设。在图5中，我们有三个维度、输入或特性以及两个可能的类：病(红色)和非病(蓝色)。区分这两个区域的地方称为线性判别，因为它是线性的，它有助于区分属于di的样本。不同分类：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片5.png) <br>
&emsp;&emsp;图5  “分离两类的线性决策曲面”<br><br>
&emsp;&emsp;如果您的数据样本不是线性可分的，您可以通过将数据转换为高维空间，添加更多的特性来实现它们。<br>
### 分类与Logistic回归
&emsp;&emsp;在上一节中，我们学习了如何将连续数量(例如，电视广告对公司销售的影响)预测为输入值的线性函数(例如，TV、收音机和广告)。但对于其他任务，产量将不会是连续的数量。例如，预测某人是否患病是一个分类问题，我们需要一种不同的学习算法表演这个。在这一部分中，我们将更深入地研究逻辑回归的数学分析，这是一种分类任务的学习算法。<br>
&emsp;&emsp;在线性回归中，我们尝试用线性模型函数y=h(X)=8Tx来预测该数据集中的ith样本x(I)的输出变量y(I)的值。这不是一个很好的解决方案n用于预测二进制标签(y(I)z{0，1})等分类任务。<br>
&emsp;&emsp;Logistic回归是我们可以用于分类任务的许多学习算法之一，我们使用一个不同的假设类，同时试图预测一个特定的假设类的概率。样本属于一类，概率属于零类。因此，在Logistic回归中，我们将尝试学习以下功能：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片6.png) <br>
&emsp;&emsp;该函数![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图.jpg) 通常称为Sigmoid或Logistic函数，它将8Tx的值压缩为固定范围[0，1]，如下图所示。因为这个值会在[0，1]，然后，我们可以将h8(X)解释为概率。<br>
&emsp;&emsp;我们的目标是搜索参数8的值，以便当输入样本x属于某一类时，概率P(y=1 x-x)=h8(X)很大，当x属于零类时，概率很小：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片7.png) <br>
图6 sigmoid函数<br><br>
&emsp;&emsp;因此，假设我们有一组训练样本，其中包含相应的二进制标签{(x(I)，y(I)：i=1，.，m}。我们需要最小化以下成本函数，该函数度量给定的h8的性能：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片8.png) <br>
&emsp;&emsp;注意，对于每个训练样本，我们只有一个等式的求和为非零(取决于标签y(I)的值是0还是)。当y(I)=1时，最小化模型成本函数意味着我们需要使h8(X)变大，并且当y=0时，![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图1.jpg)我们想把1-H8做大一点。<br>
&emsp;&emsp;现在，我们有一个成本函数来计算给定的假设H8是否适合我们的训练样本。我们可以学习使用优化技术对训练样本进行分类，使J(8)最小化，并找到参数8的最佳选择。一旦我们这样做了，我们就可以使用这些参数将一个新的测试样本分类为1或0，检查这两个类标签中哪一个是最有可能的。如果P(y=1|x)<P(y=0|x)，则输出0，否则输出1，这等于在类之间定义0.5的阈值，并检查h8(X)是否>0.5。<br>
&emsp;&emsp;为了使成本函数J(8)最小化，我们可以使用一种优化技术，找到最优值为8，从而使成本函数最小化。因此，我们可以使用一个名为梯度的微积分工具，它试图找出成本函数的最大增长率。然后，我们可以向相反的方向求出这个函数的最小值；例如，梯度。用D8J(8)来表示J(8)，这意味着取成本函数的梯度关于模型参数。因此，我们需要提供一个计算J(8)和D8J(8)要求的任何选择8。如果我们导出了J(8)以上的成本函数对于8j，我们将得到以下结果：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片9.png) <br>
它可以用矢量形式写成：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片10.png) <br>
&emsp;&emsp;现在，我们已经对Logistic回归有了一个数学理解，所以让我们继续使用这种新的学习方法来解决分类任务。<br>
## 泰坦尼克号模型的建立和训练
&emsp;&emsp;泰坦尼克号的沉没是历史上最臭名昭著的事件之一。这一事件导致2224名乘客和机组人员中的1502人死亡。在这个问题上，我们将使用数据科学预测乘客是否会在这场悲剧中幸存下来，然后根据实际的悲剧统计数据检验我们的模型的性能。<br>
&emsp;&emsp;为了跟进泰坦尼克号的例子，您需要执行以下操作：<br>
&emsp;&emsp;&emsp;&emsp;1.	Download this repository in a ZIP file by clicking on https://github.com/ahmed–menshawy/ML_Titanic/archive/master.zip or execute from the terminal:<br>
&emsp;&emsp;&emsp;&emsp;2.	Git clone: https://github.com/ahmed–menshawy/MLTitanic.git<br>
&emsp;&emsp;&emsp;&emsp;3.	Install [virtualenv]: (http://virtualenv.readthedocs.org/en/latest/installation.html)<br>
&emsp;&emsp;&emsp;&emsp;4.	Navigate to the directory where you unzipped or cloned the repo and create a virtual environment with virtualenv ml_titanic<br>
&emsp;&emsp;&emsp;&emsp;5.	Activate the environment with source ml_titanic/bin/activate<br>
&emsp;&emsp;&emsp;&emsp;6.	Install the required dependencies with pip install –r requirements.txt<br>
&emsp;&emsp;&emsp;&emsp;7.	Execute the ipython notebook from the command line or terminal<br>
&emsp;&emsp;&emsp;&emsp;8.	Follow the example code in the chapter<br>
&emsp;&emsp;&emsp;&emsp;9.	When you're done, deactivate the virtual environment with deactivate<br>
### 数据处理和可视化
&emsp;&emsp;在本节中，我们将做一些数据预处理和分析。数据的探索和分析被认为是应用机器学习的最重要的步骤之一，也可能是被认为是最重要的一步，因为在这一步中，你会认识到朋友，数据，在训练过程中它会和你在一起。此外，了解数据将使您能够向下移动一组候选算法，以检查哪一种算法最适合您的数据。<br>
&emsp;&emsp;让我们从导入实现所需的包开始：<br>
```python
import matplotlib.pyplot as plt
%matplotlib inline
from statsmodels.nonparametric.kde import KDEUnivariate 
from statsmodels.nonparametric import smoothers_lowess 
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm

import numpy as np 
import pandas as pd
import statsmodels.api as sm

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
```
让我们用Pandas来阅读泰坦尼克号乘客和船员的数据：<br>
```python
#读取数据
titanic_data = pd.read_csv("data/train.csv")
titanic_data.shape
```
Output: <br>
&emsp;&emsp;&emsp;&emsp;(891,12)<br>
因此，我们总共有891个观察、数据样本或乘客/机组人员记录，以及描述这一记录的12个解释性特征：<br>
```python
list(titanic_data)
```
Output: 
['PassengerId',<br>
 'Survived',<br>
 'Pclass',<br>
 'Name',<br>
 'Sex',<br>
 'Age',<br>
 'SibSp',<br>
 'Parch',<br>
 'Ticket',<br>
 'Fare',<br>
 'Cabin',<br>
 'Embarked']<br>
 让我们看看一些样本/观测的数据：<br>
 ```python
 titanic_data[500:510]
```
Output: <br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片11.png) <br>
图7 “泰坦尼克号数据集的样本”<br><br>
&emsp;&emsp;现在，我们有了一个Pandas DataFrame，它包含了我们需要分析的891名乘客的信息。DataFrame的列表示关于每个乘客/机组人员的解释性特性，如姓名、性别或年龄。<br>
&emsp;&emsp;其中一些解释功能是完整的，没有任何缺失的值，例如存活的特性，它有891个条目。其他解释功能包含缺失的值，例如年龄特性，它只有714个条目。DataFrame中的任何缺失值都表示为NaN。<br>
&emsp;&emsp;如果您研究了所有的数据集特性，您会发现票务和客舱功能有许多缺失的值(NAN)，因此它们不会为我们的分析增加太多的价值。为了处理这件事，我们会将它们从DataFrame中删除。<br>
&emsp;&emsp;使用以下代码行可以完全从DataFrame中删除票证和机舱功能：<br>
 ```python
#完全从DataFrame中删除票证和机舱功能#完全从Dat 
titanic_data = titanic_data.drop(['Ticket','Cabin'], axis=1)
```
&emsp;&emsp;在我们的数据集中存在这样的缺失值是有很多原因的。但是为了保持数据集的完整性，我们需要处理这些缺失的值。在这个具体的问题上，我们会选择放弃。<br>
使用以下代码行可以从所有剩余功能中删除所有NaN值：<br>
 ```python
#从所有剩余功能中删除所有NaN值
titanic_data = titanic_data.dropna()
```
&emsp;&emsp;现在，我们有了一种竞争数据集，我们可以用它来进行分析。如果你决定先删除所有的nans而不删除票证和客舱功能，你会发现DataSet被删除，因为.Drona()方法从DataFrame中删除了一个观察，即使它在其中一个特性中只有一个NaN。<br>
&emsp;&emsp;让我们进行一些数据可视化，以查看某些特性的分布情况，并了解解释性特性之间的关系：<br>
 ```python
# 图参数声明
fig = plt.figure(figsize=(18,6)) 
alpha=alpha_scatterplot = 0.3 
alpha_bar_chart = 0.55
# Defining a grid of subplots to contain all the figures 
axl = plt.subplot2grid((2,3),(0,0))
# 添加第一个条形图，表示幸存的人和没有幸存的人的数量。
titanic_data.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# 添加边距
axl.set_xlim(-1, 2)
# 添加图标题
plt.title("Distribution of Survival, (l = Survived)") 
plt.subplot2grid((2,3),(0,1)) 
plt.scatter(titanic_data.Survived, titanic_data.Age, alpha=alpha_scatterplot)
# 设置y标签的值(年龄)
plt.ylabel("Age")
# 格式化
plt.grid(b=True, which='major', axis='y') 
plt.title("Survival by Age, (l = Survived)") 
ax3 = plt.subplot2grid((2,3),(0,2))
titanic_data.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart) 
ax3.set_ylim(-1, len(titanic_data.Pclass.value_counts()))
plt.title("Class Distribution") 
plt.subplot2grid((2,3),(1,0), colspan=2)
# plotting kernel density estimate of the subse of the lst class passenger’s age
titanic_data.Age[titanic_data.Pclass == 1].plot(kind='kde') 
titanic_data.Age[titanic_data.Pclass == 2].plot(kind='kde') 
titanic_data.Age[titanic_data.Pclass == 3].plot(kind='kde')
# 将x标记(年龄)添加到绘图中
plt.xlabel("Age")
plt.title("Age Distribution within classes")
# Add legend to the plot.
plt.legend(('lst Class', '2nd Class','3rd Class'),loc='best') 
ax5 = plt.subplot2grid((2,3),(1,2)) 
titanic_data.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax5.set_xlim(-1, len(titanic_data.Embarked.value_counts())) 
plt.title("Passengers per boarding location")
```
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片12.png) <br>
&emsp;&emsp;图 8 “泰坦尼克号数据样本的基本可视化”<br><br>
&emsp;&emsp;正如我们所提到的，这个分析的目的是根据现有的特征来预测某个特定的乘客是否会在悲剧中幸存下来，例如旅行舱(在数据中称为pclass)，性别，A。通用电气，还有票价。那么，让我们看看我们是否能更好地了解那些幸存和死亡的乘客。 <br>
&emsp;&emsp;首先，让我们绘制一个条形图来查看每个类中的观察数(存活/死亡)： <br>
 ```python
#绘制一个条形图来查看每个类中的观察数(存活/死亡)
plt.figure(figsize=(6,4)) 
fig, ax = plt.subplots()
titanic_data.Survived.value_counts().plot(kind='barh', color="blue", alpha=.65)
ax.set_ylim(-1, len(titanic_data.Survived.value_counts())) 
plt.title("Breakdown of survivals(O = Died, l = Survived)")
```
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片13.png) <br>
&emsp;&emsp;图9：生存分类 <br> <br>
&emsp;&emsp;让我们通过将先前的图表按性别细分，来更好地理解这些数据： <br>
 ```python
#将先前的图表按性别细分，来更好地理解这些数据
fig = plt.figure(figsize=(18,6))
#为幸存者绘制基于性别的分析。
male = titanic_data.Survived[titanic_data.Sex == 'male'].value_counts().sort_index()
female = titanic_data.Survived[titanic_data.Sex == 'female'].value_counts().sort_index()
axl = fig.add_subplot(121) 
male.plot(kind='barh',label='Male', alpha=0.55)
female.plot(kind='barh', color='#FA2379',label='Female', alpha=0.55) 
plt.title("Gender analysis of survivals (raw value counts) "); plt.legend(loc='best')
axl.set_ylim(-1, 2)
ax2 = fig.add_subplot(122) 
(male/float(male.sum())).plot(kind='barh',label='Male', alpha=0.55) 
(female/float(female.sum())).plot(kind='barh', color='#FA2379',label='Female', alpha=0.55)
plt.title("Gender analysis of survivals"); plt.legend(loc='best') 
ax2.set_ylim(-1, 2)
```
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片14.png) <br>
&emsp;&emsp;图 10：按性别分列的泰坦尼克号数据 <br> <br>
&emsp;&emsp;现在，我们有更多关于这两个可能的类(存活和死亡)的信息。探索和可视化步骤是必要的，因为它可以让您更深入地了解数据的结构帮助您选择适合您的问题的学习算法。正如您所看到的，我们从非常基本的绘图开始，然后增加了绘图的复杂性，以发现更多关于我们正在处理的数据的信息。 <br>
### 数据分析-监督机器学习
&emsp;&emsp;这项分析的目的是预测幸存者。因此，结果是否能够存活，这是一个二进制分类问题；在它中，只有两个可能的类。<br>
有很多学习算法可以用来解决二进制分类问题。Logistic回归就是其中之一。正如维基百科所解释的：<br>
&emsp;&emsp;In statistics, logistic regression or logit regression is a type of regression analysis used for predicting the outcome of a categorical dependent variable (a dependent variable that can take on a limited number of values, whose magnitudes are not meaningful but whose ordering of magnitudes may or may not be meaningful) based on one or more predictor variables. That is, it is used in estimating empirical values of the parameters in a qualitative response model. The probabilities describing the possible outcomes of a single trial are modeled, as a function of the explanatory (predictor) variables, using a logistic function. Frequently (and subsequently in this article) "logistic regression" is used to refer specifically to the problem in which the dependent variable is binary—that is, the number of available categories is two—and problems with more than two categories are referred to as multinomial logistic regression or, if the multiple categories are ordered, as ordered logistic regression. Logistic regression measures the relationship between a categorical dependent variable and one or more independent variables, which are usually (but not necessarily) continuous, by using probability scores as the predicted values of the dependent variable.[1] As such it treats the same set of problems as does probit regression using similar techniques.<br>
为了使用逻辑回归，我们需要创建一个公式，告诉模型我们要给它的特性/输入类型：<br>
 ```python
#为了使用logistic回归，我们需要创建一个公式，告诉模型我们要给它的特性/输入类型
# 模型公式
# ~表示=, 数据集的特征被写成预测生存的公式。C()让我们的回归知道这些变量是绝对的。
# Ref: http://patsy.readthedocs.org/en/latest/formulas.html
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + C(Embarked)'
# create a results dictionary to hold our regression results for easy analysis later
results = {}
# 使用 patsy's dmatrices 函数创建一个回归友好的数据框架 
y,x = dmatrices(formula, data=titanic_data, return_type='dataframe')
# 实例化模型
model = sm.Logit(y,x)
# 将我们的模型与培训数据相匹配
res = model.fit()
# 保存结果，以便稍后输出预测
results['Logit'] = [res, formula]
res.summary()
```
Output:<br>
Optimization terminated successfully.<br>
        &emsp;&emsp; Current function value: 0.444388<br>
        &emsp;&emsp; Iterations 6<br>
        ![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片15.png) <br>
&emsp;&emsp;&emsp;&emsp;图 11 “Logistic回归结果”<br><br>
现在，让我们绘制模型与实际模型的预测，以及残差，这是目标变量的实际值和预测值之间的差异：<br>
 ```python
#画出我们的模型和实际模型的预测以及残差(目标变量的实际值和预测值之间的差异)
plt.figure(figsize=(18,4)); 
plt.subplot(121, facecolor="#DBDBDB")
# 根据我们的拟合模型生成预测
ypred = res.predict(x)
plt.plot(x.index, ypred,'bo',x.index,y,'mo', alpha=.25); 
plt.grid(color='white', linestyle='dashed')
plt.title('Logit predictions, Blue: \nFitted/predicted values: Red');
# 剩余误差
ax2 = plt.subplot(122, facecolor="#DBDBDB") 
plt.plot(res.resid_dev, 'r-') 
plt.grid(color='white', linestyle='dashed') 
ax2.set_xlim(-1, len(res.resid_dev)) 
plt.title('Logit Residuals');
```
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片16.png) <br>
&emsp;&emsp;&emsp;&emsp;图12“理解Logistic回归模型”<br><br>
&emsp;&emsp;现在，我们已经建立了我们的Logistic回归模型，在此之前，我们对数据集进行了一些分析和探索。前面的示例向您展示了用于构建机器学习解决方案的一般管道。 <br>
&emsp;&emsp;大多数情况下，实践者陷入了一些技术陷阱，因为他们缺乏理解机器学习概念的经验。例如，某人可能在测试集上获得99%的准确性，然后不对数据中的类的分布进行任何调查(例如，有多少个样本是阴性的，以及如何)。许多样本是阳性的)，他们部署模型。 <br>
&emsp;&emsp;为了突出这些概念中的一些，并区分您需要注意的不同类型的错误以及您应该真正关心的错误，我们将继续讨论下一节。 <br>
## 不同类型的错误
&emsp;&emsp;在机器学习中，有两种类型的错误，作为数据科学的新手，您需要了解两者之间的关键区别。如果你最终将错误的错误最小化，整个学习系统将是无用的，你将无法在实践中使用它来处理看不见的数据。为了尽量减少从业者对这两类错误的误解，我们将在以下两节中解释这些错误。 <br>
## 表观(训练集)误差
&emsp;&emsp;这是您不必关心的第一种错误。获得这类错误的小值并不意味着您的模型能够很好地处理未见数据(泛化)。为了更好地理解这类错误，我们将给出一个简单的类场景示例。在课堂上解决问题的目的不是在考试中再次解决同样的问题，而是能够解决不一定与你类似的其他问题。在教室里练习。考试问题可能来自同一个家庭的课堂问题，但不一定相同。 <br>
&emsp;&emsp;表观误差是训练模型在我们已经知道真实结果/输出的训练集上执行的能力。如果您设法在培训集上获得0错误，那么这是一个很好的指示，说明您的模型(大部分)在未见数据上不能很好地工作(不会泛化)。另一方面，数据科学是将训练集作为学习算法的基础知识，以便在未来的未见数据上很好地工作。 <br>
&emsp;&emsp;在图13中，红色曲线表示明显的误差。每当您增加模型记忆事物的能力(例如通过增加解释特性的数量来增加模型的复杂性)，您会发现这种明显的错误方法是零。可以表明，如果你的特征和观测/样本一样多，那么表观误差将为零：<br>
![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter02/图片17.png) <br>
&emsp;&emsp;&emsp;&emsp;图13“表观误差(红色曲线)和泛化/真误差(浅蓝色)<br><br>
## 泛化/真误差
&emsp;&emsp;这是数据科学中第二种也是更重要的错误类型。建立学习系统的全部目的是在测试集上获得较小的泛化错误；换句话说，使该模型在一组未在训练阶段使用的观测/样本上运行良好。如果您仍然考虑上一节中的类场景，则可以将泛化错误视为解决不一定类似于您解决的问题的考试问题的能力。在课堂上学习和熟悉这门课。因此，泛化性能是指模型能够利用它在训练阶段学到的技能(参数)来正确地预测未知数据的结果/输出。<br>
&emsp;&emsp;在图13中，浅蓝线表示泛化错误。您可以看到，随着模型复杂度的增加，泛化误差将减少，直到模型开始失去其不断增长的功率和泛化误差时为止。曲线的这一部分，当你得到泛化误差而失去它不断增长的泛化能力时，被称为过拟合。<br>
&emsp;&emsp;从本节得到的信息是尽可能最小化泛化错误。<br>
## 总结
&emsp;&emsp;线性模型是一个非常强大的工具，如果数据与它的假设相匹配，您可以使用它作为初始学习算法。理解线性模型将有助于您理解使用线性模型作为构建块的更复杂的模型。<br>
&emsp;&emsp;接下来，我们将继续使用泰坦尼克号的例子，更详细地讨论模型的复杂性和评估。模型复杂性是一个非常强大的工具，您需要仔细使用它，以提高泛化误差导致过度适应的问题。<br>

学号|姓名|专业
-|-|-
201802110484|杨杰|计算机应用技术

