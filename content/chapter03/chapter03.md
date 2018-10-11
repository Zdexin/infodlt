
# 第三章
## 特征工程与模型复杂度 – 泰坦尼克号的例子重现

&emsp;&emsp;模型的复杂性和评估是建立一个成功的数据科学系统的必由之路。有很多工具可以用来评估和选择你的模型。在这一章中，我们将通过添加更多的描述性特性和从现有特性中提取有意义的信息来解决一些工具，这些工具可以帮助你增加数据的价值。我们也要解决其他工具相关的最优数量特性，并了解为什么有大量的特性和较少的培训样本/观察这个问题。<br>

&emsp;&emsp;以下是将在本章中解释的主题：<br>
&emsp;&emsp;&emsp;&emsp;--- 特征工程<br>
&emsp;&emsp;&emsp;&emsp;--- 维数灾难<br>
&emsp;&emsp;&emsp;&emsp;--- 泰坦尼克号例子的重现<br>
&emsp;&emsp;&emsp;&emsp;--- 偏置-方差分解<br>
&emsp;&emsp;&emsp;&emsp;--- 学习能见度<br>

### 特征工程
&emsp;&emsp;特征工程是影响模型性能的关键部件之一。是一个只要有正确的特性，就能比复杂的模型表现得更好的简单的模型。你可以将特征工程过程看作是决定你的预测模型是否成功的最重要的一步。如果您了解这些数据，理解特征工程就会容易得多。<br>

&emsp;&emsp;任何使用机器学习来解决一个问题的人都广泛使用特征工程，那就是：如何充分利用数据样本进行预测建模？这是特征工程的过程和实践所解决的问题，以及您的数据科学技能的成功，首先是知道如何很好地表示您的数据。<br>

&emsp;&emsp;预测建模是将一系列特性或输入变量(x1、x2、.、xn)转换为感兴趣的输出/目标(Y)的公式或规则。那么，什么是功能工程？这是一个过程，从现有的输入变量(x1，x2，.，xn)创建新的输入变量或特性(z1，z2，.，zn)。我们不只是创建任何新特性；新创建的特性应该对模型的输出做出贡献并与其相关。创建与模型输出相关的这些特性将是一个简单的过程，需要了解该领域(如市场营销、医疗等)。 即使机器学习实践者在此过程中与一些领域专家进行交互，特征工程过程的结果也会更好。<br>

&emsp;&emsp;领域知识可能有帮助的一个例子是，在给定一组输入变量/特征(温度、风速和云量百分比)的情况下，模拟降雨的可能性。对于这个特定的例子，我们可以构造一个名为“overcast”的新二进制特性，当云量百分比小于20%时，它的值等于1或0，否则等于0或yes。在这个例子中， 领域知识对于指定阈值或截止百分比至关重要。输入越周到、越有用，模型的可靠性和预测性就越好。<br>

### 特征工程的类型
&emsp;&emsp;特征工程作为一种技术，有三个主要的子范畴。作为一名深造的实践者，你可以自由地在两者之间做出选择，或者以某种方式将它们结合起来。<br>

#### 特征选择
&emsp;&emsp;有时称为特征重要性（feature importance），这是根据输入变量对目标/输出变量的贡献对其进行排序的过程。该过程还可以被认为是一个根据输入变量在模型预测能力中的值排序的过程。<br>
&emsp;&emsp;一些学习方法将这种特征排序或重要性作为其内部过程(如决策树)的一部分。大多数情况下，这类方法都是利用熵来筛选出价值较低的变量。在某些情况下，深度学习的实践者使用这样的学习方法来选择最重要的特征，然后将它们输入到更好的学习算法中。<br>
    
#### 降维
&emsp;&emsp;降维有时是特征提取，它是将现有的输入变量组合成一组新的大量减少的输入变量的过程。<br>       
&emsp;&emsp;在这类特征工程中，最常用的方法之一是主成分分析(PCA)，它利用数据的差异来减少原始输入变量的数量。<br>

#### 特征构建
&emsp;&emsp;特征构建是一种常用的特征工程，人们在谈论特征工程时通常会用到它。这种技术是从原始数据中手工制作或构造新特性的过程。与其他特性工程技术一样，特征构建的目的是提高模型的可预测性。<br>

&emsp;&emsp;一个简单的特征构造示例是使用日期标记特性生成两个新特性，如AM和PM，这可能有助于区分白天和黑夜。我们也可以通过计算噪声特征的平均值，然后确定给定行是否大于或小于该平均值，将有噪声的数字特征转换为简单的标称值。<br>
   
### 泰坦尼克号的例子
&emsp;&emsp;在本节中，我们将再次介绍泰坦尼克号的例子，但是在使用特征工程工具时，我们将从不同的角度看问题。如果您跳过第2章，行动中的数据建模-泰坦尼克号示例，泰坦尼克号的例子是一场Kaggle竞赛，目的是预测某位乘客是否幸存下来。<br>
&emsp;&emsp;在“泰坦尼克号”这个例子的重温中，我们将使用 scikit-learn和pandas。因此，首先，让我们从阅读train和test开始，得到一些统计数据:<br>
                                     
```# reading the train and test sets using pandas train_data = pd.read_csv('data/train.csv', header=O) test_data =        pd.read_csv('data/test.csv', header=O)

   # concatenate the train and test set together for doing the overall feature engineering stuff
   df_titanic_data = pd.concat([train_data, test_data])
 
   # removing duplicate indices due to coming the train and test set by re– indexing the data
   df_titanic_data.reset_index(inplace=True)

   # removing the index column the reset_index() function generates df_titanic_data.drop('index', axis=l, inplace=True)

   # index the columns to be l–based index
   df_titanic_data = df_titanic_data.reindex_axis(train_data.columns, axis=l)
```
    
&emsp;&emsp;我们需要指出前面的一些代码片段的问题:<br>
&emsp;&emsp;正如所展示的，我们使用了pandas的CONAT功能来组合train和test的数据帧。这对于特征工程任务非常有用，因为我们需要全面查看输入变量/特性的分布情况。<br>
&emsp;&emsp;在合并了两个数据帧之后，我们需要对输出数据帧进行一些修改。<br>

#### 缺失值
&emsp;&emsp;在从客户那里获得新的数据集之后，缺失值是第一件要考虑的事情，因为几乎每个数据集中都会有丢失/不正确的数据。有些学习算法能够自行处理丢失的值，而另一些则需要我们手动处理丢失的数据。在这个例子中，我们将使用来自Scikit-Learning的随机森林分类器，这需要单独处理丢失的数据，可以使用不同的方法来处理。<br>

##### 删除带有缺失值的示例
&emsp;&emsp;如果是一个包含大量缺失值的小数据集，这种方法将不是一个很好的选择，因为删除缺失值的样本将产生无用的数据。如是有大量的数据，并且删除它不会对原始数据集产生太大的影响，这可能是一个快速而容易的选择。<br>

##### 输入缺失值
&emsp;&emsp;当您有分类数据时，此方法非常有用。这种方法背后的直觉是，缺少的值可能与其他变量相关，而删除它们将导致信息的丢失。 这会对模型产生显著的影响。<br>

&emsp;&emsp;例如，如果我们有一个具有两个可能值-1和1的二进制变量，我们可以添加另一个值(0)来表示缺少的值。可以使用以下代码替换 带有U0的机舱功能：<br>
 ```# replacing the missing value in cabin variable "U0" 
    df_titanic_data['Cabin'][df_titanic_data.Cabin.isnull()] = 'U0'
 ```

##### 分配平均值
&emsp;&emsp;这也是一种常见的方法，因为它的简单性。在数值特性的情况下，您只需用平均值或中值替换丢失的值。您也可以使用此方法。 在分类变量的情况下，通过将模式(发生次数最多的值)分配给缺失的值。<br>
&emsp;&emsp;下面的代码将fare功能的不丢失值的中位数分配给缺失的值:<br>
 ```# handling the missing values by replacing it with the median fare 
    df_titanic_data['Fare'][np.isnan(df_titanic_data['Fare'])] = df_titanic_data['Fare'].median()
 ```

##### 使用回归或其他简单模型预测缺失变量的值
&emsp;&emsp;这是我们将使用的方法，在年龄特征的泰坦尼克号的例子。年龄特征是预测乘客生存和应用先前方法的重要一步。采取这种手段会使我们失去一些信息。<br>

&emsp;&emsp;为了预测丢失的值，您需要使用一种监督学习算法，该算法以可用的特征作为输入，以及要预测的特征的可用值。 它作为输出的缺失值。在下面的代码片段中，我们使用随机森林分类器来预测年龄特征的缺失值：<br>

```# Define a helper function that can use RandomForestClassifier for handling the missing values of the age variable
def set_missing_ages(): global df_titanic_data

age_data = df_titanic_data[['Age', 'Embarked', 'Fare', 'Parch', '3ib3p', 'Title_id', 'Pclass', 'Names', 'CabinLetter']]
input_values_RF = age_data.loc[(df_titanic_data.Age.notnull())].values[:, l::]
target_values_RF = age_data.loc[(df_titanic_data.Age.notnull())].values[:, O]

# Creating an object from the random forest regression function of sklearn<use the documentation for more details>
regressor = RandomForestRegressor(n_estimators=2OOO, n_jobs=–l)

# building the model based on the input values and target values above regressor.fit(input_values_RF, target_values_RF)

# using the trained model to predict the missing values 
predicted_ages = regressor.predict(age_data.loc[(df_titanic_data.Age.isnull())].values[:, l::])


# Filling the predicted ages in the original titanic dataframe 
age_data.loc[(age_data.Age.isnull()), 'Age'] = predicted_ages
```

#### 特征变换
&emsp;&emsp;在前两节中，我们讨论了阅读train和test集并将它们结合起来。我们还处理了一些丢失的值。现在，我们将使用scikit-learn的随机森林分类器学习去预测乘客的生存。随机林算法的不同实现接受不同类型的数据。scikit-learn实现的随机森林只接受数字数据。所以我们需要把分类特征转换成数字特征。<br>
&emsp;&emsp;这有两种特征类型:<br>
&emsp;&emsp;定量的: 定量特征是在一个数字尺度上测量的，可以有意义地分类。在泰坦尼克号数据样本中，年龄特征是定量特征的一个例子。<br>

&emsp;&emsp;定性的: 定性也称为范畴变量，不是数值变量。在泰坦尼克号的数据样本中，乘船（Embarke）特征是定性特征的一个例子。<br>

&emsp;&emsp;我们可以对不同的变量应用不同类型的转换。以下是一些可以用来转换定性/分类特征的方法。<br>

##### 虚拟特征
&emsp;&emsp;这些变量也称为分类或二进制特性。如果要转换的特性有少量不同的值，这种方法将是一个很好的选择。在泰坦尼克号上 一个样本，已启动的特性只有三个不同的值(3，C，和Q)，经常发生。因此，我们可以将已启动的特性转换为三个虚拟变量('Embarked_3', 'Embarked_C', and 'Embarked_Q'） 可以使用随机森林分类器。<br>
&emsp;&emsp;下面的代码将向您展示如何进行这种转换：<br>
```# constructing binary features 
def process_embarked():
    global df_titanic_data

# replacing the missing values with the most common value in the variable
df_titanic_data.Embarked[df.Embarked.isnull()] = df_titanic_data.Embarked.dropna().mode().values

# converting the values into numbers 
df_titanic_data['Embarked'] = pd.factorize(df_titanic_data['Embarked'])[O]

# binarizing the constructed features if 
keep_binary:
  df_titanic_data = pd.concat([df_titanic_data, pd.get_dummies(df_titanic_data['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=l)
  ```

##### 因式分解
&emsp;&emsp;该方法用于从任何其他特征创建数字分类特征。在pandas中，factorize()函数可以做到这一点。如果您的特性是字母数字范畴变量，则这种类型的转换非常有用。数字范畴变量在泰坦尼克号数据样本中，我们可以将Cabin特征转换为表示客舱字母的分类特征:<br>
```# the cabin number is a sequence of of alphanumerical digits, so we are going to create some features
# from the alphabetical part of it
df_titanic_data['CabinLetter'] = df_titanic_data['Cabin'].map(lambda l: get_cabin_letter(l))
df_titanic_data['CabinLetter'] = pd.factorize(df_titanic_data['CabinLetter'])[O]

def get_cabin_letter(cabin_value):
# searching for the letters in the cabin alphanumerical value 
letter_match = re.compile("([a–zA–Z]+)").search(cabin_value)

if letter_match:
    return letter_match.group() 
else:
    return 'U'
```
&emsp;&emsp;我们还可以使用以下方法之一将转换应用于数量特性。<br>

##### 缩放比例
&emsp;&emsp;这种变换只适用于数值特征。<br>

&emsp;&emsp;例如，在泰坦尼克号的数据中，年龄特征可以达到100，但家庭收入可能高达数百万。有些模型对值的大小很敏感，因此对这些特性进行缩放会有帮助。 软管模型表现更好。此外，缩放可以用来压缩变量的值，使其在特定的范围内。<br>
下面的代码将通过将年龄特征的平均值从每个值和标度中移除到单位方差来缩放它：<br>

```# scale by subtracting the mean from each value 
scaler_processing = preprocessing.3tandard3caler()
df_titanic_data['Age_scaled'] = scaler_processing.fit_transform(df_titanic_data['Age'])
```

##### 数据分箱技术
&emsp;&emsp;这种量化变换用于生成分位数。在这种情况下，定量特征值将是转换的有序变量。这种方法对于线性回归并不是一个很好的选择，适用于分类变量。
下面的代码将这种转换应用于票价特性：<br>
```# Binarizing the features by binning them into quantiles 
df_titanic_data['Fare_bin'] = pd.qcut(df_titanic_data['Fare'], 4)

if keep_binary:
df_titanic_data = pd.concat( [df_titanic_data,pd.get_dummies(df_titanic_data['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))],axis=l)
```

#### 导出特征
&emsp;&emsp;在上一节中，我们对泰坦尼克号数据进行了一些转换，以便能够使用Scikit-Learning的随机森林分类器(它只接受数值数据)。在这个部分 ，我们将定义另一种类型的变量，它是从一个或多个其他特性派生出来的。<br>

&emsp;&emsp;根据这个定义，我们可以说，上一节中的一些转换也称为派生特性。在本节中，我们将研究其他复杂的转换。<br>
&emsp;&emsp;在前面的部分中，我们提到您需要使用您的特征工程技能来获得新的特性，以增强模型的预测能力。我们还谈到了数据科学管道中的特征工程，以及为什么您应该花费大量的时间和精力来开发有用的特性。在本节中，领域知识将非常有用。<br>
&emsp;&emsp;派生功能的简单示例将类似于从电话号码中提取国家代码和/或区域代码。您还可以从GPS坐标中提取国家/区域。<br> 
泰坦尼克号的数据非常简单，不包含太多的变量，但是我们可以尝试从它中的文本特性中派生出一些特性。<br>

##### name（名称变量）
&emsp;&emsp;名称变量本身对于大多数数据集是无用的，但是它有两个有用的属性。第一个是你名字的长度。例如，您的名字的长度可能反映了y的一些内容。 我们的地位以及你登上救生艇的能力：<br>

```# getting the different names in the names variable 
df_titanic_data['Names'] = df_titanic_data['Name'].map(lambda y: len(re.split(' ', y)))
```
&emsp;&emsp;第二个有趣的属性是名称标题，它也可以用来表示身份和/或性别：<br>

```# Getting titles for each person
df_titanic_data['Title'] = df_titanic_data['Name'].map(lambda y: re.compile(", (.*?)\.").findall(y)[O])

# handling the low occurring titles 
df_titanic_data['Title'][df_titanic_data.Title == 'Jonkheer'] = 'Master' df_titanic_data['Title'][df_titanic_data.Title.isin(['Ms', 'Mlle'])] = 'Miss'
df_titanic_data['Title'][df_titanic_data.Title == 'Mme'] = 'Mrs' df_titanic_data['Title'][df_titanic_data.Title.isin(['Capt', 'Don', 'Major', 'Col', '3ir'])] = '3ir' df_titanic_data['Title'][df_titanic_data.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
# binarizing all the features
if keep_binary:
df_titanic_data = pd.concat( [df_titanic_data,pd.get_dummies(df_titanic_data['Title']).rename(columns=lambda x: 'Title_'+ str(x))],axis=l)
```
&emsp;&emsp;你还可以尝试从Name特征中找到其他有趣的特性。例如，您可能会考虑使用姓氏功能来了解泰坦尼克号上家庭成员的大小。

##### cabin（客舱变量）
&emsp;&emsp;在“泰坦尼克号”的数据中，cabin的特征由一个字母表示，一个字母表示甲板，一个数字表示房间号。船尾的房间数增加了，而且这将为测试乘客的位置提供了一些有用的措施。我们还可以从不同的甲板获得乘客的身份，这将有助于确定谁上了救生艇：<br>

```# repllacing the missing value in cabin variable "UO" 
df_titanic_data['Cabin'][df_titanic_data.Cabin.isnull()] = 'UO'

# the cabin number is a sequence of of alphanumerical digits, so we are going to create some features
# from the alphabetical part of it
df_titanic_data['CabinLetter'] = df_titanic_data['Cabin'].map(lambda l: get_cabin_letter(l))
df_titanic_data['CabinLetter'] = pd.factorize(df_titanic_data['CabinLetter'])[O]

# binarizing the cabin letters features if keep_binary:
cletters = pd.get_dummies(df_titanic_data['CabinLetter']).rename(columns=lambda x: 'CabinLetter_' + str(x))
df_titanic_data = pd.concat([df_titanic_data, cletters], axis=l)

# creating features from the numerical side of the cabin 
df_titanic_data['CabinNumber'] = df_titanic_data['Cabin'].map(lambda x: get_cabin_num(x)).astype(int) + l
```
##### ticket（船票变量）
&emsp;&emsp;票证功能的代码还不清楚，但我们可以做一些猜测，并尝试对它们进行分组。在查看票务功能之后，您可能会得到这些线索：<br>
&emsp;&emsp;&emsp;&emsp;近四分之一的门票以一个字符开头，其余的只有数字。<br>
&emsp;&emsp;售票码中的数字部分似乎对乘客的等级有一些指示。例如，从1开始的数字通常是头等舱票，2通常是第二位，而3ar则是3 ar。 e第三。我说这通常是因为它适用于大多数的例子，但不是所有的例子。也有从4-9开始的机票号码，这些都是罕见的，而且几乎完全是三等舱。几个人可以共享一个票号，这可能意味着一个家庭或密友一起旅行，并表现得像一个家庭。<br>

&emsp;&emsp;下面的代码试图分析票证特性代码，以便提供前面的线索：<br>
```# Helper function for constructing features from the ticket variable 
def process_ticket(): global df_titanic_data

df_titanic_data['TicketPrefix'] = df_titanic_data['Ticket'].map(lambda y: get_ticket_prefix(y.upper()))
df_titanic_data['TicketPrefix'] = df_titanic_data['TicketPrefix'].map(lambda y: re.sub('[\.?\/?]', '', y))
df_titanic_data['TicketPrefix'] = df_titanic_data['TicketPrefix'].map(lambda y: re.sub('3TON', '3OTON', y))

df_titanic_data['TicketPrefixId'] = pd.factorize(df_titanic_data['TicketPrefix'])[O]

# binarzing features for each ticket layer if keep_binary:
prefixes = pd.get_dummies(df_titanic_data['TicketPrefix']).rename(columns=lambda y: 'TicketPrefix_' + str(y))
df_titanic_data = pd.concat([df_titanic_data, prefixes], axis=l)
df_titanic_data.drop(['TicketPrefix'], axis=l, inplace=True) df_titanic_data['TicketNumber'] = df_titanic_data['Ticket'].map(lambda
y: get_ticket_num(y)) df_titanic_data['TicketNumberDigits'] =
df_titanic_data['TicketNumber'].map(lambda y: len(y)).astype(np.int) df_titanic_data['TicketNumber3tart'] =
df_titanic_data['TicketNumber'].map(lambda y: y[O:l]).astype(np.int)


df_titanic_data['TicketNumber'] = df_titanic_data.TicketNumber.astype(np.int)

if keep_scaled:
scaler_processing = preprocessing.3tandard3caler() df_titanic_data['TicketNumber_scaled'] =
scaler_processing.fit_transform(
df_titanic_data.TicketNumber.reshape(–l, l))


def get_ticket_prefix(ticket_value):
# searching for the letters in the ticket alphanumerical value 
match_letter = re.compile("([a–zA–Z\.\/]+)").search(ticket_value) if match_letter:
return match_letter.group() else:
return 'U'
def get_ticket_num(ticket_value):
# searching for the numbers in the ticket alphanumerical value 
match_number = re.compile("([\d]+$)").search(ticket_value)
if match_number:
return match_number.group() else:
return 'O'
```
#### 交互特征
&emsp;&emsp;交互特征是通过对特征集进行数学运算而得到的，并表示变量间关系的影响。我们对数值特征使用基本的数学运算，并看到变量之间关系的影响。 数字特征及变量间关系的影响：<br>

```# Constructing features manually based on	the interaction between the individual features
numeric_features = df_titanic_data.loc[:,['Age_scaled', 'Fare_scaled', 'Pclass_scaled', 'Parch_scaled', '3ib3p_scaled',
'Names_scaled', 'CabinNumber_scaled', 'Age_bin_id_scaled', 'Fare_bin_id_scaled']]
print("\nUsing only numeric features for automated feature generation:\n", numeric_features.head(lO))
new_fields_count = O
for i in range(O, numeric_features.columns.size – l):
for j in range(O, numeric_features.columns.size – l): if i <= j:
name = str(numeric_features.columns.values[i]) + "*" + str(numeric_features.columns.values[j])
df_titanic_data = pd.concat([df_titanic_data, pd.3eries(numeric_features.iloc[:, i] * numeric_features.iloc[:, j], name=name)],
axis=l) new_fields_count += l
if i < j:
name = str(numeric_features.columns.values[i]) + "+" + str(numeric_features.columns.values[j])
df_titanic_data = pd.concat([df_titanic_data, pd.3eries(numeric_features.iloc[:, i] + numeric_features.iloc[:, j], name=name)],
axis=l) new_fields_count += l
if not i == j:
name = str(numeric_features.columns.values[i]) + "/" + str(numeric_features.columns.values[j])
df_titanic_data = pd.concat([df_titanic_data, pd.3eries(numeric_features.iloc[:, i] / numeric_features.iloc[:, j], name=name)],
axis=l)
name = str(numeric_features.columns.values[i]) + "–" + str(numeric_features.columns.values[j])
df_titanic_data = pd.concat([df_titanic_data, pd.3eries(numeric_features.iloc[:, i] – numeric_features.iloc[:, j], name=name)],
axis=l) new_fields_count += 2

print("\n", new_fields_count, "new features constructed")
```
&emsp;&emsp;这种特征工程可以产生大量的特征。在前面的代码片段中，我们使用了9个特性来生成176个交互特性。<br>
&emsp;&emsp;我们还可以删除高度相关的特性，因为这些特性的存在不会向模型添加任何信息。我们可以使用Spearman的相关性来识别和删除高度相关的专长 肯定是。Spearman方法的输出中有一个秩系数，可以用来识别高度相关的特征：<br>

```# using 3pearman correlation method to remove the feature that have high correlation

# calculating the correlation matrix
df_titanic_data_cor = df_titanic_data.drop(['3urvived', 'PassengerId'],

axis=l).corr(method='spearman')

# creating a mask that will ignore correlated ones 
mask_ignore = np.ones(df_titanic_data_cor.columns.size) – np.eye(df_titanic_data_cor.columns.size) df_titanic_data_cor = mask_ignore * df_titanic_data_cor

features_to_drop = []

# dropping the correclated features
for column in df_titanic_data_cor.columns.values:

# check if we already decided to drop this variable if np.inld([column], features_to_drop):
continue

# finding highly correlacted variables
corr_vars = df_titanic_data_cor[abs(df_titanic_data_cor[column]) >
O.98].index
features_to_drop = np.unionld(features_to_drop, corr_vars)

print("\nWe are going to drop", features_to_drop.shape[O], " which are highly correlated features...\n") df_titanic_data.drop(features_to_drop, axis=l, inplace=True)
```
### 维数灾难
&emsp;&emsp;为了更好地解释维数灾难和过度拟合的问题，我们将通过一个有一组图像的例子。每个图像中都有一只猫或一只狗。所以 我们想要建立一个能够区分猫和狗的图像的模型。就像第一章“数据科学-鸟瞰”中的鱼类识别系统一样，我们需要找到学习算法可以用来区分这两类(猫和狗)。在本例中，我们可以认为颜色是一个很好的描述符，可以用于不同的 猫和狗之间。因此，平均红色、平均蓝色和平均绿色可以作为区分这两个类的解释性特征。<br>

&emsp;&emsp;然后，该算法将这三个特征以某种方式结合起来，在两个类之间形成一个决策边界。<br>

&emsp;&emsp;这三个特性的简单线性组合如下所示：<br>
`If O.5*red + O.3*green + O.2*blue > O.6 : return cat; else return dog;`

&emsp;&emsp;这些描述性的特性不足以获得一个良好的分类，所以我们可以决定增加更多的功能，以提高模型的预测能力，以区分猫和狗。 例如，我们可以考虑通过计算图像的平均边缘或梯度强度来增加一些特征，例如图像的纹理。<br>

&emsp;&emsp;图像的维数，X和Y。在添加这两个特征后，模型的精度将提高。我们甚至可以通过增加更多的分类器来使模型/分类器获得更精确的分类能力。 基于颜色、纹理直方图、统计矩等的特征。我们可以很容易地添加几百个这样的特性来增强模型的预测性。但是违反直觉的 超过一定限度增加功能后，效果会更差。通过查看图1，您将更好地理解这一点：<br>
![图片1](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter03/image1.jpg)<br>
&emsp;&emsp;图1 模型性能与特性数量之比<br>
&emsp;&emsp;图1显示，随着特征数的增加，分类器的性能也会提高，直到我们达到最优的特征数为止。在相同训练集的基础上添加更多的特征将降低分类器的性能。<br>

#### 避免维数灾难
&emsp;&emsp;在前面的例子中，我们发现当特征数超过某一最优点时，分类器的性能会下降。理论上，如果你有无限多的训练样本，那么维数灾难就不会存在。最优的功能数量完全取决于数据的大小。<br>
&emsp;&emsp;避免维数灾难的一种方法是从大量的特征N中提取M特征，其中M<N，M的每个特征可以是N中某些特征的组合。 还有一种常用的技术是主成分分析(PCA)，PCA试图找到较小数量的特征，捕捉原始数据的最大方差。<br>
&emsp;&emsp;博客：http：//www.visiondummy.com/2Ol4/O5/feature–extraction–using–pca/.<br>

&emsp;&emsp;将PCA应用于原始培训功能的一个有用而简单的方法是使用以下代码：<br>

```# minimum variance percentage that should be covered by the reduced number of variables
variance_percentage = .99

# creating PCA object
pca_object = PCA(n_components=variance_percentage)

# trasforming the features
input_values_transformed = pca_object.fit_transform(input_values, target_values)

# creating a datafram for the transformed variables from PCA pca_df = pd.DataFrame(input_values_transformed)

print(pca_df.shape[l], " reduced components which describe ", str(variance_percentage)[l:], "% of the variance")
```
&emsp;&emsp;在泰坦尼克号的例子中，我们尝试在原始特征上使用PCA和不应用PCA来构建分类器。因为我们在最后使用了随机森林分类器，所以我们发现应用PCA是 没有多大帮助；没有任何特性转换，随机森林运行得很好，甚至相关的特性也不会对模型产生太大影响。<br>

### 泰坦尼克号例子的完整重现
&emsp;&emsp;代码见.py文件<br>

### 偏置-方差分解
&emsp;&emsp;在上一节中，我们知道如何为模型选择最佳的超参数。在最小交叉验证误差的基础上，选择了一组最佳的超参数。现在，我们需要了解模型将如何处理未见数据，或者所谓的“样本外数据”，即指在模型培训阶段没有看到的新数据样本。<br>
&emsp;&emsp;考虑下面的示例：我们有一个大小为10，000的数据样本，我们将对相同的模型进行不同训练集大小的训练，并在每一步绘制测试错误图。例如，我们准备拿出1，000作为测试集，然后用另外的9，000进行训练。所以<br>
&emsp;&emsp;在第一轮训练中，我们将随机挑选一组大小为100的train。我们将根据最佳的超参数集对模型进行训练，并使用test集，最后绘制出列车(样本内)误差和测试(样本外)误差.我们重复这种训练、测试和绘图操作，以适应不同的列车尺寸(例如，重复WIT)。 9，000人中有500人，9，000人中有1，000人，依此类推)
在完成所有这些训练、测试和绘图之后，我们将得到一个由两条曲线组成的曲线图，用相同的模型表示train和test误差，但是跨越不同的train集合大小。从这个图里我们会知道我们的模型有多好。<br>

&emsp;&emsp;输出图将包含表示训练和测试错误的两条曲线，它将是图2所示的四种可能的形状之一。 <br>
&emsp;&emsp;课程内容(https：/www.Coursera.org/Learning/Machine-Learning)。<br>
![图片2](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter03/image2.jpg)<br>
&emsp;&emsp;图2 绘制不同训练集大小上的训练和测试错误的可能形状<br>
&emsp;&emsp;那么，我们什么时候应该接受我们的模式并投入生产呢？我们什么时候知道我们的模型在test集上表现不佳，因此不会有一个糟糕的泛化误差？对于这些问题，取决于从绘制train误差到不同训练集大小上的测试错误所得到的形状：<br>

&emsp;&emsp;如果您的形状看起来像左上角的形状，则表示一个较低的训练误差，并且在测试集上具有很好的泛化性。这个形状是一个赢家，你应该继续使用这种型号的生产。<br>
&emsp;&emsp;如果您的形状与右上角的形状相似，则表示一个较高的训练误差(该模型未能从训练样本中学习)，甚至在test集上具有更差的泛化性能。 最好的一组。这个形状是完全失败的，您需要回去看看您的数据、选择的学习算法和/或所选的超参数有什么问题。<br>
&emsp;&emsp;如果您的形状与左下角的形状相似，则表示一个糟糕的训练误差，因为模型无法捕获数据的底层结构，这也符合新的测试数据。<br>
&emsp;&emsp;如果您的形状类似于右下角的形状，则表示高度的偏差和差异。这意味着您的模型没有很好地计算出培训数据，因此没有得到很好的概括。<br>

&emsp;&emsp;偏差和方差是我们可以用来判断我们的模型有多好的组成部分。在监督学习中，有两个相反的误差来源，并使用图2，我们可以找出我们的模型所受的影响。具有高方差和低偏差的问题称为过拟合，这意味着模型在训练过程中表现良好。但在测试集上没有得到很好的概括。<br>

&emsp;&emsp;另一方面，高偏差和低方差的问题被称为欠拟合，这意味着模型没有利用数据，也没有设法估计输出/目标。 输入功能。可以使用不同的方法来避免陷入这些问题之一。但通常情况下，加强其中一项将以牺牲第二项为代价。<br>

&emsp;&emsp;我们可以通过增加更多的特征来解决高方差的情况，使模型能够从中学习。这个解决方案很可能会增加偏见，所以您需要在它们之间做一些权衡。<br>

### 学习能见度
&emsp;&emsp;有许多伟大的数据科学算法，可以用来解决不同领域的问题，但关键的组成部分，使学习过程可见的是拥有足够的数据。你也许 询问需要多少数据才能使学习过程可见并值得进行。根据经验法则，研究人员和机器学习实践者一致认为，你需要的数据样本至少是模型自由度的10倍。<br>
&emsp;&emsp;例如，在线性模型的情况下，自由度表示数据集中的特征数量。如果数据中有50个解释性特性，那么数据中至少需要500个数据样本/观察。<br>

### 打破经验法则
&emsp;&emsp;在实践中，您可以通过使用数据中的特性数不到10倍来学习这个规则；如果您的模型很简单，并且您使用的是名为 正规化(在下一章中讨论)。<br>

&emsp;&emsp;JakeVanderplas写了一篇文章，(https：//jakevdp.github.io/blog/2Ol5/O7/O6/model–复杂性-神话/)，以表明一个人可以学习，即使数据有更多的参数，而不是例子。为了证明这一点，他用的是正规化。<br>

### 总结
&emsp;&emsp;在这一章中，我们介绍了机器学习实践者为了理解他们的数据和从他们的数据中获取最大限度的学习算法而使用的最重要的工具。<br>
&emsp;&emsp;特征工程是数据科学中第一个也是最常用的工具，它是任何数据科学管道中必不可少的组件。此工具的目的是为您的d提供更好的表示。 增加模型的预测能力。<br>
&emsp;&emsp;我们看到了大量的特征是如何有问题的，并导致更差的分类器性能。我们还看到了应该使用的最优特性数来获得最大的模型。 性能，这个最优的特性数量是一个函数的数据样本/观察得到的数量。<br>
&emsp;&emsp;随后，我们介绍了最强大的工具之一，即偏差-方差分解.这个工具被广泛用于测试模型在测试集上有多好。<br>
&emsp;&emsp;最后，我们学习了可见性，这就回答了这个问题。<br>
&emsp;&emsp;我们需要的数据，以获得商业和做机器学习。经验法则表明，我们需要数据样本/观察，至少是数据中特性数量的10倍。然而， 这个经验法则可以通过使用另一个叫做正则化的工具来打破，这个工具将在下一章中更详细地讨论。<br>
&emsp;&emsp;接下来，我们将继续增加我们的数据科学工具，我们可以使用这些工具从我们的数据中进行有意义的分析，并面临一些日常应用机器学习的问题。<br>
