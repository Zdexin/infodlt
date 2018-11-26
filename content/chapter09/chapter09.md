# 第九章 目标检测-卷积神经网络的迁移学习<br>
### 如何在一个环境迁移到另一个具有相似特征的环境中加以应用。
                                                       – E. L. Thorndike, R. S. Woodworth (1991)
&emsp;&emsp; 迁移学习（TL）是数据科学中的一个研究问题，主要涉及在解决特定任务中获得的知识的持续性，并利用所获得的知识来解决另一个不同但相似的任务。在这一章中，我们将演示迁移学习。数据科学领域中使用的一种现代实践和共同主题。这里的想法是如何从具有非常大的数据集的领域获得对数据集较小的领域的帮助。最后，我们将重温CIFAR-10的目标检测示例，并尝试通过迁移学习减少训练时间和性能误差。<br>
&emsp;&emsp; 本章将讨论以下主题：<br>
     •迁移学习<br>
     •CIFAR-10目标检测回顾<br>
# 迁移学习<br>
&emsp;&emsp; 深度学习的架构是依赖于数据的，并且在训练集中只有少量样本不会使我们从中获益。迁移学习通过将已学习或获得的知识/表示为未解决具有大数据集的任务转移到具有小数据集的另一个不同但相似的任务来解决这个问题。迁移学习不仅适用于小训练集的情况，而且我们可以使用它使训练过程更快。从零开始训练大型的深度学习架构有时可能非常慢，因为我们在这些架构中有数百万的权重需要学习。取而代之的是，人们可以通过微调学习到的权重来利用迁移学习，就像他/她试图解决的问题一样。<br>
## 迁移学习背后的直觉<br>
&emsp;&emsp; 让我们用下面的师生类比建立迁移学习背后的直觉。一位教师在他所教的教学模块中有多年的经验。另一方面，学生从老师给出的讲座中得到一个紧凑的主题概述。所以你可以说老师正在把他们的知识以简洁而紧凑的方式传递给学生。<br>
&emsp;&emsp; 教师和学生的类比同样适用于我们在深度学习或一般的神经网络中传递知识的情况。因此，我们的模型从数据中学习一些表示，这是由网络的权值来表示的。这些学习的表征/特征（权重）可以被转移到另一个不同但相似的任务。将学习的权重转移到另一个任务的过程将减少对深度学习架构的巨大数据集收敛的需要，与从头开始训练模型相比，并且也将减少模型适应新数据集所需的时间。<br>
&emsp;&emsp; 现在，深度学习被广泛使用，但通常大多数人在训练深度学习体系结构时使用迁移学习；很少有人从头开始训练深度学习体系结构，因为大多数时候很少有足够的数据集来进行深层次的学习。因此，在像IMANET这样的大型数据集上使用预先训练的模型是非常普遍的，它有大约120万张图像，并将其应用到新的任务中。我们可以使用预先训练好的模型的权重作为特征提取器，或者我们可以用它初始化我们的体系结构，然后根据新的任务对其进行微调。现在使用迁移学习有三种主要方案：<br>
## 1.使用卷积网络作为固定特征提取器<br>
&emsp;&emsp; 在此场景中，诸如ImageNet之类的大型数据集上使用经过预训练的卷积模型，并使用它来解决问题。例如，在ImageNet上预先训练的卷积模型将有一个完全连接的层，该层具有ImageNet所具有的1000个类别的输出分数。因此，需要删除这个层，因为对IMANET的类不再感兴趣。然后，将所有其他层视为特征提取器。一旦使用预先训练的模型提取了特征，就可以将这些特征提供给任何线性分类器，例如Softmax分类器，甚至线性SVM（SVM支持向量机：是一种二类分类模型，他的基本模型是定义在特征空间上的间隔最大的线性分类器。 间隔最大使它有别于感知机；支持向量机还包括核技巧，这使它成为实质上的线性分类器。支持向量机的学习策略就是间隔最大化，可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题，支持向量机的学习算法是求解凸二次规划的最优化算法）。<br>
## 2.卷积神经网络的微调<br>
&emsp;&emsp; 第二种方案涉及第一种方案，但是要额外使用反向传播来微调新任务上的预先训练的权重。通常，人们把大部分的层固定下来，只对网络的顶端进行微调。尝试微调整个网络，但是大多数层可能会导致过度拟合。所以，你可能只关注微调那些与图像的语义层次特征有关的层。对早期的层固定下来的层的直觉是它们包含通用的或低级的特性，这些特性在大多数成像任务中都是通用的，比如角、边等等。如果要引入新类，那么对网络的较高层或顶端层进行微调将是非常有用的，这些类在模型预先训练的原始数据集中是不存在的。<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter09/aq1.jpg)<br>
图10.1：对预先训练的卷积神经网络进行一项新任务的微调
## 3.预训练模型<br>
&emsp;&emsp; 第三种方案广泛使用的场景是人们在因特网上下载可用的检查点。你可以不去使用这个具有大的计算能力模型，而是从头开始训练模型，所以你只要使用释放的检查站来初始化模型，然后做一些精细调整.<br>
## 传统的机器学习和迁移学习之间的区别<br>
&emsp;&emsp; 正如在前一节中注意到的，我们应用机器学习的传统方法与涉及迁移学习的机器学习之间存在明显的区别（如下图所示）。在传统的机器学习中，不会将任何知识或表示传递给任何其他任务，迁移学习的情况并非如此。有时，人们使用迁移学习的方式是错误的，所以我们将提到一些条件，在这些条件下，我们才能使用迁移学习来达到最大化的收益。<br>
&emsp;&emsp; 以下是应用迁移学习的条件：<br>
#### &emsp;&emsp; 1.不同于传统的机器学习，源和目标任务或域名没有来自相同的分布，但他们都是相似的。<br>
#### &emsp;&emsp; 2. 如果你没有必要的计算能力，也可以在较少的训练样本的情况下使用迁移学习。<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter09/ap2.jpg)<br>
图10.2：传统机器学习与迁移机器学习<br>
## CIFAR-10目标检测回顾<br>
&emsp;&emsp; 在前一章中，我们在CIFAR-10数据集上训练了一个简单的卷积神经网络（CNN）模型。在这里，我们将演示使用预先训练的模型作为特征提取器，同时移除预先训练模型的完全连接层的情况，然后我们将这些提取的特征或转移的值馈送到softmax层。<br>
&emsp;&emsp; 在这个实现过程中的预先训练模型将是初始模型，将对ImageNet进行预训练。但这一方法是基于前两章介绍卷积神经网络的。<br>
### 解决方案概要<br>
&emsp;&emsp; 再者，我们要替换的是最终完全连接层的预先训练的起始模型，然后使用起始模型的其余部分作为特征提取模型。所以，首先给原始图像创建模型，并从中提取特征，然后输出我们所称为的迁移价值。<br>
&emsp;&emsp; 从初始模型提取得到的特征值迁移后，由于你花费了很多时间运行程序，但还是运行不快，者可能需要将特征值保存到你的桌面上，坚持把他们放到桌面上将会节省你的运行时间。在TensorFlow教程，使用的术语是“瓶颈”价值而非迁移的价值，但它是同样的涵义，只是一个不同的名字。<br>
&emsp;&emsp; 获取特征值或负载值传递到桌面后，根据我们的新任务我们可以给他们分配任何线性分类器。在这里，我们将提取的特征值转移到另一个神经网络，然后再训练CIFAR-10中的新类。<br>
&emsp;&emsp; 下面的图表显示了我们将遵循的一般解决方案概要：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter09/ap3.JPG)<br>
图10.3是使用迁移学习的方法对CIFAR-10数据集的对象进行检测的解决方案概述<br>
### 加载和探索CIFAR-10<br>
&emsp;&emsp; 让我们开始导入实现这个程序所需的包：
```python %matplotlib inline
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np 
import time
from datetime import timedelta import os
# Importing a helper module for the functions of the Inception model. import inception
```
&emsp;&emsp; 下一步，我们需要加载另一个脚本来帮助下载和处理CIFAR-10数据集：
```python 
import cifarlO
#importing number of classes of CIFAR–lO from cifarlO import num_classes
```
&emsp;&emsp; 如果您还没有设置，您需要设置CIFAR-10的路径。此路径将被cifar–lO.py脚本用于保存数据集：
```python
cifarlO.data_path = "data/CIFAR–lO/"
The CIFAR–lO dataset is about l7O MB, the next line checks if the dataset is already downloaded if not it downloads the dataset and store in the previous data_path:
cifarlO.maybe_download_and_extract</span>() Output:
–	Download progress: lOO.O%
Download finished. Extracting files. Done.
```
&emsp;&emsp; 让我们看看CIFAR-10数据集中的类别：
```python
#Loading the class names of CIFAR–lO dataset 
class_names = cifarlO.load_class_names() 
class_names
```
输出：
```python
Loading data: data/CIFAR–lO/cifar–lO–batches–py/batches.meta ['airplane',
'automobile', 'bird',
'cat',
'deer',
'dog',
'frog',
'horse',
'ship',
'truck']
Load the training–set.
```
它将图像、类号作为整数，并将类号作为one hot编码数组称为标签：
```python
training_images, training_cls_integers, trainig_one_hot_labels = cifarlO.load_training_data()
```
输出：
```python
Loading data: data/CIFAR–lO/cifar–lO–batches–py/data_batch_l 
Loading data: data/CIFAR–lO/cifar–lO–batches–py/data_batch_2 
Loading data: data/CIFAR–lO/cifar–lO–batches–py/data_batch_3 
Loading data: data/CIFAR–lO/cifar–lO–batches–py/data_batch_4 
Loading data: data/CIFAR–lO/cifar–lO–batches–py/data_batch_5 
Load the test–set.
```
&emsp;&emsp; 接下来，让我们对测试集执行相同的操作，通过加载目标类的图像和其对应的整数来表示其单热编码：
```python
#Loading the test images, their class integer, and their corresponding one– hot encoding
testing_images, testing_cls_integers, testing_one_hot_labels = cifarlO.load_test_data()
Output:
Loading data: data/CIFAR–lO/cifar–lO–batches–py/test_batch
```
让我们看看CIFAR–10中训练和测试集的分布情况：
```python
print("–Number of images in the training set:\t\t(}".format(len(training_images))) 
print("–Number of images in the testing set:\t\t(}".format(len(testing_images)))
```
输出:
```python
–Number of images in the training set:	5OOOO
–Number of images in the testing set:	lOOOO
```
&emsp;&emsp; 让我们定义一些辅助函数，使我们能够探索数据集。以下辅助函数在网格中绘制了一组九幅图像：
```python
def plot_imgs(imgs, true_class, predicted_class=None): assert len(imgs) == len(true_class)
# Creating a placeholders for 9 subplots fig, axes = plt.subplots(3, 3)
# Adjustting spacing.
if predicted_class is None: hspace = O.3
else:
hspace = O.6 fig.subplots_adjust(hspace=hspace, wspace=O.3)
for i, ax in enumerate(axes.flat):
# There may be less than 9 images, ensure it doesn't crash. if i < len(imgs):
# Plot image. ax.imshow(imgs[i],
interpolation='nearest')
array 
# Get the actual name of the true class from the class_names true_class_name = class_names[true_class[i]]
# 3howing labels for the predicted and true classes if predicted_class is None:
xlabel = "True: (O}".format(true_class_name) else:
# Name of the predicted class.
predicted_class_name = class_names[predicted_class[i]]
xlabel = "True: (O}\nPred: (l}".format(true_class_name, predicted_class_name)
ax.set_xlabel(xlabel)
# Remove ticks from the plot. ax.set_xticks([]) ax.set_yticks([])
plt.show()
```
我们先将测试集中的一些图像连同它们对应的实际类可视化：
```python
# get the first 9 images in the test set imgs = testing_images[O:9]
# Get the integer representation of the true class. true_class = testing_cls_integers[O:9]
# Plotting the images
plot_imgs(imgs=imgs, true_class=true_class)
```
输出：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter09/chapter_09image/ap1.JPG)<br>
图10.4：测试集的前九个图像
## 初始模型迁移值
&emsp;&emsp; 正如我们前面提到的，我们将在IMANET数据集上使用预训练的初始模型。所以，我们需要从互联网上下载这个预先训练的模型。<br>
让我们从初始模型的定义开始：
`inception.data_dir = 'inception/'`
&emsp;&emsp; 预训练起始模型的权重约为85 MB。如果在前面定义的data_dir中不存在，那么下面的代码行将下载它：
`inception.maybe_download() `
&emsp;&emsp; 我们将加载初始模型，以便我们可以使用它作为CIFAR–10图像的特征提取器：
```python
# Loading the inception model so that we can inialized it with the pre– trained weights and customize for our model
inception_model = inception.Inception()
```
&emsp;&emsp; 如前所述，计算CIFAR-10数据集的传输值需要一些时间，因此我们需要缓存它们以便将来使用。谢天谢地，在开始模块中有一个辅助函数可以帮助我们做到这一点：
`from inception import transfer_values_cache`
&emsp;&emsp; 下一步，我们需要为缓存的训练和测试文件设置文件路径：
```python
file_path_train = os.path.join(cifarlO.data_path, 'inception_cifarlO_train.pkl')
file_path_test = os.path.join(cifarlO.data_path, 'inception_cifarlO_test.pkl')
print("Processing Inception transfer–values for the training images of Cifar–lO ...")
# First we need to scale the imgs to fit the Inception model requirements as it requires all pixels to be from O to 255,
# while our training examples of the CIFAR–lO pixels are between O.O and l.O
imgs_scaled = training_images * 255.O
# Checking if the transfer–values for our training images are already calculated and loading them, if not calculate and save them. transfer_values_training = transfer_values_cache(cache_path=file_path_train,images=imgs_scaled, model=inception_model)
print("Processing Inception transfer–values for the testing images of Cifar–lO ...")
# First we need to scale the imgs to fit the Inception model requirements as it requires all pixels to be from O to 255,
# while our training examples of the CIFAR–lO pixels are between O.O and l.O
imgs_scaled = testing_images * 255.O
# Checking if the transfer–values for our training images are already calculated and loading them, if not calcaulate and save them. transfer_values_testing = transfer_values_cache(cache_path=file_path_test,images=imgs_scaled, model=inception_model)
```
&emsp;&emsp; 如前所述，我们在CIFAR-10数据集的训练集中有50000个图像。让我们检查这些图像的传输值的形状。这个训练集中的每个图像应该是2048：
`transfer_values_training.shape`
输出：
(5OOOO, 2O48)
&emsp;&emsp; 我们需要设置相同的测试数据;
`transfer_values_testing.shape`
输出：
(lOOOO, 2O48)
&emsp;&emsp; 直观地从迁移价值来看，我们将从训练或测试集定义一个辅助函数，使我们能够使用一个特定的图像细节来传递特征值：
```python
def plot_transferValues(ind): print("Original input image:")
# Plot the image at index ind of the test set. plt.imshow(testing_images[ind], interpolation='nearest') plt.show()
print("Transfer values using Inception model:")
# Visualize the transfer values as an image. transferValues_img = transfer_values_testing[ind] transferValues_img = transferValues_img.reshape((32, 64))
# Plotting the transfer values image.
plt.imshow(transferValues_img, interpolation='nearest', cmap='Reds') plt.show()
plot_transferValues(i=l6) Input image:
```
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter09/chapter_09image/ap2.JPG)<br>
图片10.5 输出图片<br。
使用初始模型传递图像的值：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter09/chapter_09image/ap3.JPG)<br>
图10.6：图10.3中输入图像的迁移值<br>
`plot_transferValues(i=l7)`
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter09/chapter_09image/ap4.JPG)<br>
图10.7：输入图片<br>
使用初始模型传递图像的值：<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter09/chapter_09image/ap5.JPG)<br>
图10.8：图10.5中输入图像的迁移值<br>
## 迁移学习的价值分析
&emsp;&emsp; 在这一部分中，我们将对训练图像所得到的传递值做一些分析。这个分析的目的是要看看这些传递值是否足以对CIFAR-10中的图像进行分类。
&emsp;&emsp; 对于每个输入图像，我们有2048个传输值。为了绘制这些传递值并对其做进一步的分析，我们可以使用诸如scikit-learning中的主成分分析(PCA)法，例如降维技术。我们将把转移值从2048减少到2，以便能够得到可视化结果，并查看它们是否是区分不同类别CIFAR-10的良好特征：
`from sklearn.decomposition import PCA`
&emsp;&emsp; 下一步，我们需要创建一个PCA对象，其中组件的数量只有2个：
`pca_obj = PCA(n_components=2)`
&emsp;&emsp; 将传输值从2048减少到2需要很多时间，因此我们将在具有以下传输值的5000个图像中仅对其中3000个图像形成的子集进行处理
`subset_transferValues = transfer_values_training[O:3OOO]`
&emsp;&emsp; 我们还需要得到这些图像的类号：
`cls_integers = testing_cls_integers[O:3OOO]`
&emsp;&emsp; 我们可以通过打印传输值的形状来检查我们的子设置：
`subset_transferValues.shape`
输出：
(3OOO, 2O48)
&emsp;&emsp; 接下来，我们使用我们的PCA对象来减少从2048到2的传输值：
`reduced_transferValues = pca_obj.fit_transform(subset_transferValues)`
&emsp;&emsp; 现在，让我们看看PCA还原过程的输出：
`reduced_transferValues.shape`
输出：
(3OOO, 2)
&emsp;&emsp; 在将传递值的维数降低到仅2之后，让我们绘制这些值：
```python
#Importing the color map for plotting each class with different color. import matplotlib.cm as color_map
def plot_reduced_transferValues(transferValues, cls_integers):
# Create a color–map with a different color for each class. c_map = color_map.rainbow(np.linspace(O.O, l.O, num_classes))
# Getting the color for each sample. colors = c_map[cls_integers]
# Getting the x and y values. x_val = transferValues[:, O] y_val = transferValues[:, l]
# Plot the transfer values in a scatter plot plt.scatter(x_val, y_val, color=colors) plt.show()
```
&emsp;&emsp; 在这里，我们正在绘制从训练集到子集减少的转移值。我们在CIFAR-10中有10个类，所以我们将用不同的颜色绘制它们对应的传输值。从下面的图表可以看出，我们把不同传输值根据相应的类进行分组。不同组之间的重叠主要是因为PCA的缩小过程中不能正确地分离传递值：
`plot_reduced_transferValues(reduced_transferValues, cls_integers) from sklearn.manifold import T3NE`
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter09/chapter_09image/ap6.JPG)<br>
&emsp;&emsp; 同样，我们将减少传输值的维数，原来是2048，但是这次是50个值也不是2个：
```python
pca_obj = PCA(n_components=5O)
transferValues_5Od = pca_obj.fit_transform(subset_transferValues)
```
&emsp;&emsp; 接下来，我们使用第二降维技术进行叠加，并将PCA过程的输出反馈给叠加值：
`reduced_transferValues = tsne_obj.fit_transform(transferValues_5Od)`
&emsp;&emsp; 并仔细检查是否有正确的形状：
`reduced_transferValues.shape`
输出：(3OOO, 2)<br>
&emsp;&emsp; 们用T-SNE方法绘制减少的传递值。正如在下一个图像中看到的，T-SNE能够比PCA更好地分离不同组的传输值。
&emsp;&emsp; 从该分析中得出的结论是，通过将输入图像反馈送到预先训练的初始模型而获得的提取传输值可用于将训练图像分成10类。由于以下图表中的微小重叠，这种分离不会达到100%精确，但是我们可以通过对预先训练的模型进行一些微调来消除这种重叠：
`plot_reduced_transferValues(reduced_transferValues, cls_integers)`
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter09/chapter_09image/ap7.JPG)<br>
图10.10：使用T-SNE减少传输值<br>
&emsp;&emsp; 现在我们有从我们的训练图像中提取的转移值，我们知道这些值能够在一定程度上区分CIFAR-10所具有不同的类。接下来，我们需要建立一个线性分类器，并将这些转移值反馈并进行实际分类。<br>
## 模型的构建与训练
&emsp;&emsp; 因此，让我们从指定输入占位符变量开始，这些变量将被输入到我们的神经网络模型中。第一个输入变量的形状（将包含提取的传输值）将是[None,transfer_len]。第二个占位符变量将以一个向量格式保存，是训练集的实际类标签：
```python
transferValues_arrLength = inception_model.transfer_len input_values = tf.placeholder(tf.float32, shape=[None,transferValues_arrLength], name='input_values') y_actual = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_actual')
```
&emsp;&emsp; 我们还可以通过定义另一个占位符变量，从1到10得到每个类的相应的整数值：
`y_actual_cls = tf.argmax(y_actual, axis=l)`
&emsp;&emsp; 接下来，我们需要建立实际分类的神经网络，该神经网络将采用这些输入占位符并产生预测的类：
```python
def new_weights(shape):
return tf.Variable(tf.truncated_normal(shape, stddev=O.O5))
def new_biases(length):
return tf.Variable(tf.constant(O.O5, shape=[length]))
def new_fc_layer(input,	# The previous layer.num_inputs,	# Num. inputs from prev. layer. num_outputs,	# Num. outputs.use_relu=True): 
# Use Rectified Linear Unit (ReLU)?
# Create new weights and biases.
weights = new_weights(shape=[num_inputs, num_outputs]) biases = new_biases(length=num_outputs)
# Calculate the layer as the matrix multiplication of
# the input and weights, and then add the bias–values. layer = tf.matmul(input, weights) + biases
# Use ReLU? if use_relu:
layer = tf.nn.relu(layer) return layer
# First fully–connected layer.
layer_fcl = new_fc_layer(input=input_values,num_inputs=2O48, num_outputs=lO24, use_relu=True)
# 3econd fully–connected layer.
layer_fc2 = new_fc_layer(input=layer_fcl,num_inputs=lO24, num_outputs=num_classes, use_relu=False)
# Predicted class–label.
y_predicted = tf.nn.softmax(layer_fc2)
# Cross–entropy for the classification of each image. cross_entropy = \
tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_actual)
# Loss aka. cost–measure.
# This is the scalar value that must be minimized. loss = tf.reduce_mean(cross_entropy)
```
&emsp;&emsp; 然后，我们需要定义在分类器的训练期间使用的优化准则。在这个实现中，我们将使用AdamOptimizer优化器。对应于CIFAR-10数据集中的类数，这个分类器的输出将是一个有10个概率分数的数组。然后，我们将在这个数组上应用ARGMAX操作，将得到的最大的类分配给这个输入样本：
```python
step = tf.Variable(initial_value=O,name='step', trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=le–4).minimize(loss, step) y_predicted_cls = tf.argmax(y_predicted, axis=l)
#compare the predicted and true classes
correct_prediction = tf.equal(y_predicted_cls, y_actual_cls)
#cast the boolean values to fload
model_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
&emsp;&emsp; 下一步，我们需要定义一个TensorFlow会话，它将实际执行该图，然后初始化在本实现中更早定义的变量：
`session = tf.3ession() session.run(tf.global_variables_initializer())`
&emsp;&emsp; 在这个实现中，我们将使用随机梯度下降（SGD），所以我们需要定义一个函数，来从我们的50000幅图像的训练集中随机生成特定大小的批次。
因此，我们将定义一个辅助函数，用于从输入值的传递值集合中生成一个随机集合：
```
#defining the size of the train batch train_batch_size = 64
#defining a function for randomly selecting a batch of images from the dataset
def select_random_batch():
# Number of images (transfer–values) in the training–set. num_imgs = len(transfer_values_training)
# Create a random index.
ind = np.random.choice(num_imgs,size=training_batch_size, replace=False)
# Use the random index to select random x and y–values.
# We use the transfer–values instead of images as x–values. x_batch = transfer_values_training[ind]
y_batch = trainig_one_hot_labels[ind] return x_batch, y_batch
```
&emsp;&emsp; 接下来，我们需要定义一个辅助函数来执行实际的优化过程，这将细化网络的权重。它将在每一次迭代中生成一批传递值，并基于该批次优化网络：
```python
def optimize(num_iterations):
for i in range(num_iterations):
# 3electin a random batch of images for training
# where the transfer values of the images will be stored in input_batch
# and the actual labels of those batch of images will be stored in y_actual_batch
input_batch, y_actual_batch = select_random_batch()
# storing the batch in a dict with the proper names
# such as the input placeholder variables that we define above. feed_dict = (input_values: input_batch,y_actual: y_actual_batch)
# Now we call the optimizer of this batch of images
# TensorFlow will automatically feed the values of the dict we created above
# to the model input placeholder variables that we defined above. i_global, _ = session.run([step, optimizer],feed_dict=feed_dict)
# print the accuracy every lOO steps.
if (i_global % lOO == O) or (i == num_iterations – l):
# Calculate the accuracy on the training–batch. batch_accuracy = session.run(model_accuracy,feed_dict=feed_dict)
msg = "3tep: (O:>6}, Training Accuracy: (l:>6.l%}" print(msg.format(i_global, batch_accuracy))
```
&emsp;&emsp; 我们将定义一些辅助函数来显示先前神经网络的结果，并显示预测结果的混淆矩阵：
```python
def plot_errors(cls_predicted, cls_correct):
# cls_predicted is an array of the predicted class–number for
# all images in the test–set.
# cls_correct is an array with boolean values to indicate
# whether is the model predicted the correct class or not.
# Negate the boolean array. incorrect = (cls_correct == False)
# Get the images from the test–set that have been
# incorrectly classified.
incorrectly_classified_images = testing_images[incorrect]
# Get the predicted classes for those images. cls_predicted = cls_predicted[incorrect]
# Get the true classes for those images. true_class = testing_cls_integers[incorrect]
n = min(9, len(incorrectly_classified_images))
# Plot the first n images. plot_imgs(imgs=incorrectly_classified_images[O:n],true_class=true_class[O:n], predicted_class=cls_predicted[O:n])
Next, we need to define the helper function for plotting the confusion matrix:
from sklearn.metrics import confusion_matrix def plot_confusionMatrix(cls_predicted):
# cls_predicted array of all the predicted
# classes numbers in the test.
# Call the confucion matrix of sklearn
cm = confusion_matrix(y_true=testing_cls_integers,y_pred=cls_predicted)
# Printing the confusion matrix for i in range(num_classes):
# Append the class–name to each line.
class_name = "((}) (}".format(i, class_names[i]) print(cm[i, :], class_name)
# labeling each column of the confusion matrix with the class number cls_numbers = [" ((O})".format(i) for i in range(num_classes)]
print("".join(cls_numbers))
```
&emsp;&emsp; 此外，我们将定义另一个辅助函数，以便在测试集上运行经过训练的分类器，得到测试集上经过训练的模型的精准度：
```python
# 3plit the data–set in batches of this size to limit RAM usage. batch_size = l28
def predict_class(transferValues, labels, cls_true):
# Number of images.
num_imgs = len(transferValues)
# Allocate an array for the predicted classes which
# will be calculated in batches and filled into this array. cls_predicted = np.zeros(shape=num_imgs, dtype=np.int)
# Now calculate the predicted classes for the batches.
# We will just iterate through all the batches.
# There might be a more clever and Pythonic way of doing this.
# The starting index for the next batch is denoted i. i = O
while i < num_imgs:
# The ending index for the next batch is denoted j. j = min(i + batch_size, num_imgs)
# Create a feed–dict with the images and labels
# between index i and j.
feed_dict = (input_values: transferValues[i:j], y_actual: labels[i:j]}
# Calculate the predicted class using TensorFlow. cls_predicted[i:j] = session.run(y_predicted_cls,feed_dict=feed_dict)
# 3et the start–index for the next batch to the
# end–index of the current batch. i = j
# Create a boolean array whether each image is correctly classified. correct = [a == p for a, p in zip(cls_true, cls_predicted)]
return correct, cls_predicted
#Calling the above function making the predictions for the test def predict_cls_test():
return predict_class(transferValues = transfer_values_test,labels = labels_test, cls_true = cls_test)
def classification_accuracy(correct):
# When averaging a boolean array, False means O and True means l.
# 3o we are calculating: number of True / len(correct) which is
# the same as the classification accuracy.
# Return the classification accuracy
# and the number of correct classifications. return np.mean(correct), np.sum(correct)
def test_accuracy(show_example_errors=False,show_confusion_matrix=False):
# For all the images in the test–set,
# calculate the predicted classes and whether they are correct. correct, cls_pred = predict_class_test()
# Classification accuracypredict_class_test and the number of correct classifications.accuracy, num_correct = classification_accuracy(correct)
# Number of images being classified. num_images = len(correct)
# Print the accuracy.
msg = "Test set accuracy: (O:.l%} ((l} / (2})" print(msg.format(accuracy, num_correct, num_images))
# Plot some examples of mis–classifications, if desired. if show_example_errors:
print("Example errors:") plot_errors(cls_predicted=cls_pred, cls_correct=correct)
# Plot the confusion matrix, if desired. if show_confusion_matrix:
print("Confusion Matrix:") plot_confusionMatrix(cls_predicted=cls_pred)
```
&emsp;&emsp; 让我们看看之前的神经网络模型的性能，然后再做其他优化：
```python
test_accuracy(show_example_errors=True,
show_confusion_matrix=True) Accuracy 
```
&emsp;&emsp; 正如您所看到的，网络的性能非常低，但是基于我们已经定义的优化标准进行一些优化之后，它会变得更好。因此，我们将优化器进行10000次迭代，然后测试模型准确度：
```python
optimize(num_iterations=lOOOO) test_accuracy(show_example_errors=True,
show_confusion_matrix=True) Accuracy
```
Example errors:<br>
![image](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter09/chapter_09image/ap8.JPG)<br>
图10.11：来自测试集的一些错误分类图像<br>
```
Confusion	Matrix:	
[926	6	l3	2	3	O	l	l	29	l9]	(O)	airplane
[	9	92l	2	5	O	l	l	l	2	58]	(l)	automobile
[	l8	l	883	3l	32	4	22	5	l	3]	(2)	bird
[	7	2	l9	855	23	57	24	9	2	2]	(3)	cat
[	5	O	2l	25	896	4	24	22	2	l]	(4)	deer
[	2	O	l2	97	l8	843	lO	l5	l	2]	(5)	dog
[	2	l	l6	l7	l7	4	94O	l	2	O]	(6)	frog
[	8	O	lO	l9	28	l4	l	9l4	2	4]	(7)	horse
[	42	6	l	4	l	O	2	O	932	l2]	(8)	ship
[	6	l9	2	2	l	O	l	l	9	959]	(9)	truck
(O) (l) (2) (3) (4) (5) (6) (7) (8) (9)
```
为了总结这一点，我们将关闭已经运行的模型：
```
model.close() 
session.close()
```
# 总结
&emsp;&emsp; 在本章中，我们介绍了最广泛使用的深度学习的最佳实践之一——图像的目标检测。迁移学习是一个非常令人兴奋的工具，我们可以使用它来深入学习体系结构，以便从小型数据集中得到训练模型，但是要以正确的方式迁移到其他领域使用它。
&emsp;&emsp; 接下来，我们将介绍广泛用于自然语言处理的深度学习体系结构。这些递归型结构在以下大多数NLP（自然语言处理）领域都取得了突破：机器翻译、语音识别、语言建模和情感分析。
学号|姓名|专业
-|-|-
2011802110485|李忠|计算机应用技术





