# 目标检测-卷积神经网络的迁移学习<br>
## 如何在一个环境迁移到另一个具有相似特征的环境中加以应用。
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
### &emsp;&emsp; 1.不同于传统的机器学习，源和目标任务或域名没有来自相同的分布，但他们都是相似的。<br>
### &emsp;&emsp; 2. 如果你没有必要的计算能力，也可以在较少的训练样本的情况下使用迁移学习。<br>
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
–Number of images in the training set:  5OOOO
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





