# &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;对象检测__CIFAR-10实例
在介绍了卷积神经网络（CNN）背后的基础知识和创造动机之后，我们将在可用于对象检测的最流行的数据集之一上演示这一点。 我们还将看到CNN的初始层如何获得我们这类对象最基本的特征，但是最终的卷积层将获得更多语义级特征，这些特征是基于第一层中的那些基本特征构建的。<br>
本章将讨论以下主题：<br>
<br>
1.对象检测<br>
2.CIFAR-10目标检测在图像建模与训练中的应用<br>
## 对象检测
维基百科声明：

&emsp;&emsp;对象检测——计算机视觉领域的技术，用于在图像或视频序列中寻找和识别物体。尽管物体的图像在不同的视角有许多不同的大小和尺度，甚至当它们被平移或旋转时，可能有些不同，甚至当对象被部分遮挡时，人类仍然可以很容易识别图像中的大量物体。这一任务对计算机视觉系统仍然是一个挑战。这项任务的许多方法已经实施了几十年。
&emsp;&emsp;图像分析是深学习中最为突出的领域之一。图像易于生成和处理，并且它们正是用于机器学习的正确类型的数据：对于人类来说容易理解，但是对于计算机来说难。令人惊讶的是，图像分析在深神经网络的历史中起着关键的作用。<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter08/chapter08_image/1.jpg) <br>
随着自动汽车、人脸检测、智能视频监控和计数法的兴起，对快速准确的目标检测系统的需求越来越大。这些系统不仅要对图像中的对象识别和分类，而且还可以通过在它们周围绘制适当的框来定位其中的每一个。这使得对象检测比其传统的计算机视觉前身图像分类更难。<br>
在本章中，我们将研究对象检测-找出哪些对象在图像中。例如，想象一辆自动驾驶汽车需要检测道路上的其他车辆。对象检测有很多复杂的算法。它们通常需要庞大的数据集、非常深的卷积网络和长时间的训练。
## CIFAR-10 – 建模和训练
此示例显示如何在CIFAR-10数据集中使用卷积神经网络对图像进行分类。 我们将使用一个简单的卷积神经网络实现几个卷积层和完全连接的层。<br>
尽管网络架构非常简单，但是当它尝试识别CIFAR-10图像中的对象时，你就会知道他很好用。
### 调用包
我们为实现导入所有必需的包：
```
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from urllib.request import urlretrieve from os.path import isfile, isdir
from tqdm import tqdm import tarfile
import numpy as np import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer from sklearn.preprocessing import OneHotEncoder

import pickle
import tensorflow as tf
```
### 加载CIFIR-10 数据集
在这个实现中，我们将使用CIFIR-10，这是用于对象检测的最广泛使用的数据集之一。因此，让我们首先定义一个帮助类来下载和提取CIFIR-10数据集，如果它还没有被下载下来：
```
cifarlO_batches_dir_path = 'cifar–lO–batches–py' tar_gz_filename = 'cifar–lO–python.tar.gz'
class DLProgress(tqdm): last_block = O

def hook(self, block_num=l, block_size=l, total_size=None): self.total = total_size
self.update((block_num – self.last_block) * block_size) self.last_block = block_num

if not isfile(tar_gz_filename):
 

with DLProgress(unit='B', unit_scale=True, miniters=l, desc='CIFAR–lO Python Images Batches') as pbar:
urlretrieve(
'https://www.cs.toronto.edu/~kriz/cifar–lO–python.tar.gz', tar_gz_filename,
pbar.hook)

if not isdir(cifarlO_batches_dir_path):
with tarfile.open(tar_gz_filename) as tar: tar.extractall()
tar.close()

```
下载并提取CIFIR-10数据集后，您会发现它已经分成五批。CIFIR-10包含10个类别/类的图像：
1.airplane 
2.automobile 
3.bird
4.cat 
5.deer 
6.dog 
7.frog 
8.horse 
9.ship 
10.truck
