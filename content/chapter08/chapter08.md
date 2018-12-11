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
下载并提取CIFIR-10数据集后，您会发现它已经分成五批。CIFIR-10包含10个类别/类的图像：<br>
1.airplane <br>
2.automobile <br>
3.bird<br>
4.cat <br>
5.deer <br>
6.dog <br>
7.frog <br>
8.horse <br>
9.ship <br>
10.truck<br>
在深入研究网络核心之前，先对部分数据进行分析和预处理。
### 数据分析和准备
我们需要对数据集进行分析，并进行一些基本的预处理。因此，让我们首先定义一些辅助函数，这些函数将使我们能够从五个数据集加载特定的数据集，并输出关于这个批次及其示例的一些分析：
```
# Defining a helper function for loading a batch of images 
def load_batch(cifarlO_dataset_dir_path, batch_num):
    with open(cifarlO_dataset_dir_path + '/data_batch_' + str(batch_num), mode='rb') as file:
batch = pickle.load(file, encoding='latinl')

input_features = batch['data'].reshape((len(batch['data']), 3, 32,
32)).transpose(0, 2, 3, l) 
target_labels = batch['labels']

return input_features, target_labels
```
然后，我们定义一个函数，可以帮助我们显示特定批次中特定样本的统计数据：
```
#Defining a function to show the stats for batch ans specific sample def batch_image_stats(cifarlO_dataset_dir_path, batch_num, sample_num):

    batch_nums = list(range(l, 6))

    #checking if the batch_num is a valid batch number if batch_num not in batch_nums:
       print('Batch Num is out of Range. You can choose from these Batch nums: (}'.format(batch_nums))
       return None

input_features, target_labels = load_batch(cifarlO_dataset_dir_path, batch_num)

#checking if the sample_num is a valid sample number if not (O <= sample_num < len(input_features)):
print('(} samples in batch (}. (} is not a valid sample number.'.format(len(input_features), batch_num, sample_num))
return None

print('\n3tatistics of batch number (}:'.format(batch_num)) 
print('Number of samples in this batch:(('.format(len(input_features))) 
print('Per class counts of each Label:(('.format(dict(zip(*np.unique(target_labels, return_counts=True)))))

image = input_features[sample_num] label = target_labels[sample_num]
cifarlO_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print('\n3ample Image Number (}:'.format(sample_num)) 
print('3ample image – Minimum pixel value: (} Maximum pixel value:
(}'.format(image.min(), image.max()))
print('3ample image – 3hape: (}'.format(image.shape)) 
print('3ample Label – Label Id: (} Name: (}'.format(label,
cifarlO_class_names[label])) plt.axis('off') 
plt.imshow(image)
```
现在，我们可以使用此函数来处理我们的数据集和可视化特定图像：
```
# Explore a specific batch and sample from the dataset 
batch_num = 3
sample_num = 6
batch_image_stats(cifarlO_batches_dir_path, batch_num, sample_num)
```
输出如下：
```
3tatistics of batch number 3:
Number of samples in this batch: lOOOO
Per class counts of each Label: (O: 994, l: lO42, 2: 965, 3: 997, 4: 99O,
5: lO29, 6: 978, 7: lOl5, 8: 96l, 9: lO29}

3ample Image Number 6:
3ample image – Minimum pixel value: 3O Maximum pixel value: 242 3ample image – 3hape: (32, 32, 3)
3ample Label – Label Id: 8 Name: ship
```
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter08/chapter08_image/2.jpg) <br>
在继续前行并将数据集馈送到模型之前，我们需要将其归一化到0到1的范围。<br>
批量归一化优化网络培训。它已被证明有几个好处：<br>

&emsp;&emsp;1.更快的训练：每个训练步骤将更慢，因为在网络的正向通过期间需要额外的计算，而在网络的反向传播通过期间需要额外的超参数来训练。然而，它应该更快地收敛，所以培训应该更快地整体。<br>

&emsp;&emsp;2.更高的学习率：梯度下降算法通常需要较小的学习率才能使网络收敛到损失函数的最小值。随着神经网络越来越深，它们的梯度值在反向传播过程中变得越来越小，因此它们通常需要更多的迭代。使用批量标准化的想法允许我们使用更高的学习率，这进一步提高了网络训练的速度。<br>

&emsp;&emsp;3.易于初始化权重：权重初始化可能很困难，如果使用深层神经网络则更困难。批量归一化似乎可以让我们在选择初始起始权重时更加谨慎。<br>

因此，让我们继续定义一个函数，该函数负责规范化输入图像列表，使这些图像的所有像素值都在0到1之间：<br>
```
#Normalize CIFAR–l0 images to be in the range of [0,l]

def normalize_images(images):
# initial zero ndarray
normalized_images = np.zeros_like(images.astype(float))
# The first images index is number of images where the other indices indicates
# hieight, width and depth of the image num_images = images.shape[0]
# Computing the minimum and maximum value of the input image to do the normalization based on them
maximum_value, minimum_value = images.max(), images.min()
# Normalize all the pixel values of the images to be from 0 to l for img in range(num_images):
normalized_images[img,...] = (images[img, ...] – float(minimum_value)) / float(maximum_value – minimum_value)

return normalized_images
```
接下来，我们需要实现另一个辅助函数来编码输入图像的标签。在此函数中，我们将使用独热编码sklearn，其中每个图像标签由零向量表示，区别于此向量表示的图像的类索引<br>
输出向量的大小将取决于我们在数据集中具有的类的数量，对于CIFAR-10数据，这是10个类：
```
#encoding the input images. Each image will be represented by a vector of 
zeros except for the class index of the image
# that this vector represents. The length of this vector depends on number 
of classes that we have
# the dataset which is lO in CIFAR–lO

def one_hot_encode(images): 
    num_classes = lO
    #use sklearn helper function of OneHotEncoder() to do that encoder = OneHotEncoder(num_classes)
    #resize the input images to be 2D
    input_images_resized_to_2d = np.array(images).reshape(–l,l)
    one_hot_encoded_targets =
encoder.fit_transform(input_images_resized_to_2d)
    return one_hot_encoded_targets.toarray()
```
现在，是时候调用前面的辅助函数来进行预处理并保留数据集，以便我们以后可以使用它：
```
def preprocess_persist_data(cifarlO_batches_dir_path, normalize_images, one_hot_encode):
 num_batches = 5
 valid_input_features = []
 valid_target_labels = []

for batch_ind in range(l, num_batches + l):
   #Loading batch
   input_features, target_labels = load_batch(cifarlO_batches_dir_path, batch_ind)
   num_validation_images = int(len(input_features) * O.l)

   # Preprocess the current batch and perisist it for future use
   input_features = normalize_images(input_features[:–num_validation_images])
   target_labels = one_hot_encode( target_labels[:– num_validation_images])
   #Persisting the preprocessed batch
   pickle.dump((input_features, target_labels),open('preprocess_train_batch_' + str(batch_ind) + '.p', 'wb'))

   # Define a subset of the training images to be used for validating our model
   valid_input_features.extend(input_features[– num_validation_images:])
 

    valid_target_labels.extend(target_labels[–num_validation_images:])

# Preprocessing and persisting the validationi subset input_features = normalize_images( np.array(valid_input_features)) target_labels = one_hot_encode(np.array(valid_target_labels))
pickle.dump((input_features, target_labels), open('preprocess_valid.p', 'wb'))

#Now it's time to preporcess and persist the test batche
with open(cifarlO_batches_dir_path + '/test_batch', mode='rb') as file: test_batch = pickle.load(file, encoding='latinl')


test_input_features = test_batch['data'].reshape((len(test_batch['data']), 3, 32,
32)).transpose(O, 2, 3, l)
test_input_labels = test_batch['labels']

# Normalizing and encoding the test batch
input_features = normalize_images( np.array(test_input_features))
target_labels = one_hot_encode(np.array(test_input_labels))
pickle.dump((input_features, target_labels), open('preprocess_test.p',
'wb'))
# Calling the helper function above to preprocess and persist the training, validation, and testing set preprocess_persist_data(cifarlO_batches_dir_path, normalize_images, one_hot_encode)
```
因此，我们将预处理的数据保存到磁盘。我们还需要加载验证集，以便在训练过程的不同时期对运行训练模型：

`# Load the Preprocessed Validation data valid_input_features, valid_input_labels = pickle.load(open('preprocess_valid.p', mode='rb'))`
### 建立网络
&emsp;&emsp;现在是时候构建我们的分类应用程序的核心，这是该CNN架构的计算图，但是为了最大化这种实现的好处，我们不会使用TensorFlow层API。取而代之的是，我们将使用它的神经网络版本。<br>
&emsp;&emsp;因此，让我们首先定义模型输入占位符，该占位符将输入图像、目标类和dropout层的保持概率参数（这有助于我们通过删除一些连接来降低架构的复杂性，从而减少过拟合的可能性）：<br>
```
# Defining the model inputs def images_input(img_shape):
return tf.placeholder(tf.float32, (None, ) + img_shape, name="input_images")

def target_input(num_classes):

target_input = tf.placeholder(tf.int32, (None, num_classes), name="input_images_target")
return target_input

#define a function for the dropout layer keep probability def keep_prob_input():
return tf.placeholder(tf.float32, name="keep_prob")
```
接下来，我们需要使用tensorflow神经网络完成本来构建具有max pooling的卷积层：
```
# Applying a convolution operation to the input tensor followed by max pooling
def conv2d_layer(input_tensor, conv_layer_num_outputs, conv_kernel_size, conv_layer_strides, pool_kernel_size, pool_layer_strides):


  input_depth = input_tensor.get_shape()[3].value
  weight_shape = conv_kernel_size + (input_depth, conv_layer_num_outputs,)


  #Defining layer weights and biases
  weights = tf.Variable(tf.random_normal(weight_shape))
  biases = tf.Variable(tf.random_normal((conv_layer_num_outputs,)))

  #Considering the biase variable
  conv_strides = (l,) + conv_layer_strides + (l,)


  conv_layer = tf.nn.conv2d(input_tensor, weights, strides=conv_strides, padding='3AME')
  conv_layer = tf.nn.bias_add(conv_layer, biases) 
  conv_kernel_size = (l,) + conv_kernel_size + (l,)
 

  pool_strides = (l,) + pool_layer_strides + (l,)
  pool_layer = tf.nn.max_pool(conv_layer, ksize=conv_kernel_size, strides=pool_strides, padding='3AME')
  return pool_layer
```
正如您在前一章中看到的那样，max pooling操作的输出是4D张量，这与完全连接的层所需的输入格式不兼容。 因此，我们需要实现一个平坦层，将最大池化层的输出从4D转换为2D张量：
```
#Flatten the output of max pooling layer to be fing to the fully connected layer which only accepts the output
# to be in 2D
def flatten_layer(input_tensor):
return tf.contrib.layers.flatten(input_tensor)
```
下一步，我们需要定义一个辅助函数，使我们能够为我们的体系结构添加一个完全连接的层：
```
#Define the fully connected layer that will use the flattened output of the stacked convolution layers
#to do the actuall classification
def fully_connected_layer(input_tensor, num_outputs): 
  return tf.layers.dense(input_tensor, num_outputs)
```
最后，在使用这些辅助函数来创建整个体系结构之前，我们需要创建另一个辅助函数，该辅助函数将获取全连接层的输出，并产生对应于我们在数据集中的类数量的10个实值：
```
#Defining the output function
def output_layer(input_tensor, num_outputs):
return	tf.layers.dense(input_tensor, num_outputs)
```
所以，让我们继续并定义将所有这些碎片组合在一起并创建具有三个卷积层的CNN的功能。 其中每一个都遵循max pooling操作。 我们还将有两个全连接的层，其中每个层后面都有一个压差层，以降低模型复杂性并防止过度拟合。 最后，我们将使输出层生成10个实值向量，其中每个值表示每个类的得分是正确的：
```
def build_convolution_net(image_data, keep_prob):

# Applying 3 convolution layers followed by max pooling layers conv_layer_l = conv2d_layer(image_data, 32, (3,3), (l,l), (3,3), (3,3))
conv_layer_2 = conv2d_layer(conv_layer_l, 64, (3,3), (l,l), (3,3), (3,3))
conv_layer_3 = conv2d_layer(conv_layer_2, l28, (3,3), (l,l), (3,3), (3,3))
 

# Flatten the output from 4D to 2D to be fed to the fully connected layer flatten_output = flatten_layer(conv_layer_3)

# Applying 2 fully connected layers with drop out fully_connected_layer_l = fully_connected_layer(flatten_output, 64) fully_connected_layer_l = tf.nn.dropout(fully_connected_layer_l,
keep_prob)
fully_connected_layer_2 = fully_connected_layer(fully_connected_layer_l, 32)
fully_connected_layer_2 = tf.nn.dropout(fully_connected_layer_2, keep_prob)

#Applying the output layer while the output size will be the number of categories that we have
#in CIFAR–lO dataset
output_logits = output_layer(fully_connected_layer_2, lO)

#returning output 
return output_logits
```
让我们调用前面的辅助函数来构建网络并定义它的缺失和优化准则：
```
#Using the helper function above to build the network

#First off, let's remove all the previous inputs, weights, biases form the previous runs
tf.reset_default_graph()

# Defining the input placeholders to the convolution neural network input_images = images_input((32, 32, 3))
input_images_target = target_input(lO) keep_prob = keep_prob_input()

# Building the models
logits_values = build_convolution_net(input_images, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training logits_values = tf.identity(logits_values, name='logits')

# defining the model loss model_cost =
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_values
, labels=input_images_target))

# Defining the model optimizer
model_optimizer = tf.train.AdamOptimizer().minimize(model_cost)
 

# Calculating and averaging the model accuracy correct_prediction = tf.equal(tf.argmax(logits_values, l), tf.argmax(input_images_target, l))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='model_accuracy')
tests.test_conv_net(build_convolution_net)
```
现在我们已经建立了这个网络的计算体系结构，现在开始测试过程并查看结果
### 模式训练
因此，让我们定义一个辅助函数，使我们能够启动训练过程。此函数将输入图像，目标类的独热编码和保持概率值作为输入。然后，将这些值反馈到计算图并调用模型优化器：
```
#Define a helper function for kicking off the training process
def train(session, model_optimizer, keep_probability, in_feature_batch, target_batch):
session.run(model_optimizer, feed_dict=(input_images: in_feature_batch, input_images_target: target_batch, keep_prob:keep_probability})
```
我们需要在训练过程中的不同时间分步骤验证我们的模型，因此我们将定义一个辅助函数，该函数将在验证集上打印出模型的准确性：
```
#Defining a helper funcitno for print information about the model accuracy and it's validation accuracy as well
def print_model_stats(session, input_feature_batch, target_label_batch, model_cost, model_accuracy):
validation_loss = session.run(model_cost, feed_dict=(input_images: input_feature_batch, input_images_target: target_label_batch, keep_prob: l.O})
validation_accuracy = session.run(model_accuracy, feed_dict=(input_images: input_feature_batch, input_images_target: target_label_batch, keep_prob: l.O})
print("Valid Loss: %f" %(validation_loss)) print("Valid accuracy: %f" % (validation_accuracy))
```
我们还定义模型的参数，我们可以使用它来调整模型以获得更好的性能：
```
# Model Hyperparameters
num_epochs = lOO
batch_size = l28
keep_probability = O.5
```
现在，让我们开始培训过程，但仅针对一批CIFAR-10数据集，并查看基于此批次的模型准确性。<br>
然而，在此之前，我们需要定义一个辅助函数，它将加载批处理训练并将输入图像与目标类分开：
```
# 3plitting the dataset features and labels to batches
def batch_split_features_labels(input_features, target_labels, train_batch_size):
for start in range(O, len(input_features), train_batch_size): end = min(start + train_batch_size, len(input_features)) yield input_features[start:end], target_labels[start:end]

#Loading the persisted preprocessed training batches
def load_preprocess_training_batch(batch_id, batch_size): filename = 'preprocess_train_batch_' + str(batch_id) + '.p'
input_features, target_labels = pickle.load(open(filename, mode='rb'))

# Returning the training images in batches according to the batch size defined above
return batch_split_features_labels(input_features, target_labels, train_batch_size)
```
现在，让我们开始训练：
```
print('Training on only a 3ingle Batch from the CIFAR–lO Dataset...') with tf.3ession() as sess:

# Initializing the variables sess.run(tf.global_variables_initializer())

# Training cycle
for epoch in range(num_epochs): batch_ind = l

for batch_features, batch_labels in load_preprocess_training_batch(batch_ind, batch_size):
train(sess, model_optimizer, keep_probability, batch_features, batch_labels)

print('Epoch number (:>2}, CIFAR–lO Batch Number (}: '.format(epoch + l, batch_ind), end='')
print_model_stats(sess, batch_features, batch_labels, model_cost, accuracy)
 

Output:
.
.
.
Epoch number 85, CIFAR–lO Batch Number l: Valid Loss: l.49O792 Valid accuracy: O.55OOOO
Epoch number 86, CIFAR–lO Batch Number l: Valid Loss: l.487ll8 Valid accuracy: O.525OOO
Epoch number 87, CIFAR–lO Batch Number l: Valid Loss: l.3O9O82 Valid accuracy: O.575OOO
Epoch number 88, CIFAR–lO Batch Number l: Valid Loss: l.446488 Valid accuracy: O.475OOO
Epoch number 89, CIFAR–lO Batch Number l: Valid Loss: l.43O939 Valid accuracy: O.55OOOO
Epoch number 9O, CIFAR–lO Batch Number l: Valid Loss: l.48448O Valid accuracy: O.525OOO
Epoch number 9l, CIFAR–lO Batch Number l: Valid Loss: l.345774 Valid accuracy: O.575OOO
Epoch number 92, CIFAR–lO Batch Number l: Valid Loss: l.425942 Valid accuracy: O.575OOO

Epoch number 93, CIFAR–lO Batch Number l: Valid Loss: l.45lll5 Valid accuracy: O.55OOOO
Epoch number 94, CIFAR–lO Batch Number l: Valid Loss: l.3687l9 Valid accuracy: O.6OOOOO
Epoch number 95, CIFAR–lO Batch Number l: Valid Loss: l.336483 Valid accuracy: O.6OOOOO
Epoch number 96, CIFAR–lO Batch Number l: Valid Loss: l.383425 Valid accuracy: O.575OOO
Epoch number 97, CIFAR–lO Batch Number l: Valid Loss: l.378877 Valid accuracy: O.625OOO
Epoch number 98, CIFAR–lO Batch Number l: Valid Loss: l.34339l Valid accuracy: O.6OOOOO
Epoch number 99, CIFAR–lO Batch Number l: Valid Loss: l.3l9342 Valid accuracy: O.625OOO
Epoch number lOO, CIFAR–lO Batch Number l: Valid Loss: l.34O849 Valid accuracy: O.525OOO
```
如您所见，仅在单个批次上进行培训时，验证准确性并不高。 让我们看看验证准确性如何仅根据模型的完整培训流程进行更改：
```
model_save_path = './cifar–lO_classification'

with tf.3ession() as sess:
# Initializing the variables sess.run(tf.global_variables_initializer())
 

# Training cycle
for epoch in range(num_epochs):

# iterate through the batches num_batches = 5

for batch_ind in range(l, num_batches + l): for batch_features, batch_labels in
load_preprocess_training_batch(batch_ind, batch_size): train(sess, model_optimizer, keep_probability, batch_features,
batch_labels)

print('Epoch number(:>2}, CIFAR–lO Batch Number (}: '.format(epoch + l, batch_ind), end='')
print_model_stats(sess, batch_features, batch_labels, model_cost, accuracy)

# 3ave the trained Model saver = tf.train.3aver()
save_path = saver.save(sess, model_save_path)

Output:
.
.
.
Epoch number94, CIFAR–lO Batch Number 5: Valid Loss: O.3l6593 Valid accuracy: O.925OOO
Epoch number95, CIFAR–lO Batch Number l: Valid Loss: O.285429 Valid accuracy: O.925OOO
Epoch number95, CIFAR–lO Batch Number 2: Valid Loss: O.3474ll Valid accuracy: O.825OOO
Epoch number95, CIFAR–lO Batch Number 3: Valid Loss: O.232483 Valid accuracy: O.95OOOO
Epoch number95, CIFAR–lO Batch Number 4: Valid Loss: O.2947O7 Valid accuracy: O.9OOOOO
Epoch number95, CIFAR–lO Batch Number 5: Valid Loss: O.29949O Valid accuracy: O.975OOO
Epoch number96, CIFAR–lO Batch Number l: Valid Loss: O.3O2l9l Valid accuracy: O.95OOOO
Epoch number96, CIFAR–lO Batch Number 2: Valid Loss: O.347O43 Valid accuracy: O.75OOOO
Epoch number96, CIFAR–lO Batch Number 3: Valid Loss: O.25285l Valid accuracy: O.875OOO
Epoch number96, CIFAR–lO Batch Number 4: Valid Loss: O.29l433 Valid accuracy: O.95OOOO
Epoch number96, CIFAR–lO Batch Number 5: Valid Loss: O.286l92 Valid accuracy: O.95OOOO
Epoch number97, CIFAR–lO Batch Number l: Valid Loss: O.277lO5
 

Valid accuracy: O.95OOOO
Epoch number97, CIFAR–lO Batch Number 2: Valid Loss: O.3O5842 Valid accuracy: O.85OOOO
Epoch number97, CIFAR–lO Batch Number 3: Valid Loss: O.2l5272 Valid accuracy: O.95OOOO
Epoch number97, CIFAR–lO Batch Number 4: Valid Loss: O.3l376l Valid accuracy: O.925OOO
Epoch number97, CIFAR–lO Batch Number 5: Valid Loss: O.3l35O3 Valid accuracy: O.925OOO
Epoch number98, CIFAR–lO Batch Number l: Valid Loss: O.265828 Valid accuracy: O.925OOO
Epoch number98, CIFAR–lO Batch Number 2: Valid Loss: O.3O8948 Valid accuracy: O.8OOOOO
Epoch number98, CIFAR–lO Batch Number 3: Valid Loss: O.232O83 Valid accuracy: O.95OOOO
Epoch number98, CIFAR–lO Batch Number 4: Valid Loss: O.298826 Valid accuracy: O.925OOO
Epoch number98, CIFAR–lO Batch Number 5: Valid Loss: O.29723O Valid accuracy: O.95OOOO
Epoch number99, CIFAR–lO Batch Number l: Valid Loss: O.3O42O3 Valid accuracy: O.9OOOOO
Epoch number99, CIFAR–lO Batch Number 2: Valid Loss: O.3O8775 Valid accuracy: O.825OOO
Epoch number99, CIFAR–lO Batch Number 3: Valid Loss: O.225O72 Valid accuracy: O.925OOO
Epoch number99, CIFAR–lO Batch Number 4: Valid Loss: O.263737 Valid accuracy: O.925OOO
Epoch number99, CIFAR–lO Batch Number 5: Valid Loss: O.2786Ol Valid accuracy: O.95OOOO
Epoch numberlOO, CIFAR–lO Batch Number l: Valid Loss: O.2935O9 Valid accuracy: O.95OOOO
Epoch numberlOO, CIFAR–lO Batch Number 2: Valid Loss: O.3O38l7 Valid accuracy: O.875OOO
Epoch numberlOO, CIFAR–lO Batch Number 3: Valid Loss: O.244428 Valid accuracy: O.9OOOOO
Epoch numberlOO, CIFAR–lO Batch Number 4: Valid Loss: O.28O7l2 Valid accuracy: O.925OOO
Epoch numberlOO, CIFAR–lO Batch Number 5: Valid Loss: O.278625 Valid accuracy: O.95OOOO
```
### 测试模型
让我们根据CIFAR-10数据集的测试集部分测试训练的模型。 首先，我们将定义一个辅助函数，它将帮助我们可视化一些示例图像及其相应的真实标签的预测：
```
#A helper function to visualize some samples and their corresponding predictions
def display_samples_predictions(input_features, target_labels, samples_predictions):

num_classes = l0

cifarl0_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

label_binarizer = LabelBinarizer() label_binarizer.fit(range(num_classes))
label_inds = label_binarizer.inverse_transform(np.array(target_labels))

fig, axies = plt.subplots(nrows=4, ncols=2) fig.tight_layout()
fig.suptitle('3oftmax Predictions', fontsize=20, y=l.l)

num_predictions = 4 margin = 0.05
ind = np.arange(num_predictions)
width = (l. – 2. * margin) / num_predictions

for image_ind, (feature, label_ind, prediction_indicies, prediction_values) in enumerate(zip(input_features, label_inds, samples_predictions.indices, samples_predictions.values)):
prediction_names = [cifarl0_class_names[pred_i] for pred_i in prediction_indicies]
correct_name = cifarl0_class_names[label_ind]

axies[image_ind][0].imshow(feature) axies[image_ind][O].set_title(correct_name) axies[image_ind][O].set_axis_off()

axies[image_ind][l].barh(ind + margin, prediction_values[::–l], width) axies[image_ind][l].set_yticks(ind + margin) axies[image_ind][l].set_yticklabels(prediction_names[::–l]) axies[image_ind][l].set_xticks([0, 0.5, l.0])
```
现在，让我们恢复经过训练的模型，并对测试集进行测试：
```
test_batch_size = 64
save_model_path = './cifar–lO_classification'
#Number of images to visualize num_samples = 4

#Number of top predictions top_n_predictions = 4

#Defining a helper function for testing the trained model def test_classification_model():

input_test_features, target_test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
loaded_graph = tf.Graph()
with tf.3ession(graph=loaded_graph) as sess:

# loading the trained model
model = tf.train.import_meta_graph(save_model_path + '.meta') model.restore(sess, save_model_path)

# Getting some input and output Tensors from loaded model model_input_values = loaded_graph.get_tensor_by_name('input_images:O') model_target = loaded_graph.get_tensor_by_name('input_images_target:O') model_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:O') model_logits = loaded_graph.get_tensor_by_name('logits:O') model_accuracy = loaded_graph.get_tensor_by_name('model_accuracy:O')

# Testing the trained model on the test set batches test_batch_accuracy_total = O
test_batch_count = O

for input_test_feature_batch, input_test_label_batch in batch_split_features_labels(input_test_features, target_test_labels, test_batch_size):
test_batch_accuracy_total += sess.run( model_accuracy,
feed_dict=(model_input_values: input_test_feature_batch, model_target: input_test_label_batch, model_keep_prob: l.O})
test_batch_count += l

print('Test set accuracy: (}\n'.format(test_batch_accuracy_total/test_batch_count))

# print some random images and their corresponding predictions from the test set results
random_input_test_features, random_test_target_labels =
 

tuple(zip(*random.sample(list(zip(input_test_features, target_test_labels)), num_samples)))

random_test_predictions = sess.run( tf.nn.top_k(tf.nn.softmax(model_logits), top_n_predictions), feed_dict=(model_input_values: random_input_test_features, model_target:
random_test_target_labels, model_keep_prob: l.O})

display_samples_predictions(random_input_test_features, random_test_target_labels, random_test_predictions)

#Calling the function test_classification_model()

Output:
INFO:tensorflow:Restoring parameters from ./cifar–lO_classification Test set accuracy: O.754OOO796l783439
```
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter08/chapter08_image/3.jpg) <br>
让我们想象另一个例子来看到一些错误：<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter08/chapter08_image/4.jpg) <br>
现在，我们的测试精度大约为75％，对与我们所使用的简单CNN来说已经好了
## 总结
本章向我们展示了如何制作CNN来对CIFAR-10数据集中的图像进行分类。 测试集的分类准确度约为79％-80％。 还绘制了卷积层的输出，但是很难看到神经网络如何识别和分类输入图像。需要更好的可视化技术。<br>
下一章，我们将使用一个现代和令人兴奋的深学习实践，即转移学习。传输学习允许您使用数据密集型架构的深入学习与小数据集。<br>


学号|姓名|专业
-|-|-
201802110480|肖浒|计算机软件与理论
