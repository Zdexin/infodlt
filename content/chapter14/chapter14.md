# &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;第14章 生成对抗网络
&emsp;&emsp;生成式对抗网络（GAN）是一种深层神经网络体系结构，它由两个相互对立的网络（因此称为对抗网络）组成。<br>
&emsp;&emsp;在2014年蒙特利尔大学的Ian Good.和其他研究人员（包括Yoshua Bengio）的一篇论文（https://arxiv.org/abs/l4O6.266l） 中介绍了GAN。Facebook的人工智能研究主管YannLeCu提到GAN，称对抗性训练是过去10年机器学习中最有趣的想法。<br>
&emsp;&emsp;GAN的潜力是巨大的，因为它们可以学习模拟数据的任何分布。也就是说，GAN可以被教导在任何领域创造出与我们自己极其相似的世界：图像、音乐、演讲或散文。在某种意义上，他们是机器人艺术家，他们的作品令人印象深刻（https://www.nytimes.com/2Ol7/O8/l4/././google–ai–how–create–new–.–andnew–artists–project–magenta.html） ，并且也令人印象深刻。<br>
&emsp;&emsp;本章将讨论以下主题：<br>
&emsp;&emsp;&emsp;	直观的介绍<br>
&emsp;&emsp;&emsp;	GAN的简单实现<br>
&emsp;&emsp;&emsp;	深度卷积GAN<br>
## 直观的介绍<br>
&emsp;&emsp;在这一节中，我们将以非常直观的方式介绍GAN。为了了解GAN是如何工作的，我们将采用一个假想的场景来获得派对的门票。<br>
&emsp;&emsp;故事开始于一个非常有趣的聚会或活动正在某处举行，你非常有兴趣参加它。你很晚才听说这件事，所有的票都卖光了，但是你会尽一切努力去参加聚会的。所以你想出一个主意！您将尝试伪造一张票，需要完全与原始的一样，或非常，非常相似。但是因为生活不容易，还有一个挑战：你不知道原票是什么样子的。因此，根据你参加这类派对的经验，你开始想象门票的样子，并开始根据你的想象来设计门票。<br>
&emsp;&emsp;您将尝试设计门票，然后前往该事件，并显示门票给安全人员。希望他们会相信并让你参与进来。但是您不想多次向保安人员展示您的门票，因此您决定向朋友寻求帮助，朋友将根据您对原始票的初步猜测，并将其展示给保安人员。如果他们不让他进去，他会根据看到一些人拿着实际票进来的情况，给你一些关于票可能看起来什么样子的信息。你会根据你朋友的评论修改门票，直到保安让他进来。在这一点上，仅在这一点上，只有，你会设计另一个完全相同的外观，让自己进入。<br>
&emsp;&emsp;确实，想想这个故事有多不现实，但是GAN的工作方式与这个故事非常相似。GAN是当今非常流行的，人们正在将它们用于计算机视觉领域的许多应用。<br>
&emsp;&emsp;有许多有趣的应用程序可以使用GAN，我们将实现并提及其中的一些。在GAN中，有两个主要组件在许多计算机视觉领域取得了突破。第一个组件称为生成器，第二个组件称为鉴别器：<br>
&emsp;&emsp;&emsp;	生成器将尝试从特定的概率分布生成数据样本，这与试图复制事件票证的家伙非常相似<br>
&emsp;&emsp;&emsp;	鉴别器将判断（像那些试图在票上发现缺陷的安全人员一样，以决定它是否是原创的还是伪造的）它的输入是来自原始的训练集（原创的票）还是来自生成器部分（由试图复制orig的家伙设计）。客票）：<br>
![image001](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter14/chapter14_image/image001.png)<br>
图1：GANS——通用体系结构<br>
## GAN的简单实现<br>
&emsp;&emsp;从伪造门票的故事到一个事件，GAN的想法似乎非常直观。因此，为了清楚地理解GAN如何工作以及如何实现它们，我们将在MNIST数据集上演示GAN的简单实现。<br>
&emsp;&emsp;首先，我们需要构建GAN网络的核心，它由两个主要部分组成：生成器和鉴别器。如我们所说，生成器将尝试从特定的概率分布中想象或伪造数据样本；鉴别器可以访问并查看实际数据样本，它将判断发生器的输出在设计中是否存在缺陷，或者它非常接近原始数据样本。类似于事件的场景，生成器的整个目的是试图使鉴别器相信生成的图像来自真实数据集，因此试图欺骗他。训练过程具有类似于事件故事的结束；生成器将最终设法生成看起来非常类似于原始数据样本的图像：<br>
![image002](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter14/chapter14_image/image002.png)<br>
图2 .MNIST数据集的GAN通用体系结构<br>
&emsp;&emsp;任何GAN的典型结构如图2所示，将在MNIST数据集上进行训练。此图中的潜在样本部分是随机思想或向量，生成器使用该随机思想或向量来用假图像复制真实图像。<br>
正如我们提到的，鉴别器作为一个法官，它将试图从发生器设计的假图像中分离出真实的图像。因此，该网络的输出将是二进制的，其可以由具有0（意味着输入是假图像）和1（意味着输入是真实图像）的sigmoid函数表示。<br>
让我们继续执行并开始实现这个体系结构，看看它在MNIST数据集上如何执行。<br>
让我们开始为这个实现导入所需的库：<br>
```%matplotlib inline

import matplotlib.pyplot as plt
import pickle as pkl

import numpy as np
import tensorflow as tf
```
我们将使用MNIST数据集，因此我们将使用TensorFlow帮助程序来获取数据集并将其存储在某个地方：
```from tensorflow.examples.tutorials.mnist import input_data
mnist_dataset = input_data.read_data_sets('MNI3T_data')
Output:
Extracting MNI3T_data/train–images–idx3–ubyte.gz
Extracting MNI3T_data/train–labels–idxl–ubyte.gz
Extracting MNI3T_data/tlOk–images–idx3–ubyte.gz
Extracting MNI3T_data/tlOk–labels–idxl–ubyte.gz
```
## 模型输入<br>
&emsp;&emsp;在深入构建由生成器和鉴别器表示的GAN的核心之前，我们将定义计算图的输入。如图2所示，我们需要两个输入。第一个将是真实图像，将被馈送到鉴别器。另一个输入称为潜空间，它将被馈送到生成器，并用于生成其伪图像：<br>
```# Defining the model input for the generator and discrimator
def inputs_placeholders(discrimator_real_dim, gen_z_dim):
real_discrminator_input = tf.placeholder(tf.float32, (None,discrimator_real_dim),name="real_discrminator_input)
generator_inputs_z = tf.placeholder(tf.float32, (None, gen_z_dim), name="generator_input_z")
return real_discrminator_input, generator_inputs_z
```
![image003](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter14/chapter14_image/image003.png)<br>
图3：MNIST GaN实现的体系结构
&emsp;&emsp;现在是投入构建我们架构的两个核心组件的时候了。我们将从生成器部分开始。如图3所示，生成器将由至少一个隐藏层组成，该隐藏层将作为近似器工作。另外，我们将使用称为泄漏ReLU的函数，而不是使用正常的ReLU激活函数。这将允许梯度值在没有任何约束的情况下流经层（有关泄漏RelU的更多内容将在下一节介绍）。<br>
## 可变范围<br>
&emsp;&emsp;&emsp;变量范围是张力流的一个特性，它帮助我们完成以下操作：<br>
&emsp;&emsp;&emsp;&emsp;	确保我们稍后有一些命名约定来检索它们，例如，通过使它们从单词生成器或鉴别器开始，这将在网络训练期间帮助我们。我们可以使用名称范围特性，但是这个特性对于第二个目的没有帮助。<br>
&emsp;&emsp;&emsp;&emsp;	确保我们稍后有一些命名约定来检索它们，例如，通过使它们从单词生成器或鉴别器开始，这将在网络训练期间帮助我们。我们可以使用名称范围特性，但是这个特性对于第二个目的没有帮助。<br>
下面的语句将展示如何使用TensorFlow的变量范围特性：<br>
` with tf.variable_scope('scopeName', reuse=False):`<br>
`# Write your code here `
您可以在https://www..orflow.org//programmers 上阅读更多关于使用变量作用域特性的益处的信息。<br>
## 泄漏ReLU<br>
&emsp;&emsp;我们提到，我们将使用与ReLU激活函数不同的版本，该函数称为泄漏ReLU。ReLU激活函数的传统版本只是在输入值与零之间取最大值，通过其他方法将负值截断为零。泄漏ReLU（我们将要使用的版本）允许一些负值存在，因此命名为泄漏ReLU。<br>
&emsp;&emsp;有时，如果我们使用传统的ReLU激活函数，网络会陷入一种称为死亡状态的流行状态，这是因为网络对所有输出只产生零。
使用泄漏ReLU的想法是允许一些负值通过，从而防止这种死亡状态。<br>
&emsp;&emsp;使发生器工作的整个想法是从鉴别器接收梯度值，如果网络陷入死胡同，学习过程就不会发生。下图说明了传统ReLU及其泄漏版本之间的差异：<br>
![image004](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter14/chapter14_image/image004.png)<br>
图4：Relu函数<br>
![image005](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter14/chapter14_image/image005.png)<br>
图5：泄漏μRelu激活函数<br>
&emsp;&emsp;泄漏的RELU激活函数在TensorFlow中没有实现，所以我们需要自己实现。如果输入是正的，则该激活函数的输出将是正的，如果输入是负的，则该激活函数的输出将是受控的负值。我们将通过一个称为alpha的参数来控制负值，该参数将允许一些负值通过，从而引入网络的容差。<br>
下面的等式表示我们将实施的泄漏Relu：<br>
![image006](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter14/chapter14_image/image006.png)<br>
## 生成器<br>
&emsp;&emsp;MNIST图像在0和1之间归一化，其中sigmoid激活函数可以最佳地工作。但在实践中，发现tanh激活函数比任何其他函数都具有更好的性能。因此，为了使用tanh激活函数，我们需要将这些图像的像素值的范围重新缩放为-1到1：<br>
```def generator(gen_z, gen_out_dim, num_hiddern_units=l28, reuse_vars=False, leaky_relu_alpha=O.Ol):
''' Building the generator part of the network
Function arguments
–––––––––
gen_z : the generator input tensor
gen_out_dim : the output shape of the generator
num_hiddern_units : Number of neurons/units in the hidden layer
reuse_vars : Reuse variables with tf.variable_scope
leaky_relu_alpha : leaky ReLU parameter
Function Returns
–––––––
tanh_output, logits_layer:
'''
with tf.variable_scope('generator', reuse=reuse_vars):
# Defining the generator hidden layer
hidden_layer_l = tf.layers.dense(gen_z, num_hiddern_units, activation=None)
# Feeding the output of hidden_layer_l to leaky relu
hidden_layer_l = tf.maximum(hidden_layer_l,
leaky_relu_alpha*hidden_layer_l)
# Getting the logits and tanh layer output
logits_layer = tf.layers.dense(hidden_layer_l, gen_out_dim, activation=None)
tanh_output = tf.nn.tanh(logits_layer)
return tanh_output, logits_layer
```
现在我们已经准备好生成器部件了。让我们先来定义网络的第二个组成部分。<br>
## 鉴别器<br>
&emsp;&emsp;接下来，我们将构建生成对抗网络中的第二个主要组件，即鉴别器。鉴别器与发生器几乎相同，但是代替使用tanh激活函数，我们将使用sigmoid激活函数；它将产生一个二进制输出，该二进制输出将表示鉴别器在输入图像上的判断：<br>
```def discriminator(disc_input, num_hiddern_units=l28, reuse_vars=False, leaky_relu_alpha=O.Ol):
''' Building the discriminator part of the network Function Arguments
–––––––––
disc_input : discrminator input tensor
num_hiddern_units : Number of neurons/units in the hidden layer reuse_vars : Reuse variables with tf.variable_scope leaky_relu_alpha : leaky ReLU parameter
Function Returns
–––––––
sigmoid_out, logits_layer:
'''
with tf.variable_scope('discriminator', reuse=reuse_vars):
# Defining the generator hidden layer
hidden_layer_l = tf.layers.dense(disc_input, num_hiddern_units, activation=None)
# Feeding the output of hidden_layer_l to leaky relu hidden_layer_l = tf.maximum(hidden_layer_l,
leaky_relu_alpha*hidden_layer_l)
logits_layer = tf.layers.dense(hidden_layer_l, l, activation=None) sigmoid_out = tf.nn.sigmoid(logits_layer)
return sigmoid_out, logits_layer
```
## 构建GAN网络<br>
&emsp;&emsp;在定义了构建生成器和鉴别器部件的主要功能之后，是时候将它们堆叠起来并为此实现定义模型损失和优化器了。<br>
## 模型超参数<br>
&emsp;&emsp;我们可以通过改变下面的超参数集对GANS进行微调：<br>
```# size of discriminator input image
#28 by 28 will flattened to be 784
input_img_size = 784

# size of the generator latent vector gen_z_size = lOO

# number of hidden units for the generator and discriminator hidden layers gen_hidden_size = l28
disc_hidden_size = l28

#leaky ReLU alpha parameter which controls the leak of the function leaky_relu_alpha = O.Ol

# smoothness of the label label_smooth = O.l
```
## 生成器和鉴别器的定义<br>
&emsp;&emsp;在定义了用于生成伪MNIST映像（看起来与真实映像完全相同）的架构的两个主要部分之后，是时候使用我们迄今定义的功能来构建网络了。为了构建网络，我们将遵循以下步骤：<br>
&emsp;&emsp;1、	定义我们的模型的输入，它由两个变量组成。这些变量之一是真实图像，该图像将被馈送给鉴别器，而第二个变量是潜在的空间，供生成器用于复制原始图像。<br>
&emsp;&emsp;2.	调用定义的生成器函数来构建网络的生成器部分。<br>
&emsp;&emsp;3.	调用定义的鉴别器函数来构建网络的鉴别器部分，但是我们将调用这个函数两次。一个调用将用于实际数据，第二个调用将用于来自生成器的伪数据。<br>
&emsp;&emsp;4.	通过重用变量，保持真实图像和假图像的权重相同：<br>
```tf.reset_default_graph()

# creating the input placeholders for the discrminator and generator
real_discrminator_input, generator_input_z = inputs_placeholders(input_img_size, gen_z_size)
#Create the generator network
gen_model, gen_logits = generator(generator_input_z, input_img_size, gen_hidden_size, reuse_vars=False, leaky_relu_alpha=leaky_relu_alpha)

# gen_model is the output of the generator
#Create the generator network disc_model_real, disc_logits_real =
discriminator(real_discrminator_input, disc_hidden_size, reuse_vars=False, leaky_relu_alpha=leaky_relu_alpha) disc_model_fake, disc_logits_fake = discriminator(gen_model, disc_hidden_size, reuse_vars=True, leaky_relu_alpha=leaky_relu_alpha)
```
## 鉴频器和生成器损失<br>
&emsp;&emsp;在这一部分中，我们需要定义鉴别器和发电机损耗，这可以被认为是这个实现中最棘手的部分。<br>
&emsp;&emsp;我们知道，发生器试图复制原始图像，鉴别器作为判断，接收来自发生器的图像和原始输入图像。因此，在设计每一部分的损失时，我们需要针对两件事。<br>
&emsp;&emsp;首先，我们需要网络的鉴别器部分能够区分由生成器生成的伪图像和来自原始训练示例的真实图像。在训练期间，我们将向鉴别器部分提供一批，分为两类。第一类为来自原始输入的图像，第二类为来自由生成器生成的伪图像的图像。<br>
因此，鉴别器的最终一般损失将是其接受真实损失和检测假损失的能力的总和，那么最终的总损失将是：<br>
![image007](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter14/chapter14_image/image007.png)<br>
`tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_layer, labels=labels))`<br>
&emsp;&emsp;所以我们需要计算两个损失才能得出最终的鉴别器损失。
第一个损失disc_loss_real，将基于我们将从鉴别器和标签获得的logits值来计算，在本例中，这些值都是1，因为我们知道这个小批次中的所有图像都来自MNIST数据集的实际输入图像。为了提高模型在测试集上的泛化能力，给出更好的结果，人们发现，实际地将值从1改变到0.9更好。这种对标签的改变引入了一种称为标签平滑的东西：<br>
`labels = tf.ones_like(tensor) * (l – smooth)`<br>
&emsp;&emsp;对于鉴别器损失的第二部分，即鉴别器检测假图像的能力，损失将在我们将从鉴别器和标签得到的logits值之间；所有这些都是零，因为我们知道这个小批次的所有图像都来自生成器，而不是来自原始输入。<br>
&emsp;&emsp;既然我们已经讨论了鉴别器损失，我们还需要计算生成器损失。生成器损失将称为gen_loss，它将是disc_logits_fake(伪图像的鉴别器的输出)和标签(由于产生器试图说服鉴别器进行伪图像的设计，所以都是)之间的损失：<br>
```# calculating the losses of the discrimnator and generator disc_labels_real = tf.ones_like(disc_logits_real) * (l – label_smooth) disc_labels_fake = tf.zeros_like(disc_logits_fake)

disc_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_labels_real, logits=disc_logits_real)
disc_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_labels_fake, logits=disc_logits_fake)

#averaging the disc loss
disc_loss = tf.reduce_mean(disc_loss_real + disc_loss_fake)

#averaging the gen loss gen_loss = tf.reduce_mean(
tf.nn.sigmoid_cross_entropy_with_logits( labels=tf.ones_like(disc_logits_fake), logits=disc_logits_fake))
```
## 优化器<br>
&emsp;&emsp;最后，优化器部分！在本节中，我们将定义在训练过程中将使用的优化标准。首先，我们将分别更新生成器和鉴别器的变量，因此我们需要能够检索每个部分的变量。
对于第一个优化器，即生成器，我们将从计算图的可训练变量中检索以名称生成器开始的所有变量；然后我们可以通过引用其名称来检查哪个变量是哪个。我们还将对鉴别器变量进行同样的处理，方法是让所有变量都以鉴别器开始。之后，我们可以将希望优化的变量列表传递给优化器。<br>
&emsp;&emsp;因此，TensorFlow的变量范围特性使我们能够检索以某个字符串开始的变量，然后我们可以拥有两个不同的变量列表，一个用于生成器，另一个用于鉴别器：<br>
```# building the model optimizer learning_rate = O.OO2
# Getting the trainable_variables of the computational graph, split into Generator and Discrimnator parts
trainable_vars = tf.trainable_variables() gen_vars = [var for var in trainable_vars if var.name.startswith("generator")]
disc_vars = [var for var in trainable_vars if var.name.startswith("discriminator")]

disc_train_optimizer = tf.train.AdamOptimizer().minimize(disc_loss, var_list=disc_vars)
gen_train_optimizer = tf.train.AdamOptimizer().minimize(gen_loss, var_list=gen_vars)
```
## 模型训练<br>
&emsp;&emsp;现在让我们开始训练过程，看看GAN将如何设法生成类似于MNIST的图像：<br>
```train_batch_size = lOO num_epochs = lOO generated_samples = []
model_losses = []
saver = tf.train.3aver(var_list = gen_vars) with tf.3ession() as sess:
sess.run(tf.global_variables_initializer()) for e in range(num_epochs):
for ii in range(mnist_dataset.train.num_examples//train_batch_size):
input_batch = mnist_dataset.train.next_batch(train_batch_size)
# Get images, reshape and rescale to pass to D input_batch_images = input_batch[O].reshape((train_batch_size,
 
784))
 

input_batch_images = input_batch_images*2 – l
# 3ample random noise for G
gen_batch_z = np.random.uniform(–l, l, size=(train_batch_size,
 
gen_z_size))
# Run optimizers
_ = sess.run(disc_train_optimizer, feed_dict=(real_discrminator_input: input_batch_images, generator_input_z: gen_batch_z})
_ = sess.run(gen_train_optimizer, feed_dict=(generator_input_z: gen_batch_z})
# At the end of each epoch, get the losses and print them out train_loss_disc = sess.run(disc_loss, (generator_input_z:
gen_batch_z, real_discrminator_input: input_batch_images}) train_loss_gen = gen_loss.eval((generator_input_z: gen_batch_z}) print("Epoch (}/(}...".format(e+l, num_epochs),
"Disc Loss: (:.3f}...".format(train_loss_disc), "Gen Loss: (:.3f}".format(train_loss_gen))
# 3ave losses to view after training model_losses.append((train_loss_disc, train_loss_gen))
# 3ample from generator as we're training for viegenerator_inputs_zwing afterwards
gen_sample_z = np.random.uniform(–l, l, size=(l6, gen_z_size)) generator_samples = sess.run(
generator(generator_input_z, input_img_size,
reuse_vars=True),
feed_dict=(generator_input_z: gen_sample_z})
generated_samples.append(generator_samples) saver.save(sess, './checkpoints/generator_ck.ckpt')

# 3ave training generator samples
with open('train_generator_samples.pkl', 'wb') as f: pkl.dump(generated_samples, f)
Output:
Epoch 7l/lOO... Disc Loss: l.O78... Gen Loss: l.36l Epoch 72/lOO... Disc Loss: l.O37... Gen Loss: l.555 Epoch 73/lOO... Disc Loss: l.l94... Gen Loss: l.297 Epoch 74/lOO... Disc Loss: l.l2O... Gen Loss: l.73O Epoch 75/lOO... Disc Loss: l.l84... Gen Loss: l.425 Epoch 76/lOO... Disc Loss: l.O54... Gen Loss: l.534 Epoch 77/lOO... Disc Loss: l.457... Gen Loss: O.97l Epoch 78/lOO... Disc Loss: O.973... Gen Loss: l.688 Epoch 79/lOO... Disc Loss: l.324... Gen Loss: l.37O Epoch 8O/lOO... Disc Loss: l.l78... Gen Loss: l.7lO Epoch 8l/lOO... Disc Loss: l.O7O... Gen Loss: l.649 Epoch 82/lOO... Disc Loss: l.O7O... Gen Loss: l.53O Epoch 83/lOO... Disc Loss: l.ll7... Gen Loss: l.7O5 Epoch 84/lOO... Disc Loss: l.O42... Gen Loss: 2.2lO Epoch 85/lOO... Disc Loss: l.l52... Gen Loss: l.26O Epoch 86/lOO... Disc Loss: l.327... Gen Loss: l.3l2 Epoch 87/lOO... Disc Loss: l.O69... Gen Loss: l.759 Epoch 88/lOO... Disc Loss: l.OOl... Gen Loss: l.4OO Epoch 89/lOO... Disc Loss: l.2l5... Gen Loss: l.448 Epoch 9O/lOO... Disc Loss: l.lO8... Gen Loss: l.342 Epoch 9l/lOO... Disc Loss: l.227... Gen Loss: l.468 Epoch 92/lOO... Disc Loss: l.l9O... Gen Loss: l.328 Epoch 93/lOO... Disc Loss: O.869... Gen Loss: l.857 Epoch 94/lOO... Disc Loss: O.946... Gen Loss: l.74O Epoch 95/lOO... Disc Loss: O.925... Gen Loss: l.7O8 Epoch 96/lOO... Disc Loss: l.O67... Gen Loss: l.427 Epoch 97/lOO... Disc Loss: l.O99... Gen Loss: l.573 Epoch 98/lOO... Disc Loss: O.972... Gen Loss: l.884 Epoch 99/lOO... Disc Loss: l.292... Gen Loss: l.6lO Epoch lOO/lOO... Disc Loss: l.lO3... Gen Loss:l.736
```
在运行100个时期的模型后，我们有一个训练有素的模型，能够生成类似于我们输入到鉴别器的原始输入图像的图像：<br>
```fig, ax = plt.subplots()
model_losses = np.array(model_losses) plt.plot(model_losses.T[O], label='Disc loss') plt.plot(model_losses.T[l], label='Gen loss') plt.title("Model Losses")
plt.legend()
Output:
```
![image008](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter14/chapter14_image/image008.png)<br>
图6：鉴频器和生成器损失<br>
如上图所示，可以看到由鉴别器和生成器线表示的模型损失正在收敛。<br>
## 训练样本<br>
&emsp;&emsp;让我们测试模型的性能，甚至看看生成器的生成技能（为事件设计门票）在接近训练过程结束时如何得到增强：<br>
```def view_generated_samples(epoch_num, g_samples):
fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
print(gen_samples[epoch_num][l].shape)
for ax, gen_image in zip(axes.flatten(), g_samples[O][epoch_num]): ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
img = ax.imshow(gen_image.reshape((28,28)), cmap='Greys_r') return fig, axes
```
在绘制训练过程中最后一个时期的一些生成图像之前，我们需要在训练过程中加载包含每个时期生成的样本的持久文件：<br>
```# Load samples from generator taken while training with open('train_generator_samples.pkl', 'rb') as f:
gen_samples = pkl.load(f)
```
现在，让我们从训练过程的最后一个时期绘制16个生成的图像，看看生成器如何生成有意义的数字，如3,7和2：<br>
`_ = view_generated_samples(–l, gen_samples)`<br>
![image009](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter14/chapter14_image/image009.png)<br>
图7：最终训练时期的样本<br>
我们甚至可以在不同的时代看到发电机的设计技巧。 因此，让我们在每10个时期可视化由它生成的图像：<br>
```rows, cols = lO, 6
fig, axes = plt.subplots(figsize=(7,l2), nrows=rows, ncols=cols, sharex=True, sharey=True)

for gen_sample, ax_row in zip(gen_samples[::int(len(gen_samples)/rows)], axes):
for image, ax in zip(gen_sample[::int(len(gen_sample)/cols)], ax_row): ax.imshow(image.reshape((28,28)), cmap='Greys_r') ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
```
![image010](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter14/chapter14_image/image010.png)<br>
图8：当网络被训练时产生的图像，永远是10个时期<br>
如你所见，生成器的设计技巧以及生成假图像的能力最初非常有限，然后在训练过程结束时得到增强。<br>
## 生成器采样<br>
&emsp;&emsp;在上一节中，我们介绍了在这个GAN体系结构的训练过程中生成的一些示例。我们还可以通过加载我们已经保存的检查点来从生成器生成完全新的图像，并向生成器提供新的潜在空间，生成器可以使用该空间来生成新的图像：<br>
```# 3ampling from the generator
saver = tf.train.3aver(var_list=g_vars) with tf.3ession() as sess:
#restoring the saved checkpints
saver.restore(sess, tf.train.latest_checkpoint('checkpoints')) gen_sample_z = np.random.uniform(–l, l, size=(l6, z_size)) generated_samples = sess.run(
generator(generator_input_z, input_img_size,
reuse_vars=True),
feed_dict=(generator_input_z: gen_sample_z})
view_generated_samples(O, [generated_samples])
```
![image011](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter14/chapter14_image/image011.png)<br>
图9：来自生成器的样本<br>
&emsp;&emsp;在实现此示例时，您可以提出一些观察结果。在训练过程的最初阶段，生成器没有任何技能来生成与真实图像类似的图像，因为它不知道它们看起来像什么。甚至鉴别器也不知道如何区分由生成器生成的伪图像和。在训练开始时，出现两个有趣的情况。首先，生成器不知道如何创建像我们最初提供给网络的真实图像那样的图像。第二，鉴别器不知道真实图像和假图像的区别。<br>
&emsp;&emsp;稍后，生成器开始伪造在某种程度上有意义的图像，这是因为生成器将学习原始输入图像来自的数据分布。同时，该鉴别器将能够区分假图像和真实图像，并且它将在训练过程结束时被愚弄。<br>
## 总结<br>
GAN现在正被用于许多有趣的应用。GAN可用于不同的设置，例如半监督和非监督任务。此外，由于大量研究人员在GAN上工作，这些模型正在日益进步，并且它们生成图像或视频的能力越来越好。这些类型的模型可以用于许多有趣的商业应用，比如在Photoshop中添加一个插件，它可以接受一些命令，比如使我的微笑更有吸引力。它们也可以用于图像去噪。
