
# &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;第13章 自动编码器-特征提取和去噪
&emsp;&emsp;自动编码器网络现在是广泛使用的深度学习架构之一。 它主要用于高效解码任务的无监督学习。 它还可以通过学习特定数据集的编码或表示用于来降低维数。 在本章中使用自动编码器，我们将展示如何通过构建具有相同尺寸但噪声较小的另一个数据集来对数据集进行去噪。 为了在实践中使用这个概念，我们将从MNIST数据集中提取重要特征，并试着看看如何通过这个显着增强性能。<br>
&emsp;&emsp;&emsp;本章将介绍以下主题：<br>
&emsp;&emsp;&emsp;&emsp;---自动编码器简介<br>
&emsp;&emsp;&emsp;&emsp;---自动编码器实例<br>
&emsp;&emsp;&emsp;&emsp;---自动编码器架构<br>
&emsp;&emsp;&emsp;&emsp;---压缩MNIST数据集<br>
&emsp;&emsp;&emsp;&emsp;---卷积自动编码器<br>
&emsp;&emsp;&emsp;&emsp;---去噪自动编码器<br>
&emsp;&emsp;&emsp;&emsp;---自动编码器的应用<br>
