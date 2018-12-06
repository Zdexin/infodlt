# 第7章 卷积神经网络的介绍
&emsp;&emsp;在数据时代，卷积神经网络（cnn）是一种特殊的深度学习体系结构，它使用卷积运算来提取输入图像的相关解释特征。卷积层作为前馈神经网络进行连接，同时使用卷积操作模拟人脑进行识别。单个的外层神经元接受外界刺激并做出反应。特别的，生物界的成像问题会成为一大难题。但在此单元，我们会学习到如何使用卷积神经网络来发现图像中的图案。
以下内容会出现在此章中：
1.	卷积运算
2.	推动发展
3.	卷积神经网络中的不同层
4.	卷积神经网络基础实例：MNIST数字分类
## 一.卷积运算
&emsp;&emsp;卷积计算在计算机视觉领域得到了广泛的应用，其性能超过了我们所使用的大多数传统的计算机视觉技术。卷积神经网络结合了著名的卷积运算和神经网络，因此得名卷积神经网络。因此，在深入学习卷积神经网络之前，我们将介绍卷积运算，看看它是如何工作的。<br>
&emsp;&emsp;卷积操作的主要目的是从图像中提取信息或特征。任何图像都可以被视为一个数值矩阵，该矩阵中的一组特定值将形成一个特征。卷积操作的目的是扫描该矩阵并尝试提取该图像的相关或解释性特征。例如，考虑一个5乘5的图像，其对应的强度或像素值显示为0和1：<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/1.jpg) <br>
&emsp;&emsp;并考虑以下3*3的矩阵：<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/2.jpg) <br>
我们可以使用3 x 3图像对5 x 5图像进行卷积，如下所示 :<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/3.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/4.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/5.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/6.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/7.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/8.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/9.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/10.jpg) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://github.com/computeryanjiusheng2018/infodlt/blob/master/content/chapter07/11.jpg) <br>
上图可归纳如下。 为了使用3 x 3卷积内核对原始5 x 5图像进行卷积，我们需要执行
以下操作：
1.	使用橙色矩阵扫描原始绿色图像，每次只移动1个像素（步幅）
2.	对于橙色图像的每个位置，我们在橙色矩阵和绿色矩阵中的相应像素值之间进行逐元素乘法
3.	将这些逐元素乘法运算的结果加在一起以获得单个整数，该整数将在输出粉色矩阵中形成单个值<br>
&emsp;&emsp;从上图中可以看出，橙色3乘3矩阵仅在每次移动（步幅）中操作一次原始绿色图像的一部分，或者只出现一次。
 所以，让我们把以上操作运用到卷积神经网络中：
1.	橙色3 x 3矩阵称为kernel（卷积核）, feature detector,或则filter(过滤器)
2.	结果矩阵被称为feature map（特征图）<br>
&emsp;&emsp;因为我们基于原始输入图像中的卷积核和相应像素之间的元素乘法得到feature map，所以改变kernel或filter的值将每次给出不同的feature map。<br>
&emsp;&emsp;因此，我们有理由相信需要在卷积神经网络的训练过程中自己找出特征检测器的值，但在这里并不是这样的。 CNN在学习过程中找出这些数字。 因此，如果我们有更多filter，则意味着我们可以从图像中提取更多功能。<br>
&emsp;&emsp;在进入下一节之前，让我们介绍一些通常在CNN上下文中使用的术语：<br>
&emsp;&emsp;stride：我们之前简要地提到了这个术语。 通常，stride是我们在输入矩阵的像素上移filter weigh的像素数。 例如，步幅1意味着在对输入图像进行卷积时将filter移动一个像素，而步幅2意味着在对输入图像进行卷积时将filter移动两个像素。 我们的步幅越大，生成的特征映射越小。<br>
&emsp;&emsp;Zero-padding：如果我们想要包含输入图像的边框像素，那么我们的部分滤镜将位于输入图像之外。 Zero-padding是将边界周围填充0，直至满足要求。<br>
### 推动<br>
&emsp;&emsp;传统的计算机视觉技术用于执行大多数计算机视觉任务，例如检测目标和分割对象。这些传统的计算机视觉技术的性能很好，但它从未真正使用，例如自动驾驶汽车。2012年，Alex Krizhevsky 介绍了CNN，它通过将对象分类错误率从26％降低到到15％，在ImageNet竞赛中取得了突破。自此之后，CNN已经被广泛使用，并且已经发现了不同的变化。它甚至在ImageNet 竞赛中的正确率超过人类识别，如下图所示：

