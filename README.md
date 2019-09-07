# Note-for-DRIT
make some notes for the paper called DRIT

论文名称：Diverse Image-to-Image Translation via Disentangled Representations

论文地址：[https://arxiv.org/abs/1808.00948](https://arxiv.org/abs/1808.00948)

代码地址：[https://github.com/HsinYingLee/DRIT](https://github.com/HsinYingLee/DRIT)

与我们所熟知的Pix2Pix，CycleGAN一样，这篇文章所做的工作也是通过GAN实现I2I(image-to-image)。那么它们的区别是什么呢？为此，作者专门贴了一张图来说明它们之间的区别。

![comparison](image/table.png)

从上表可以看出，最早的Pix2Pix模型使用的是成对的数据来训练，然而实际情况中成对的数据难以获得或是获得的代价很大，这给Pix2Pix的应用带来了一定局限性；CycleGAN虽然解决了不成对数据的转换问题，但是它和Pix2Pix一样，都是单一态输出，即对于一幅输入图片，只有一种风格的图片输出。之后的UNIT和BicycleGAN也是如此，在二者之间不能均衡。所以，这篇文章的贡献就很明显了：第一，在缺少成对数据的情况下仍然能够实现I2I；第二，对单一的输入有着不同风格的多模态输出。

为了得到多模态的输出，这篇文章用到了一个叫Disentangled Representation的方法，即解耦表示，这是个什么东东呢？通常，我们学到的特征是混杂在一起的，这些特征在数据空间中以一种复杂的无序的方式进行编码，但是如果这些特征是可分解的，那么这些特征将具有更强的解释性，我们将更容易利用这些特征进行编码。其实，在GAN中，由于生成器的输入z是一个连续的噪声信号，并且没有任何约束，导致GAN无法利用这个z，并且无法将z的具体维度和生成数据的语义特征对应起来，不具有可解释性（比如对于手写数字问题，我们不知道什么样的噪声生成1，什么样的噪声生成数字2，影响生成数字大小，角度的是z的哪一维特征）。也正因为如此，GAN网络有着令人诟病的模式崩溃问题(mode collapse)，生成样本类型单一。InfoGAN是第一个提出用解耦表示来解决这个问题的。它把输入到生成器的噪声分为两部分：输入的普通噪声向量z和对应的语义向量c。但如果直接这样作为网络的输入，生成器还是会忽略隐藏编码c的作用，或者看成z与c相互独立。于是，InfoGAN提出应该最大化隐藏编码c与生成样本G(z,c)之间的互信息，这样c才能更好的影响到对应的生成样本，增加生成样本的多样性。

![Disentangled Representation](C:\Users\krlab\Documents\notes\Note-for-DRIT\image\DR.jpg)

