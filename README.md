# identify_cats-dogs
大一萌新的第一次尝试，[训练图片参考这个链接](https://rec.ustc.edu.cn/share/38b85c40-c925-11ec-9cb1-89ae9e4b462c) ,访问密码ustc1958.基本也是对着一些博客的代码修修补补完成科学研讨的课题。初步了解了深度学习领域的一些问题，在此实现了利用Alexnet网络识别猫狗的功能。


### 什么是神经网络？

​	神经网络，也称为人工神经网络 (ANN) 或模拟神经网络 (SNN)，是机器学习的子集，并且是深度学习算法的核心。其名称和结构是受人类大脑的启发，模仿了生物神经元信号相互传递的方式。

​	人工神经网络 (ANN) 由节点层组成，包含一个输入层、一个或多个隐藏层和一个输出层。 每个节点也称为一个人工神经元，它们连接到另一个节点，具有相关的权重和阈值。 如果任何单个节点的输出高于指定的阈值，那么该节点将被激活，并将数据发送到网络的下一层。 否则，不会将数据传递到网络的下一层。

<img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png" alt="前馈神经网络的输入层、隐藏层和输出层的可视图" style="zoom:50%;" />

<center>图. 神经网络结构示意图</center>

​	将各个节点想象成其自身的[线性回归](https://www.ibm.com/cn-zh/analytics/learn/linear-regression)模型，由输入数据、权重、偏差（或阈值）和输出组成。

​	一旦确定了输入层，就会分配权重。 这些权重有助于确定任何给定变量的重要性，与其他输入相比，较大的权重对输出的贡献更大。 将所有输入乘以其各自的权重，然后求和。 之后，输出通过一个激活函数传递，该函数决定了输出结果。 如果该输出超出给定阈值，那么它将“触发”（或激活）节点，将数据传递到网络中的下一层。 这会导致一个节点的输出变成下一个节点的输入。 这种将数据从一层传递到下一层的过程规定了该神经网络为前馈网络。



## Pytorch

### 优点

​		首先想说一说Pytorch框架,班上其他大多数同学用的是keras和tensorflow，我上网还有咨询同学调研了一下几者的区别，keras上手容易，对新手比较友好，但封装性太强，导致灵活性和效率受限；tensorflow最为热门，并有广泛的开源社区，但是其接口频繁变动且部分API设计比较晦涩；比较之下Pytorch的优点就凸显出来了

* 简洁：PyTorch的设计追求最少的封装，尽量避免重复造轮子。不像TensorFlow中充斥着session、graph、operation、name_scope、variable、tensor、layer等全新的概念，PyTorch的设计遵循tensor→variable(autograd)→nn.Module 三个由低到高的抽象层次，分别代表高维数组（张量）、自动求导（变量）和神经网络（层/模块），而且这三个抽象之间联系紧密，可以同时进行修改和操作。
* 速度： PyTorch的灵活性不以速度为代价，在许多评测中，PyTorch的速度表现胜过TensorFlow和Keras等框架 。
* 易用：PyTorch是所有的框架中面向对象设计的最优雅的一个。PyTorch的面向对象的接口设计来源于Torch，而Torch的接口设计以灵活易用而著称，加上其代码的简洁性会使读者易于理解，下文我们还会介绍。
* 活跃的社区。



## 数据处理（picture-->tensor)

```python
from torch.utils.data import Dataset
from torchvision import datasets	#加载文本或者图像数据集,dateset这个接口（类）已经封装好了
from torch.utils.data import DataLoader
import torchvision.transforms as transforms#对图像的预处理都用torchvision.transforms包
```

通过我们的具体代码了解预处理中Pytorch中部分封装的作用，

要让PyTorch能读取自己的数据集，只需要两步：

<center>制作图片数据的索引-->构建Dataset子类</center>

```python
  class MyDataset(Dataset):
        def __init__(self, data, transform, loder):
            self.data = data
            self.transform = transform
            self.loader = loder
        def __getitem__(self, item):
            img, label = self.data[item]
            img = self.loader(img)
            img = self.transform(img)
            return img, label
        def __len__(self):
            return len(self.data)
```

​		然而，如何制作这个list呢，通常的方法是将图片的路径和标签信息存储在一个txt中，然后从该txt中读取。整个读取自己数据的基本流程就是：

1. 制作存储了图片的路径和标签信息的txt.
2. 将这些信息转化为list，该list每一个元素对应一个样本.
3. 通过get item函数，读取数据和标签，并返回数据和标签(label)。

数据增强，数据预处理：torchvision中的transforms都帮我们实现好了，直接调用即可。

```python
# 预处理图像
        transforms.CenterCrop(224),#中心裁剪
        transforms.Resize((224, 224)),#缩放
        transforms.ToTensor(),   # PIL->处理图片转化为tensor进行分析
        # 归一化将数据整合为（0.5均值，0.5标准差）正态分布的数据
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  
```

​		当自定义Dataset构建好，剩下的操作就交给DataLoader了。在DataLoader中，会触发Mydataset中的getiterm函数读取一个batch大小的图片的数据和标签，并返回tensor类型。

最后是装载数据时的处理

```python
	train = MyDataset(train_data, transform=transform, loder=Myloader)
    test_data = data3 + data4
    test = MyDataset(test_data, transform=transform, loder=Myloader)
    train_data = DataLoader(dataset=train, batch_size=5, shuffle=True, num_workers=0, pin_memory=True)
    test_data = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

```

> * batch_size(int,optional)：每批加载多少个样本；
> * shuffle(bool,optional)：设置为True时，在每个epoch对数据打乱；
> * num_workers(int,optional)：用于加载数据的子进程数，0表示数据将在主进程中加载；
> * pin_memory(bool,optional)：设置为True时，数据加载器会将张量复制到CUDA固定内存，然后再返回它们。



 ###  神经网络基本结构

​		一个完整的神经网络结构一共包含3个部分：输入层 (Input Layer)，输出层 (Output Layer)和隐藏层 (Hidden Layer)。图4就是一个简单的神经网络的示意图。从这张示意图中可以看出，一个神经网络的输入层和输出层只有一个，但隐藏层可以有很多。在相邻两个层之间，每个神经元之间都有联系，这个“联系”对应着相应神经元之间的权重。像这样层与层之间神经元相互连接，层内神经元互不连接，而且下一层神经元连接上一层所有的神经元的神经网络叫做全连接神经网络 (Deep Nerual Networks, DNN)。

 如果用全连接神经网络处理大尺寸图像具有三个明显的缺点：

1. 首先将图像展开为向量会丢失空间信息；

2. 其次参数过多效率低下，训练困难；

3. 同时大量的参数也很快会导致网络过拟合。

而使用卷积神经网络可以很好地解决上面的三个问题。

## 卷积神经网络的训练过程

​    卷积神经网络主要由这几类层构成：输入层、卷积层，ReLU层、池化（Pooling）层和全连接层（全连接层和常规神经网络中的一样）。通过将这些层叠加起来，就可以构建一个完整的卷积神经网络。在实际应用中往往将卷积层与ReLU层共同称之为卷积层，**所以卷积层经过卷积操作也是要经过激活函数的**。具体说来，卷积层和全连接层（CONV/FC）对输入执行变换操作的时候，不仅会用到激活函数，还会用到很多参数，即神经元的权值w和偏差b；而ReLU层和池化层则是进行一个固定不变的函数操作。卷积层和全连接层中的参数会随着梯度下降被训练，这样卷积神经网络计算出的分类评分就能和训练集中的每个图像的标签吻合了。

>  第一个阶段是数据由低层次向高层次传播的阶段，即前向传播阶段。

> 第二个阶段是，当前向传播得出的结果与预期不相符时，将误差从高层次向底层次进行传播训练的阶段，即反向传播阶段。

> 综上训练过程为：

1. 网络进行权值的初始化；
2. 输入数据经过卷积层、下采样层、全连接层的向前传播得到输出值；
3. 求出网络的输出值与目标值之间的误差；
4. 当误差大于我们的期望值时，将误差传回网络中，依次求得全连接层，下采样层，卷积层的误差。各层的误差可以理解为对于网络的总误差，网络应承担多少；当误差等于或小于我们的期望值时，结束训练。
5. 根据求得误差进行权值更新。然后在进入到第二步。

### 向前传播过程

1. 数据规则化	彩色图像的输入通常先要分解为R（红）G（绿）B（蓝）三个通道，其中每个值介于0~255之间。
2. 卷积运算（Convolution）
   . 激活	此处使用的激活函数是Relu函数。
   . 池化（Pooling）	池化的目的是提取特征，减少向下一个阶段传递的数据量。
   . 全连接（Fully-connected layer）对卷积结果展开

## Alexnet网络

#### **优点**:

**AlexNet将CNN用到了更深更宽的网络中,相比于以前的LeNet其效果分类的精度更高**

1. ReLU的应用

2. Dropout随机失活    随机忽略一些神经元,以避免过拟合。

3. 重叠的最大池化层

4. 提出了LRN（Local Response Normalization）层

   局部响应归一化,对局部神经元创建了竞争的机制,使得其中响应小的值变得更大,并抑制反馈较小的。

5. 数据增强（data augmentation）

   <img src="https://bkimg.cdn.bcebos.com/pic/10dfa9ec8a136327c1a18bc09d8fa0ec09fac7f1?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2U4MA==,g_7,xp_5,yp_5/format,f_auto" alt="img" style="zoom:100%;" />

   > 第6,7,8层是全连接层，每一层的神经元的个数为4096，最终输出softmax为1000,因为上面介绍过，ImageNet这个比赛的分类个数为1000。全连接层中使用了RELU和Dropout。

```python
# 5个卷积层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 3个全连接层
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
```

> **nn.Conv2d：**
>
> * in_channels：输入张量的通道数；out_channels：输出张量的通道数；
> * kernel_size：卷积核的大小，当卷积是方形时，只需要一个整数边长即可，卷积不是方形时，要输入一个元组表示高和宽
> * stride：卷积核在图像窗口上每次平移的间隔，即步长；
> * padding：设置在所有边界增加值为0的边距的大小；
> * in_features：输入的二维张量大小；out_features：输出的二维张量大小，也代表该全连接层的神经元个数；

注意：ReLU指激活函数

​	我们知道，如果直接将权重和与偏置之和作为输出结果，那么整个神经网络就是线性的，但是简单的线性关系很难描述实际中的各种复杂量因此需要寻找一种方法，让这个网络中的数据映射为非线性的，这样才能使得多层之间的数据传递变得有意义，从而增强网络结构的鲁棒性。正因为上面的原因，需要引入非线性函数作为激励函数，这样深层神经网络表达能力就更加强大（不再是输入的线性组合，而是几乎可以逼近任意函数）。

#### 常见的激活函数

<img src="https://github.com/poetrilin/identify_cats-dogs/img/jihuo.png" alt="前馈神经网络的输入层、隐藏层和输出层的可视图"  />

* Sigmoid函数$\sigma(x)=\frac 1{1+e^{-x}}$

* Tanh函数 $ tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}=2sigmoid(2x)-1$

* Relu函数$f(x)=max(0,x)$

* Softmax函数$softmax(z_i)=\frac{e^{z_i}}{\sum_{j=1}^ne^{z_j} }$, 其中为第i个节点的输出值，n为输出节点的个数，即分类的类别个数特点：可以将多分类的输出值转换为范围在[0, 1]和为1的概率分布，直观得到神经网络每个预测结果的 概率。 

对于线性函数而言，ReLU的表达能力更强，尤其体现在深度网络中；而对于非线性函数而言，ReLU由于非负区间的梯度为常数

1.采用sigmoid等函数，算激活函数时（指数运算），计算量大，反向传播求误差梯度时，求导涉及除法，计算量相对大，而采用ReLu激活函数，整个过程的计算量节省很多。

 2.，对于深层网络，sigmoid函数反向传播时，很容易就会出现梯度消失的情况（在sigmoid接近饱和区时，变换太缓慢，导数趋于0，这种情况会造成信息丢失），从而无法完成深层网络的训 练。 

3.ReLu会使一部分神经元的输出为0，这样就造成 网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。 

### 反馈训练

1.优化算法（optimizer）：随机梯度下降算法（Stochastic Gradient Descent,SGD）

设 $f_i(x)$是有关索引为i的训练数据样本的损失函数，$n$训练数据样本数，$x$是模型的参数向量，那么目标函数定义为

$$f(x)=\frac1n\sum_{i=1}^nf_i(x)$$,目标函数在x处的梯度计算为$\nabla f(x)=\frac1n\sum_{i=1}^n\nabla f_i(x)$,在随机梯度下降的每次迭代中，我们随机均匀采样一个样本索引   $i\in\{1,2,…，n\}$ ，并计算梯度$\nabla f_i(x)$来迭代x: 

​														$$x\leftarrow x-\eta \nabla f_i(x) $$				这里$\eta$ 是学习率,习率一般在0 ~1之间，学习率过小，会造成函数更新过慢，导致训练时间长；学习率过高，可能会因为步长过大而错过最小值，或者使结果在最小值之间来回震荡。因此选择合适的学习率，对于神经网络训练有着重要影响。

2.损失函数——categorical_crossentropy（交叉熵损失函数）:

$$loss(x,class)=-log\frac{e^{x[class]}}{\sum_{j=1}^ne^{x_j}}=log(\sum_{j=1}^ne^{x_j})-x[class]$$

下面定义训练函数,包括优化器，损失函数和评价指标，设置参数如下

```
# 初始化数据需求
learning_rate = 1e-3  # 学习速率
batch_size = 64  # 数据样本数
epochs = 10  # 迭代次数
model = AlexNet()  # 引入模型
loss_fn = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 优化器
```

```python
#训练函数
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 定义测试函数
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

> loss_fn：损失函数，衡量预测值和真实值之间的误差；
> optimizer：优化函数，通过有限次迭代模型参数来尽可能降低损失函数的值；
> optimizer.zero_grad()：梯度初始化为零；
> loss.backward()：反向传播计算得到每个参数的梯度值；
> optimizer.step()：通过梯度下降执行一步参数更新。

### 参数保存

```python
torch.save(model, 'model.pth')  # 保存模型,直接简单，但是占用内存大，速度慢
```

```python
net = torch.load('model.pth')
```

### 结果预测

```python
outputs = net(img)
#输出最大概率类别
indices = torch.max(outputs, 1)
```

##  
