import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# 数据预处理和载入
# 以RGB形式读出图像
def Myloader(path):
    return Image.open(path).convert('RGB')

# 得到一个包含路径与标签的列表
def init_process(path, lens):
    data = []
    name = find_label(path)
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])

    return data


# 自定义dataset
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


# 判断是dog还是cat
def find_label(str):
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        if str[i] == '%' and str[i - 1] == '.':
            last = i - 1
        if (str[i] == 'c' or str[i] == 'd') and str[i - 1] == '/':
            first = i
            break

    name = str[first:last]
    if name == 'dog':
        return 1
    else:
        return 0


# 加强版dataloader
def load_data():
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # 预处理图像
        transforms.CenterCrop(224),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),   # PIL->处理图片转化为tensor进行分析
        # 归一化将数据整合为（0.5均值，0.5标准差）正态分布的数据
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  
    ])
   
    path1 = 'cats_and_dogs_filtered\\train\\cats\\cat.%d.jpg'
    data1 = init_process(path1, [0, 1000])
    path2 = 'cats_and_dogs_filtered\\train\\dogs\\dog.%d.jpg'
    data2 = init_process(path2, [0, 1000])
    path3 = 'cats_and_dogs_filtered\\validation\\cats\\cat.%d.jpg'
    data3 = init_process(path3, [2000, 2500])
    path4 = 'cats_and_dogs_filtered\\validation\\dogs\\dog.%d.jpg'
    data4 = init_process(path4, [2000, 2500])

    train_data = data1 + data2

    train = MyDataset(train_data, transform=transform, loder=Myloader)

    test_data = data3 + data4
    test = MyDataset(test_data, transform=transform, loder=Myloader)

    train_data = DataLoader(dataset=train, batch_size=5, shuffle=True, num_workers=0, pin_memory=True)
    test_data = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    return train_data, test_data


# 建立神经网络模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


# 定义训练函数
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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


# 初始化数据需求
learning_rate = 1e-3  # 学习速率
batch_size = 64  # 数据样本数
epochs = 10  # 迭代次数
model = AlexNet()  # 引入模型
loss_fn = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 优化器

# 载入数据
train_dataloader, test_dataloader = load_data()
# 开始训练
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# 保存结果
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')  # 保存参数
torch.save(model, 'model.pth')  # 保存模型
