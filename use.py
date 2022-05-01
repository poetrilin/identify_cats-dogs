from unicodedata import name  #通过字符查找名称
import torch
from PIL import Image
from torchvision import transforms

#自定义图像处理函数
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#加载全部模型（不只是参数权重）
net = torch.load('model.pth')

#打开图像
img = Image.open("C://Users//l//Desktop//sci cl//doge.jpg")

#进行处理
img = transform(img).unsqueeze(0)

#模型预测
outputs = net(img)

#输出最大概率类别
_, indices = torch.max(outputs, 1)

if indices.item() == 263:
    print('dog')
else:
    print('cat')