import torchvision
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.optim as optim

#定义transform
train_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(15),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
text_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#整理数据
train_set = torchvision.datasets.CIFAR10(root='./data_CIFAR10', train=True,transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root='./data_CIFAR10', train=False,transform=text_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=32,shuffle=True,num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=32,shuffle=True,num_workers=0)

#定义模型
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # Block 1: 提取浅层特征 (输入 3通道，输出 32通道)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 图片尺寸从 32x32 变成 16x16

        # Block 2: 提取中层特征 (输出 64通道)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 图片尺寸从 16x16 变成 8x8

        # Block 3: 提取高层特征 (输出 128通道)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # 图片尺寸从 8x8 变成 4x4

        # 全连接层分类器
        self.flatten = nn.Flatten()
        # 经过三次 2x2 池化，32 -> 16 -> 8 -> 4，所以特征图大小是 4x4
        # 128个通道，所以平铺后的维度是 128 * 4 * 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 模型变深后容易过拟合，加入Dropout随机丢弃一半神经元
        self.fc2 = nn.Linear(256, 10)  # CIFAR-10 有10个类别

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model_1 = model()
#定义损失计算，反向传播，优化
loss_fn = CrossEntropyLoss()

optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50,eta_min=1e-5)
for epoch in range(50):
    model_1.train()
    total_loss = 0
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model_1(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"第{epoch+1}次循环完毕，当前损失{avg_loss:.4f}")
    model_1.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for text_images,text_targets in test_loader:
            outputs = model_1(text_images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == text_targets).sum().item()
            total_samples += len(text_targets)
    real_accuracy = total_correct / total_samples
    print(f"第{epoch+1}次循环完毕，当前正确率{real_accuracy}")
    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    print(f"第{epoch+1}次current_lr:{current_lr:.6f}")