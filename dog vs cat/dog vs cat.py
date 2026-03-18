import torch
from torch import nn
from torchvision import transforms, datasets
from PIL import Image

my_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
    transforms.RandomRotation(15),          # 随机旋转+-15度
    transforms.ToTensor(),
    transforms.Normalize((0.4855, 0.4500, 0.4169), (0.2639, 0.2582, 0.2605))
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4855, 0.4500, 0.4169), (0.2639, 0.2582, 0.2605))
])
#dataset
full_dataset = datasets.ImageFolder(root='./data_dog-cat',transform=my_transforms)

# 划分比例 80%训练，20%测试
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# 偷梁换柱让测试集不用数据增强（保持中心裁剪）
test_dataset.dataset.transform = test_transforms

#dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=False)

print(full_dataset.class_to_idx)
#定义模型

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        # 增加网络深度：第三层卷积
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.relu = nn.ReLU()
        # 加入Dropout防止过拟合，随即丢弃30%的神经元
        self.dropout = nn.Dropout(p=0.3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(in_features=128,out_features=2) # 变为128通道

    def forward(self,x):
        x = self.pool_1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool_2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool_3(self.relu(self.bn3(self.conv3(x)))) # 走第三层

        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout(x) # 经过Dropout
        x = self.fc(x)
        return x

#损失计算
train_model = model()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(train_model.parameters(),lr=0.001)



for epoch in range(5):
    train_model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = train_model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 统计训练集准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    train_acc = 100 * correct / total
    
    # 每个 epoch 跑完测试集（验证集）准确率
    train_model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_images, val_labels in test_loader:
            val_outputs = train_model(val_images)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()
    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1:02d}: Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

#测试模型
img_path = r'C:\Users\zhangwencheng\Desktop\pycharm-pytorch\CIFAR-10\dog.jpg'
img = Image.open(img_path).convert('RGB')
# 注意：测试单张图片时，千万不要用带有 Random（如果训练包含了Random翻转裁剪）的 my_transforms
# 我们前面定义过 test_transforms，用它来保证测试的时候图片是正中央、不被奇葩扭曲的。
img_tensor = test_transforms(img).unsqueeze(0)

# 【关键错误修复】：确认你数据集里的文件夹真实的映射关系！
# dataset.class_to_idx 决定了 0 和 1 代表什么。通常按首字母排序。
# 如果你的文件夹叫 Cat 和 Dog，那么 0 就是 Cat，1 就是 Dog。
# 我们必须用原数据集自动生成的列表来做反向查询，防止猜反！
idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}

train_model.eval()
with torch.no_grad():
    output = train_model(img_tensor)
    _, predicted = torch.max(output, 1)

print(f"原始预测张量分布为: {output}")
print(f"预测类别的索引为: {predicted.item()}")
print(f"预测为：{idx_to_class[predicted.item()]}")

