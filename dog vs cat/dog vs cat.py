import torch
from torch import nn
from torchvision import transforms, datasets
from PIL import Image

# 1. 禁用调试输出，开启 CUDNN 性能基准（极大提升卷积速度）
torch.backends.cudnn.benchmark = True

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

#============================
# 定义模型 (必须放在外层，供子进程导入)
#============================
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2,stride=2) # 变成了 112x112
        self.bn1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2,stride=2) # 变成了 56x56
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2) # 变成了 28x28
        self.bn3 = nn.BatchNorm2d(num_features=128)

        # 【修复1】：增加第四层卷积，让参数继续降维，提取更深层的语义特征
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2) # 变成了 14x14
        self.bn4 = nn.BatchNorm2d(num_features=256)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4) # 加强一点 Dropout
        
        # 【修复2】：不要直接变成1x1，保留 2x2 的空间特征，然后将其展平去全连接层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        
        # 256通道 * 2 * 2 = 1024
        self.fc1 = nn.Linear(in_features=1024, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)

    def forward(self,x):
        x = self.pool_1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool_2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool_3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool_4(self.relu(self.bn4(self.conv4(x)))) # 走第四层

        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        # 全连接层的过渡
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # ============================
    # 2. 数据与设备初始化
    # ============================
    # 【修复3】：使用优雅的安全切分方式，避免破坏彼此的 transform
    full_dataset_train = datasets.ImageFolder(root='./data_dog-cat', transform=my_transforms)
    full_dataset_test  = datasets.ImageFolder(root='./data_dog-cat', transform=test_transforms)
    
    # 手动产生一份打乱的索引来划分验证集
    indices = torch.randperm(len(full_dataset_train)).tolist()
    train_size = int(0.8 * len(full_dataset_train))
    
    # 利用 Subset 分别绑定各自专用的 Transform 数据集
    train_dataset = torch.utils.data.Subset(full_dataset_train, indices[:train_size])
    test_dataset = torch.utils.data.Subset(full_dataset_test, indices[train_size:])
    
    # 类别映射依旧可用
    idx_to_class = {v: k for k, v in full_dataset_train.class_to_idx.items()}
    
    # dataloader (针对 5090 优化：大 batch_size，高 num_workers，开启 pin_memory)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    
    print(f"Class to idx mapping: {full_dataset_train.class_to_idx}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================
    # 3. 实例化模型并定义 AMP
    # ============================
    train_model = model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    # 【修复4】：降低从头训练的初始学习率，避免梯度直接爆炸掉入"瞎猜死区"
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0003, weight_decay=1e-5)
    
    # 【GPU利用率大杀器】：使用混合精度训练(AMP)
    scaler = torch.amp.GradScaler('cuda')

    # ============================
    # 4. 开始训练
    # ============================
    for epoch in range(50):
        train_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True) # 针对显存优化的梯度清空
            
            # 开启混合精度上下文
            with torch.amp.autocast('cuda'):
                outputs = train_model(images)
                loss = loss_fn(outputs, labels)
            
            # 使用 scaler 放缩 loss，反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
                val_images, val_labels = val_images.to(device, non_blocking=True), val_labels.to(device, non_blocking=True)
                
                # 推理时同样使用 AMP 加速
                with torch.amp.autocast('cuda'):
                    val_outputs = train_model(val_images)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1:02d}: Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # ============================
    # 5. 测试单张图片
    # ============================
    import os
    img_path = r'C:\Users\zhangwencheng\Desktop\pycharm-pytorch\CIFAR-10\dog.jpg'
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = test_transforms(img).unsqueeze(0)

        train_model.eval()
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            # 推理也走 AMP
            with torch.amp.autocast('cuda'):
                output = train_model(img_tensor)
            _, predicted = torch.max(output, 1)

        print(f"\n[Test Result] 原始预测张量: {output}")
        print(f"[Test Result] 预测类别索引: {predicted.item()}")
        print(f"[Test Result] 最终预测结果：{idx_to_class[predicted.item()]}")
    else:
        print(f"\nWarning: 测试图片不存在，跳过测试 -> {img_path}")

