import torch
import os
import torch.nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
#处理数据
my_dataset=datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
my_dataloader=DataLoader(my_dataset,batch_size=32,shuffle=True)
#定义我的模型
class my_module(torch.nn.Module):
    def __init__(self):
        super(my_module,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc = torch.nn.Linear(in_features=64*7*7,out_features=10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,64*7*7)
        x=self.fc(x)
        return x
#定义完模型开始使用模型
model = my_module()
#计算误差
loss_fn = torch.nn.CrossEntropyLoss()
#反向传播
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
#优化
writer = SummaryWriter("logs")
for epoch in range(5):  # 训练 5 轮
    for batch_idx, (imgs, targets) in enumerate(my_dataloader):
        # 1. 梯度清零
        optimizer.zero_grad()

        # 2. 前向传播：模型算出的预测值
        outputs = model(imgs)

        # 3. 计算误差
        loss = loss_fn(outputs, targets)

        # 4. 反向传播：计算梯度
        loss.backward()

        # 5. 更新参数
        optimizer.step()

    print(f"第 {epoch + 1} 轮训练结束，当前 Loss: {loss.item():.4f}")
    writer.add_scalar("Training Loss", loss, epoch)
# 放在整个代码的最末尾
writer.close()
os.system("tensorboard --logdir=logs")
