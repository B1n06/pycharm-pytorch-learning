import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
#处理数据
my_dataset=datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
my_dataloader=DataLoader(my_dataset,batch_size=32,shuffle=True)
#定义我的模型
class my_module(torch.nn.Module):
    def __init__(self):
        super(my_module,self).__init__()
        self.flatten=torch.nn.Flatten()
        self.fc1=torch.nn.Linear(28*28,128)
        self.fc2=torch.nn.Linear(128,10)
    def forward(self,x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))  # <--- 加这一行！
        x = self.fc2(x)
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