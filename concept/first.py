import os

'''
Tensor
.requires_grad = True：追踪该张量的运算。当执行.backward()时计算梯度,梯度保存在.grad属性中
.detach()：取消梯度运算
with torch.no_grad():包裹上下文，取消梯度运算
Function：每个tensor的grad_fn属性保存一个Function的引用
如果你想计算一个张量的导数，直接调用.backward()函数(为标量),若张量为矢量，需告知shape
若设置不需要梯度，则反向传播无法到达，要实现反向传播，网络中至少又一个
'''

'''
自动求导
本质上是计算vector和jacobian矩阵的乘积
x.backward()若x不为标量,则需要传入vector的参数
'''

'''
是用nn.Module构建简单神经网络
继承nn.Module，包含层以及forward（input）函数
1. 定义网络，包含可训练的参数
2. 迭代数据集，形成输入
3. 网络中处理输入
4. 计算损失值
5. 反向椽笔
6. 更新权重

卷积核的结构(outdepth,indepth,height,wide)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)#six cores
        self.conv2 = nn.Conv2d(6, 16, 3)#sixteen cores
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))# (32-3+1)/2
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)# (15-3+1)/2
        x = x.view(-1, self.num_flat_features(x))
        #function view is meant to reshape the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
#print(net)
input = torch.randn(1, 1, 32, 32)
out = net(input)

net.zero_grad()
out.backward(torch.randn(1,10))#backprops with random gradients because of no loss

#note: torch.nn only support a mini-batch
#for example: nn.Conv2d will take in a tensor of(batch_size,channel,height,width)
#if you have a single sample,just use input.unsqueeze(0) to add a fake dimension

'''
Loss Function
接受输入（output,target）
nn.MSELoss计算均方差
'''
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

#invoke loss.backward() to caculate grads

#use torch.optim
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your one update loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

