import torch
from torch import nn
from torchvision.datasets import MNIST
import numpy as np
from torch.autograd import Variable

train_set = MNIST('./data', train=True, download=True)
test_set = MNIST('./data', train=False, download=True)
#观察一下数据
a_data, a_label = train_set[0]
print(a_data)
print(a_label)
#这里的读入的数据是 PIL 库中的格式，我们可以非常方便地将其转换为 numpy array
a_data = np.array(a_data, dtype='float32')
print(a_data.shape)


# 对于神经网络，我们第一层的输入就是 28 x 28 = 784，所以必须将得到的数据
# 我们做一个变换，使用 reshape 将他们拉平成一个一维向量
def get_data(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x
train_set = MNIST('./data', train=True, transform=get_data, download=True)
test_set = MNIST('./data', train=False, transform=get_data, download=True)
a, a_label = train_set[0]
print(a.shape)
print(a_label)
#使用pytorch自带的DataLoader定义一个数据迭代器
#使用这样的数据迭代器是非常有必要的，如果数据量太大，就无法一次将他们全部读入内存，
#所以需要使用 python 迭代器，每次生成一个批次的数据
from torch.utils.data import DataLoader
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)
a, a_label = next(iter(train_data))

# 打印出一个批次的数据
print(a.shape)
print(a_label.shape)

# 使用Sequential定义4层神经网络
net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)
# 交叉熵在 pytorch 中已经内置了，交叉熵的数值稳定性更差，所以内置的函数已经帮我们解决了这个问题
# 定义loss函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)

# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []
for e in range(20):
     train_loss = 0
     train_acc = 0
     net.train()
     for im, label in train_data:
          im = Variable(im)
          label = Variable(label)
          #前向传播
          out = net(im)
          loss = criterion(out, label)
          #反向传播
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          #记录误差
          train_loss += loss.item()
          #计算分类的准确率
          _, pred = out.max(1)
          num_correct = (pred==label).sum().item()
          acc = num_correct / im.shape[0]
          train_acc += acc
     losses.append(train_loss / len(train_data))
     acces.append(train_acc / len(train_data))
     #在测试集上检验效果
     eval_loss = 0
     eval_acc = 0
     net.eval()
     for im, label in test_data:
          im = Variable(im)
          label = Variable(label)
          out = net(im)
          loss = criterion(out, label)
          #记录误差
          eval_loss += loss.item()
          #记录准确率
          _, pred = out. max(1)
          num_correct = (pred==label).sum().item()
          acc = num_correct / im.shape[0]
          eval_acc += acc
     eval_losses.append(eval_loss / len(test_data))
     eval_acces.append(eval_acc / len(test_data))
     print('epoch:{}, Train Loss:{:.6f}, Train Acc:{:.6f}, Eval Loss:{:.6f}, Eval Acc:{:.6f}'
     .format(e, train_loss/len(train_data), train_acc/len(train_data),
             eval_loss/len(test_data), eval_acc/len(test_data)))
