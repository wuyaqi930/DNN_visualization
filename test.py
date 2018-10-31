# -*- coding: utf-8 -*-
#----------------导入相关package-----------------
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#----------------创建相关数据集(使用torch)----------------

#----------------定义参数----------------

#定义迭代次数
times = 20000

# 生成随机参数序列 
parameter_target = torch.rand(3,9)

# 生成随机输入变量--符合高斯分布--卡方分布
input_target = torch.rand(9,times) #设定了一千条数据

# 生成随机输出变量
output_target = torch.zeros(3,times) #设定了一千条数据

# 生成随机输出变量
data_plot = torch.zeros(3,times) #设定了一千条数据

#----------------进行计算----------------

# 根据矩阵乘法运算规则
output_target = torch.mm(parameter_target,input_target)

#---------------将numpy转化成tensor----------------

x= input_target
y= output_target

print(x)
print(y)


#----------------定义相关网络-----------------

# 首先，定义所有层属性
class Net(torch.nn.Module):  # 继承 torch 的 Module
    
    #定义该神经网络：4个全连接层，每层元素128个
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.fc1 = torch.nn.Linear(n_feature, n_hidden)   # 第一个全连接层
        self.fc2 = torch.nn.Linear(n_hidden, n_hidden)   # 第二个全连接层
        self.fc3 = torch.nn.Linear(n_hidden, n_hidden)   # 第三个全连接层
        self.fc4 = torch.nn.Linear(n_hidden, n_output)   # 第四个全连接层
    
    #定义钱箱网络
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = Net(n_feature=9, n_hidden=128, n_output=3)

print(net)

#----------------定义优化方法&定义损失函数-----------------

#使用“随机梯度下降法”进行参数优化
# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  # 传入 net 的所有参数, 学习率

#使用“ADAM”进行参数优化
optimizer = torch.optim.Adam(net.parameters(), lr=0.0003) # 传入 net 的所有参数, 学习率

#定义损失函数，计算均方差
#loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)
loss_func = torch.nn.L1Loss()      # 预测值和真实值的误差计算公式 (正常绝对值)


#----------------具体训练过程-----------------
for t in range(times):

    prediction = net( x[:,t] )     # input x and predict based on x

    loss = loss_func(prediction, y[:,t])     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients


    #计算误差百分数,并储存在data_plot当中
   
    percent = 100*(prediction - y[:,t])/y[:,t]
    data_plot[:,t] = percent

    #调试使用
    #print("\ndata_plot[:,t]")
    #print(data_plot[:,t])

    #print("\npercent")
    #print(percent)

#print("\ndata_plot")
#print(data_plot)

#计算3个输出的平均值（把每列加起来）

#将数据从tensor转化成numpy
data_plot_numpy = data_plot.detach().numpy()

#print("\ndata_plot_numpy")
#print(data_plot_numpy)

#取每个元素的绝对值
data_plot_numpy_abs = np.abs(data_plot_numpy)

#print("\ndata_plot_numpy_abs")
#print(data_plot_numpy_abs)

#调用sum函数，将每列数据加起来，求误差的平均数
average = np.sum(data_plot_numpy_abs, axis=0)/3 #要除3，表示加起来求平均

print("\naverage")
print(average)

#将误差平均值可视化
X = np.linspace(1, times, times, endpoint=True)

plt.xlim(19000, times)#设置XY轴的显示范围
plt.ylim(0, 6)

plt.plot(X,average)
plt.show()


##将X、Y、Z误差可视化
#for num in range (0,3):
#    X = np.linspace(1, times, times, endpoint=True)
#    plt.plot(X,data_plot[num,:].detach().numpy())
#    plt.show()

