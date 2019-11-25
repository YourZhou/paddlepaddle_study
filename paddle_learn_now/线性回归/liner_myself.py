# 定义一个简单的线性网络，这个网络非常简单，
# 结构是：输出层-->>隐层-->>输出层，这个网络一共有2层，因为输入层不算网络的层数

import paddle.fluid as fluid
import paddle
import numpy as np

# 定义一个简单的线性网络
# 一个大小为100，激活函数是ReLU的全连接层和一个输出大小为1的全连接层，就这样构建了一个非常简单的网络
x = fluid.layers.data(name='x',shape=[13],dtype='float32')
hidden = fluid.layers.fc(input=x,size=100,act='relu')
net = fluid.layers.fc(input=hidden,size=1,act=None)

# 定义损失函数
y = fluid.layers.data(name='y',shape=[1],dtype='float32')
# 平方差损失函数(square_error_cost),else such as 叉熵损失函数(cross_entropy)
cost = fluid.layers.square_error_cost(input=net,label=y)
# fluid.layers.square_error_cost()求的是一个Batch的损失值，所以我们还要对他求一个平均值
avg_cost = fluid.layers.mean(cost)

# 在主程序（fluid.default_main_program）中克隆一个程序作为预测程序
# 复制一个主程序，方便之后使用
test_program = fluid.default_main_program().clone(for_test=True)

# 我们定义的网络结构，损失函数等等都是更加顺序记录到PaddlePaddle的主程序中的。
# 主程序定义了神经网络模型，前向反向计算，以及优化算法对网络中可学习参数的更新，是我们整个程序的核心，
# 这个是PaddlePaddle已经帮我们实现的了，我们只需注重网络的构建和训练即可

# 使用的随机梯度下降法（SGD），还有Momentum、Adagrad、Adagrad等等
# 定义优化方法
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01)
opts = optimizer.minimize(avg_cost)

# 创建一个使用CPU的执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义训练和测试数据
# 我们在定义网络的输入层时，shape是13，但是每条数据的后面12个数据是没意义的，
x_data = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).astype('float32')
y_data = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]]).astype('float32')
test_data = np.array([[7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).astype('float32')
# 这组数据是符合y = 2 * x + 1，但是程序是不知道的，

# 键值对的key对用的就是fluid.layers.data()中的name的值
for pass_id in range(10):
    train_cost = exe.run(program=fluid.default_main_program(),
                         feed={'x':x_data,'y':y_data},
                         fetch_list=[avg_cost])
    print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0]))

# 开始预测
result = exe.run(program=test_program,
                 feed={'x':test_data,'y':np.array([[0.0]]).astype('float32')},
                 fetch_list=[net])

print("当x为7.0时，y为：%0.5f:" % result[0][0][0])


