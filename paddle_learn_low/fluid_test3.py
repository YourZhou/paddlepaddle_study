# 给定一组数据 <X,Y>，求解出函数 f，使得 y=f(x)，
# 其中X,Y均为一维张量。最终网络可以依据输入x，
# 准确预测出y_predict。

import paddle.fluid as fluid
import numpy as np

# 定义X数值
train_data = np.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
# 定义期望预测的真实值y_true
y_true = np.array([[2.0], [4.0], [6.0], [8.0]]).astype('float32')
#定义输入数据类型
x = fluid.layers.data(name="x",shape=[1],dtype='float32')
y = fluid.layers.data(name="y",shape=[1],dtype='float32')
#搭建全连接网络
y_predict = fluid.layers.fc(input=x,size=1,act=None)

#定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_cost = fluid.layers.mean(cost)

#定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)

cpu = fluid.core.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())

for i in range(10000):
    outs = exe.run(
        feed={'x':train_data,'y':y_true},
        fetch_list=[y_predict.name,avg_cost.name]
    )
    if i%100 == 0:
        print(outs)

print(outs)
