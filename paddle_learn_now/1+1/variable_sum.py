import paddle.fluid as fluid
import numpy as np

# 定义两个张量
a = fluid.layers.create_tensor(dtype='int64', name='a')
b = fluid.layers.create_tensor(dtype='int64', name='b')

# 将两个张量求和
y = fluid.layers.sum(x=[a, b])

# 创建一个使用CPU的执行器
place = fluid.CPUPlace()
exe = fluid.executor.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义两个要计算的变量
a1 = np.array([3, 2]).astype('int64')
b1 = np.array([4, 5]).astype('int64')
# feed参数，这个就是要对张量变量进行赋值的
out_a, out_b, result = exe.run(program=fluid.default_main_program(),
                               feed={'a': a1, 'b': b1},
                               fetch_list=[a, b, y])

print(out_a, " + ", out_b, " = ", result)
