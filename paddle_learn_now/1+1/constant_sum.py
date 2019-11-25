import paddle.fluid as fluid

# 定义两个张量的常量x1和x2，并指定它们的形状是[2, 2]，并赋值为1铺满整个张量，类型为int64.
x1 = fluid.layers.fill_constant(shape=[2, 2], value=1, dtype='int64')
x2 = fluid.layers.fill_constant(shape=[2, 2], value=1, dtype='int64')

# 将两个张量求和
y1 = fluid.layers.sum(x=[x1, x2])

# 创建一个使用CPU的执行器
place = fluid.CPUPlace()
exe = fluid.executor.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
# fetch_list参数的值是在解析器在run之后要输出的值
result = exe.run(program=fluid.default_main_program(),
                 fetch_list=[y1])
print(result)
