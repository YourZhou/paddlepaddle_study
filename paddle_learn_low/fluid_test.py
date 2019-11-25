import paddle.fluid as fluid

# 定义数组维度及数据类型，可以修改shape参数定义任意大小的数组
data = fluid.layers.ones(shape=[5], dtype='int64')
# 在CPU上执行运算
place = fluid.CPUPlace()
# 创建执行器
exe = fluid.Executor(place)
# 执行计算
ones_result = exe.run(fluid.default_main_program(),
                      # 获取数据data
                      fetch_list=[data],
                      return_numpy=True)
# 输出结果
print(ones_result[0])

# 调用elementwise_op将生成的一维数组按位相加
add = fluid.layers.elementwise_add(data, data)
# 定义运算场所
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 执行计算
add_result = exe.run(fluid.default_main_program(),
                     fetch_list=[add],
                     return_numpy=True)
# 输出结果
print(add_result[0])

# 将一维整形数组，转换成float64类型
cast = fluid.layers.cast(x=data, dtype='float64')
# 定义运算场所执行运算
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 执行计算
cast_result = exe.run(fluid.default_main_program(),
                      fetch_list=[cast],
                      return_numpy=True)
# 输出结果
print(cast_result[0])
