import paddle.fluid as fluid

# 使用顺序执行的方式搭建网络：
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
cost = fluid.layers.square_error_cost(input=y_predict, label=y)

# 条件分支——switch、if else：
lr = fluid.layers.tensor.create_global_var(
    shape=[1],
    value=0.0,
    dtype='float32',
    persistable=True,
    name="learn_rate"
)

one_var = fluid.layers.fill_constant(
    shape=[1],
    dtype='float32',
    value=1.0
)

two_var = fluid.layers.fill_constant(
    shape=[1],
    dtype='float32',
    value=2.0
)

with fluid.layers.control_flow.Switch() as switch:
    with switch.case(global_step == zero_var):
        fluid.layers.tensor.assign(input=one_var, output=lr)
    with switch.default():
        fluid.layers.tensor.assign(input=two_var, output=lr)
