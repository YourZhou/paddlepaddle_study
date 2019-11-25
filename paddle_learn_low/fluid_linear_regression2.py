from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy
import math
import sys

BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500
    ),
    batch_size=BATCH_SIZE
)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.test(), buf_size=500
    ),
    batch_size=BATCH_SIZE
)

x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
y_predoct = fluid.layers.fc(input=x, size=1, act=None)

# 获取默认/全局主函数
main_program = fluid.default_main_program()
# 获取默认/全局启动程序
startup_program = fluid.default_startup_program()

# 利用标签数据和输出的预测数据估计方差
cost = fluid.layers.square_error_cost(input=y_predoct, label=y)
# 对方差求均值，得到平均损失
avg_loss = fluid.layers.mean(cost)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_loss)

# 克隆main_program得到test_program
# 有些operator在训练和测试之间的操作是不同的，例如batch_norm，使用参数for_test来区分该程序是用来训练还是用来测试
# 该api不会删除任何操作符,请在backward和optimization之前使用
test_program = main_program.clone(for_test=True)

# 指明executor的执行场所
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

num_epochs = 100


def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program,
            feed=feeder.feed(data_test),
            fetch_list=fetch_list
        )
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]  # 累加测试过程中的损失值
        count += 1
    return [x_d / count for x_d in accumulated]  # 计算平均损失


# %matplotlib inline
params_dirname = "fit_a_line.inference.model"
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
exe.run(startup_program)
train_prompt = "train cost"
test_prompt = "test cost"
from paddle.utils.plot import Ploter

plot_prompt = Ploter(train_prompt, test_prompt)
step = 0

exe_test = fluid.Executor(place)


