import os
import shutil
import paddle as paddle
import paddle.dataset.cifar as cifar
import paddle.fluid as fluid

# 定义一个残差神经网络，这个是目前比较常用的一个网络。
# 该神经模型可以通过增加网络的深度达到提高识别率，
# 而不会像其他过去的神经模型那样，
# 当网络继续加深时,反而会损失精度。
# 定义残差神经网络（ResNet）
def resnet_cifar10(ipt, class_dim):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
        short = shortcut(input, ch_in, ch_out, stride)
        return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

    # 残差块
    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    conv1 = conv_bn_layer(ipt, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, 5, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, 5, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, 5, 2)
    pool = fluid.layers.pool2d(input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    predict = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return predict

# 定义输入层
image = fluid.layers.data(name='image', shape=[3, 32, 32], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取残差神经网络的分类器，并指定分类大小是10，因为这个数据集有10个类别。
# 获取分类器
model = resnet_cifar10(image,10)

# 获取交叉熵损失函数和平均准确率
# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model,label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model,label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3)
opt = optimizer.minimize(avg_cost)

# 获取训练和测试数据，使用的是cifar数据集，cifar数据集有两种，
# 一种是100个类别的，一种是10个类别的，这里使用的是10个类别的。
# 获取CIFART数据
train_reader = paddle.batch(cifar.train10(),batch_size=32)
test_reader = paddle.batch(cifar.test10(),batch_size=32)

# 创建执行器，最好使用GPU，CPU速度太慢了
place = fluid.CPUPlace()
# place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 加载之前训练保存的持久化变量模型，
# 对应的保存接口是fluid.io.save_persistables。
# 使用这些模型参数初始化网络参数，进行训练。
# 加载之前训练过的检查点模型
save_path = 'models/persistables_model/'
if os.path.exists(save_path):
    print('使用持久化变量模型作为预训练模型')
    fluid.io.load_persistables(executor=exe, dirname=save_path)

# 开始训练模型。
# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

for pass_id in range(10):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))

# 保存持久化变量模型，之后用于初始化模型，进行训练。
# 保存持久化变量模型
save_path = 'models/persistables_model/'
# 删除旧的模型文件
shutil.rmtree(save_path, ignore_errors=True)
# 创建保持模型文件目录
os.makedirs(save_path)
# 保存持久化变量模型
fluid.io.save_persistables(executor=exe,dirname=save_path)