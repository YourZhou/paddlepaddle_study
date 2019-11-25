# 定义一个简单的多层感知器，一共有三层，两个大小为100的隐层和一个大小为10的输出层，
# 因为MNIST数据集是手写0到9的灰度图像，类别有10个，所以最后的输出大小是10。
# 最后输出层的激活函数是Softmax，所以最后的输出层相当于一个分类器。
# 输入层-->>隐层-->>隐层-->>输出层。

import numpy as np
import paddle as paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt


# 定义多层感知器
def multilayer_perceptron(input):
    # 第一个全连接层，激活函数为ReLU
    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')
    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    return fc


# fluid.layers.conv2d()来做一次卷积操作，我们可以通过num_filters参数设置卷积核的数量，
# 通过filter_size设置卷积核的大小，还有通过stride来设置卷积操作时移动的步长。
# 使用fluid.layers.pool2d()接口做一次池化操作，通过参数pool_size可以设置池化的大小，
# 通过参数pool_stride设置池化滑动的步长，通过参数pool_type设置池化的类型，
# 目前有最大池化和平均池化，下面使用的时最大池化，当值为avg时是平均池化。
# 输入层-->>卷积层-->>池化层-->>卷积层-->>池化层-->>输出层
# 卷积神经网络
def convoluntional_neural_network(input):
    # 第一个卷积层，卷积核大小为3*3，一共有32个卷积核
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=32,
                                filter_size=3,
                                stride=1)

    # 第一个池化层，池化大小为2*2，步长为1，最大池化
    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    # 第2个卷积层，卷积核大小为3*3，一共有64个卷积核
    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=64,
                                filter_size=3,
                                stride=1)

    # 第2个池化层，池化大小为2*2，步长为1，最大池化
    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=pool2, size=10, act='softmax')
    return fc


# 定义输入层，输入的是图像数据。图像是28*28的灰度图，所以输入的形状是[1, 28, 28]，
# 如果图像是32*32的彩色图，那么输入的形状是[3. 32, 32]，因为灰度图只有一个通道，而彩色图有RGB三个通道
# 定义输入层
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 可以尝试这两种不同的网络进行训练，观察一下他们的准确率如何
# 获取分类器
# model = multilayer_perceptron(image)
model = convoluntional_neural_network(image)

# 使用的是交叉熵损失函数，该函数在分类任务上比较常用
# 同时我们还可以定义一个准确率函数，这个可以在我们训练的时候输出分类的准确率
# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 我们使用的是Adam优化方法，同时指定学习率为0.001
# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
optimizer.minimize(avg_cost)

# 定义读取MNIST数据集的reader，指定一个Batch的大小为128，也就是一次训练128张图像
# 获取MNIST数据
train_reader = paddle.batch(mnist.train(), batch_size=128)
test_reader = paddle.batch(mnist.test(), batch_size=128)

# 定义一个使用CPU的执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 输入的数据维度是图像数据和图像对应的标签，每个类别的图像都要对应一个标签，这个标签是从0递增的整型数值。
# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 开始训练和测试
for pass_id in range(1):
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
        test_cost, test_acc = exe.run(program=fluid.default_main_program(),
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))


# 预测图像
# 首先进行灰度化，然后压缩图像大小为28*28，接着将图像转换成一维向量，最后再对一维向量进行归一化处理。
# 对图片进行预处理
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im


# 使用Matplotlib工具显示这张图像
img = Image.open('infer_0.jpg')
plt.imshow(img)
plt.show()

# 最后把图像转换成一维向量并进行预测，数据从feed中的image传入，
# label设置一个假的label值传进去。
# fetch_list的值是网络模型的最后一层分类器，所以输出的结果是10个标签的概率值，这些概率值的总和为1。
img = load_image('./infer_0.jpg')
results = exe.run(program=test_program,
                  feed={'image': img, 'label': np.array([[1]]).astype("int64")},
                  fetch_list=[model])

# 拿到每个标签的概率值之后，我们要获取概率最大的标签，并打印出来。
# 获取概率最大的label
lab = np.argsort(results)
print("该图片的预测结果的label为: %d" % lab[0][0][-1])
