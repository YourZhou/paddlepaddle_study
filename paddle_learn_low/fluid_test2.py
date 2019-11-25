import paddle.fluid as fluid

#通过fluid.layers.create_parameter来创建可学习参数
w = fluid.layers.create_parameter(name='w',shape=[1],dtype='float32')
#直接为全连接层创建连接权值（W）和偏置（ bias ）两个可学习参数
#y = fluid.layers.fc(input=x,size=128,bias_attr=True)

#定义x的维度为[3,None]，其中我们只能确定x的第一的维度为3，第二个维度未知，要在程序执行过程中才能确定
x = fluid.layers.data(name="x",shape=[3,None],dtype="int64")

#batch size无需显示指定，框架会自动补充第0维为batch size，并在运行时填充正确数值
a = fluid.layers.data(name="a",shape=[3,4],dtype='int64')

#若图片的宽度和高度在运行时可变，将宽度和高度定义为None。
#shape的三个维度含义分别是：channel、图片的宽度、图片的高度
b = fluid.layers.data(name="image",shape=[3,None,None],dtype="float32")

# fluid.layers.fill_constant 来实现常量Tensor，用户可以指定Tensor的形状，数据类型和常量值
data = fluid.layers.fill_constant(shape=[1],value=0,dtype='int64')
print(data)