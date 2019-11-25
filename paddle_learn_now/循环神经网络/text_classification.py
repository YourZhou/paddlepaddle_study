# imdb库，这个是一个数据集的库，这个是数据集是一个英文的电影评论数据集，
# 每一条数据都会有两个分类，分别是正面和负面。
import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid
import numpy as np

# 以下的代码片段是一个比较简单的循环神经网络
# fluid.layers.embedding()，这个是接口是接受数据的ID输入，因为输入数据时一个句子，
# 但是在训练的时候我们是把每个单词转换成对应的ID，再输入到网络中，所以这里使用到了embedding接口。
# 然后是一个全连接层，接着是一个循环神经网络块，在循环神经网络块之后再经过一个sequence_last_step接口，
# 这个接口通常是使用在序列函数的最后一步。最后的输出层的激活函数是Softmax，大小为2，因为数据的结果有2个，为正负面。
