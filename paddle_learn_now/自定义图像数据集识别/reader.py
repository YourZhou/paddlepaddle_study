# 这个程序就是用户训练和测试的使用读取数据的。
# 训练的时候，通过这个程序从本地读取图片，
# 然后通过一系列的预处理操作，最后转换成训练所需的Numpy数组。
# cpu_count是获取当前计算机有多少个CPU，然后使用多线程读取数据。

import os
import random
from multiprocessing import cpu_count
import numpy as np
import paddle
from PIL import Image


# 这个函数是根据传入进来的图片路径来对图片进行预处理，
# 比如训练的时候需要统一图片的大小，
# 同时也使用多种的数据增强的方式，如水平翻转、垂直翻转、角度翻转、随机裁剪，
# 这些方式都可以让有限的图片数据集在训练的时候成倍的增加。
# 最后因为PIL打开图片存储顺序为H(高度)，W(宽度)，C(通道)，PaddlePaddle要求数据顺序为CHW，
# 所以需要转换顺序。最后返回的是处理后的图片数据和其对应的标签
# 训练图片的预处理
def train_mapper(sample):
    img_path, label, crop_size, resize_size = sample
    try:
        img = Image.open(img_path)
        # 统一图片大小
        img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
        # 随机水平翻转
        r1 = random.random()
        if r1 > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # 随机垂直翻转
        r2 = random.random()
        if r2 > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # 随机角度翻转
        r3 = random.randint(-3, 3)
        img = img.rotate(r3, expand=False)
        # 随机裁剪
        r4 = random.randint(0, int(resize_size - crop_size))
        r5 = random.randint(0, int(resize_size - crop_size))
        box = (r4, r5, r4 + crop_size, r5 + crop_size)
        img = img.crop(box)
        # 把图片转换成numpy值
        img = np.array(img).astype(np.float32)
        # 转换成CHW
        img = img.transpose((2, 0, 1))
        # 转换成BGR
        img = img[(2, 1, 0), :, :] / 255.0
        return img, int(label)
    except:
        print("%s 该图片错误，请删除该图片并重新创建图像数据列表" % img_path)


# 函数是根据已经创建的图像列表解析得到每张图片的路径和其他对应的标签，
# 然后使用paddle.reader.xmap_readers()
# 把数据传递给上面定义的train_mapper()函数进行处理，
# 最后得到一个训练所需的reader。
# 获取训练的reader
def train_reader(train_list_path, crop_size, resize_size):
    father_path = os.path.dirname(train_list_path)

    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            # 打乱图像列表
            np.random.shuffle(lines)
            # 开始获取每张图像和标签
            for line in lines:
                img, label = line.split('\t')
                img = os.path.join(father_path, img)
                yield img, label, crop_size, resize_size

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 102400)


# 测试数据的预处理函数test_mapper()，这个没有做太多处理，
# 因为测试的数据不需要数据增强操作，
# 只需统一图片大小和设置好图片的通过顺序和数据类型即可。
# 测试图片的预处理
def test_mapper(sample):
    img, label, crop_size = sample
    img = Image.open(img)
    # 统一图像大小
    img = img.resize((crop_size, crop_size), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img, int(label)

# 测试的reader函数test_reader()，这个跟训练的reader函数定义一样。
# 测试的图片reader
def test_reader(test_list_path, crop_size):
    father_path = os.path.dirname(test_list_path)

    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                img = os.path.join(father_path, img)
                yield img, label, crop_size

    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 1024)

