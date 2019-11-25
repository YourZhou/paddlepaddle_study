# 按要求处理数据
# !/usr/bin/evn python
# coding:utf-8
import os

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import sys

label = []
# 分别制作 train  test  val ， 一共要跑三次哦
for set_ in ['drink_train', 'drink_test', 'drink_val']:
    xml_ROOT = 'D:\\YourZhouProject\\PaddleCamp\\drink\\{}\\Annotations'.format(set_)
    jpg_ROOT = 'D:\\YourZhouProject\\PaddleCamp\\drink\\{}\\images'.format(set_)
    out_txt_path = 'D:\\YourZhouProject\\PaddleCamp\\drink\\{}.txt'.format(set_)

    print(set_)

    xml_list = os.listdir(xml_ROOT)  # 其中包含所有待计算的文件名

    if os.path.exists(out_txt_path):
        os.remove(out_txt_path)

    txt = open(out_txt_path, 'w')

    for xml_n in xml_list:
        xml_path = os.path.join(xml_ROOT, xml_n)
        tree = ET.parse(xml_path)  # 打开xml文档
        root = tree.getroot()  # 获得root节点
        # print ("*"*10)
        filename = root.find('filename').text
        filename = os.path.join(jpg_ROOT, filename)
        # print (filename)

        all_box_str = filename + '\t'
        box_count = 0
        for object in root.findall('object'):  # 找到root节点下的所有object节点
            name = object.find('name').text  # 子节点下节点name的值
            # if name!= 'Leconte':
            #     continue

            try:
                p = label.index(name)
            except Exception as e:
                label.append(name)

            box_count += 1
            bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # 图片路径\t	{"value":"bolt","coordinate":[[769.459,241.819],[947.546,506.167]]}\t{...}... 注意，每个字段值之间用\t分割
            box_str = '{{\"value\":\"{}\",\"coordinate\":[[{:.3f},{:.3f}],[{:.3f},{:.3f}]]}}\t'. \
                format(name, xmin, ymin, xmax, ymax)
            all_box_str += box_str
            pass

        if box_count == 0:
            continue

        all_box_str += '\n'
        # print(all_box_str)
        txt.write(all_box_str)
    txt.close()
    print('{}.txt is ok '.format(set_))
print(label)
txt = open('D:\\YourZhouProject\\PaddleCamp\\drink\\label_list', 'w')
for labels in label:
    txt.write(labels + '\n')
txt.close()
print('label_list is ok')
txt = open('D:\\YourZhouProject\\PaddleCamp\\drink\\label_list.txt', 'w')
i = 0
for labels in label:
    txt.write('{}\t{}\n'.format(i, labels))
    i += 1
txt.close()
print('label_list.txt is ok')
