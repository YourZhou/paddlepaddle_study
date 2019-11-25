# 此文件用于批量改变文件夹的文件名

import os

path = input('请输入文件路径(结尾加上/)：')

# 获取该目录下所有文件，存入列表中
f = os.listdir(path)
a = 0
n = 434
for i in f:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + f[a]

    # 设置新文件名
    newname = path + "rock" + str(n) + '.jpg'

    # 用os模块中的rename方法对文件改名
    os.rename(oldname, newname)
    print(oldname, '======>', newname)
    a += 1
    n += 1
