import paddle.fluid as fluid
from PIL import Image
import numpy as np
import cv2

# 创建执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'infer_model/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)

how = 0
if how == 0:
    cap = cv2.VideoCapture(how)

# 预处理图片
def load_image(file):
    img = Image.open(file)
    # 统一图像大小
    img = img.resize((224, 224), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if how == 0:
    while (1):
            ret ,frame = cap.read()
            k = cv2.waitKey(1)
            cv2.imshow("capture",frame)
            if k == 27 :
                break
            elif k == ord('s'):
                cv2.imwrite('test.jpg',frame)
                break

# 获取图片数据
img = load_image('test.jpg')
# 执行预测
result = exe.run(program=infer_program,
                feed={feeded_var_names[0]: img},
                fetch_list=target_var)

# 显示图片并输出结果最大的label
lab = np.argsort(result)[0][0][-1]

if result[0][0][lab] >= 0.8:
    names = ['香蕉', '胡萝卜', '黄瓜', '矿泉水', '西瓜']
    print('*******************\n预测结果标签为：%d， 名称为：%s， 概率为：%f\n*******************' % (lab, names[lab], result[0][0][lab]))
elif result[0][0][lab] <= 0.8:
    print("*******************\n当前场景没有可识别物品\n*******************")

cv2.waitKey(1000)
cap.release()
cv2.destroyAllWindows()
