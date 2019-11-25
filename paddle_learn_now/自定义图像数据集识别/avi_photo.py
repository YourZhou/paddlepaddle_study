# 本文件用来从视频中获取帧数及图片

import cv2 as cv
import ffmpeg

vc = cv.VideoCapture("D:\\YourZhouDownloads\\FFPUT\\scenery_0711\\tower02.mp4")

c = 1
a = 359

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

timeF = 1

while rval:
    rval, frame = vc.read()
    if (c % timeF == 0):
        cv.imwrite("D:/YourZhouDownloads/FFPUT/scenery_0711/tower01/" + str(a) + '.jpg', frame)
        a += 1
    c += 1
    cv.waitKey(1)
vc.release()
