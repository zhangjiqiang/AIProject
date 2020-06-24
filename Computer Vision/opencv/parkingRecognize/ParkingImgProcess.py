import cv2
import os, glob
import numpy as np

class ParkingImageProcess(object):
	"""识别停车位，确定车位在图中的坐标
       该类处理所有识别车位坐标的相关的图像处理操作
	"""

    def filter_background(image):
    # 过滤掉背景,我们可以通过直方图统计，可以分析得出背景大多是颜色在120以下的。我们可以过滤掉这些像素。
    # 只保留停车场车位有关的信息，这样能提高我们查找车位的效率和准确度
    lower = np.uint8([120, 120, 120])
    upper = np.uint8([255, 255, 255])
    # 各通道低于(120, 120, 120)的部分分别变成0，(120, 120, 120)～(255, 255, 255)之间的值变成255,相当于过滤背景
    white_mask = cv2.inRange(image, lower, upper)
    cv2.imshow('white_mask', white_mask)

    masked = cv2.bitwise_and(image, image, mask=white_mask)
    cv_show('masked', masked)
    return masked

    def gray_process(image):
    	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def canny_process(image, min_threshold=50, max_threshold=200):
    	return cv2.Canny(image, min_threshold, max_threshold)

