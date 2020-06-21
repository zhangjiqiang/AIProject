import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np

# 用直方图分析下背景像素值大概是多少，好过滤掉背景。我们根据实际图像分析下背景区域
# 通过分析停车长外部背景像素分布，可以看到绝大多数值是在120以下，这些都是背景元素，和停车站的车位没有关系
# 可以过滤掉，排除一些识别停车位的干扰项
def show_hist(image):
    h, w = img.shape[:2]
    plt.figure(figsize=[15, 8])

    plt.subplot(2, 2, 1)
    plt.title('origin image')
    plt.hist(img.ravel(), 256, [0, 256])

    #plt.subplot(2, 2, 2)

    # 这一区域是图片最下面的区域，不在停车场里，这块区域几乎都是马路，即是背景部分。大多数值都低于120.
    plt.subplot(2, 2, 2)
    plt.hist(img[int(h*0.9):, int(w*0.1):].ravel(), 256, [0, 256])
    plt.title('image bottom')

    #这一区域是左上角那块，几乎都是背景组成的，停车场外面。
    plt.subplot(2, 2, 3)
    plt.hist(img[0:int(0.3*h), 0:int(w*0.2)].ravel(), 256, [0, 256])
    plt.title('left top image')

    #这一区域是图片最右面的区域，不在停车场里。
    plt.subplot(2, 2, 4)
    plt.hist(img[:, int(0.9*w):].ravel(), 256, [0, 256])
    plt.title('left top image')

    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()



def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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


img = cv2.imread('./images/parking.jpg')
print(img.shape)
show_hist(img)

filter_background(img)




