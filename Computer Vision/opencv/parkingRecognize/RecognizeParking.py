import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
import pickle
from keras.models import load_model
from ParkingImgProcess import ParkingImageProcess

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error

# 用直方图分析下背景像素值大概是多少，好过滤掉背景。我们根据实际图像分析下背景区域
# 通过分析停车长外部背景像素分布，可以看到绝大多数值是在120以下，这些都是背景元素，和停车站的车位没有关系
# 可以过滤掉，排除一些识别停车位的干扰项
def show_hist(image):
    h, w = img.shape[:2]
    plt.figure(figsize=[15, 8])
    plt.subplot(2, 2, 1)
    plt.title('origin image')
    plt.hist(img.ravel(), 256, [0, 256])

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


def draw_lines(lines_list, image):
    copy_img = image.copy()
    print("line length:", len(lines_list))
    print("line_lsit type:", type(lines_list))
    strights = []
    for line in lines_list:
        for x1,y1,x2,y2 in line:
            if abs(y2-y1)<=1 and (x2-x1)<=55:
                strights.append(line)
                cv2.line(copy_img, (x1,y1), (x2,y2), [0,0,255], 2)

    print("stright num:", len(strights))
    cv_show("lines", copy_img)

def keras_model_load(weight_path):
    model = load_model(weight_path)
    return model


img = cv2.imread('./images/parking.jpg')
#show_hist(img)
parkImgProcess = ParkingImageProcess()
filter_img = parkImgProcess.filter_background(img)
gray_img = parkImgProcess.gray_process(filter_img)
edge_img = parkImgProcess.canny_process(gray_img)
parkImgProcess.cv_show('edge', edge_img)
select_img = parkImgProcess.select_region(edge_img)
lines_list = parkImgProcess.hough_lines(select_img)
draw_lines(lines_list, img)
rects = parkImgProcess.select_park_Rect(lines_list, img)
parking_space_dict = parkImgProcess.find_park_space(img, rects)
model = keras_model_load("./park_model.h5")
parkImgProcess.parking_predict(img, model, parking_space_dict)

