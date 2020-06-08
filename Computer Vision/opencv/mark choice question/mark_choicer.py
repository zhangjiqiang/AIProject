#导入工具包
import numpy as np
import argparse
import imutils
import cv2
import operator
'''
 判断当前答题卡选项分数。我们给出一张单选答题卡。考生会用2B铅笔在正确的选项上涂黑，
 代表选中此选项为正确答案。我们通过图像处理的的方式来判断每个选择题的正确与否。每道题只有一个正确答案
'''

# 显示图片
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clockwise_order(pts):
    # 一共4个坐标点
    src_point = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标，分别是 左上，右上，右下，左下

    # 先对y坐标进行升序排列,如果y相同，则按照x大小排序
    pts = sorted(pts, key=operator.itemgetter(1, 0))
    # 根据x大小确定谁是左上和右上
    if pts[0][0] < pts[1][0]:
        src_point[0] = pts[0]
        src_point[1] = pts[1]
    else:
        src_point[0] = pts[1]
        src_point[1] = pts[0]
    # 根据x大小确定谁是左下和右下
    if pts[2][0] < pts[3][0]:
        src_point[2] = pts[3]
        src_point[3] = pts[2]
    else:
        src_point[2] = pts[2]
        src_point[3] = pts[3]

    return src_point

#进行投影变换
def perspectiveTransform(image, pts):
    src_point = clockwise_order(pts)
    (lt, rt, rb, lb) = src_point
    width_top = np.sqrt((lt[0]-rt[0])**2 + (lt[1]-rt[1])**2)
    width_bottom = np.sqrt((rb[0]-lb[0])**2 + (rb[1]-lb[1])**2)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.sqrt((lt[0]-lb[0])**2 + (lt[1]-lb[1])**2)
    height_right = np.sqrt((rt[0]-rb[0])**2 + (rt[1]-rb[1])**2)
    max_height = max(int(height_left), int(height_right))

    # 变换后矩阵的位置
    dst_point = np.array([[0, 0],[max_width - 1, 0],[max_width - 1,
                        max_height - 1],[0, max_height - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(src_point,dst_point)
    transform_image = cv2.warpPerspective(image, M, (max_width, max_height))
    return transform_image

# 正确答案字典，key表示每道选择题的序号，value表示选择地接答案
correct_answer_dict = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# 图形预处理，查找图像边缘.这样处理能够更准确地查找轮廓
img_path = './images/test1.png'
ori_image = cv2.imread(img_path)
copyImage = ori_image.copy()
grayImage = cv2.cvtColor(copyImage, cv2.COLOR_BGR2GRAY)
gaussBlurImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
edgedImage = cv2.Canny(gaussBlurImage, 75, 200)
cv_show('canny', edgedImage)

contours = cv2.findContours(edgedImage, cv2.RETR_EXTERNAL,
	       cv2.CHAIN_APPROX_SIMPLE)[1]

# 所有外轮廓中，包裹整个选择题的区域是面积最大的。我们只要能包裹全部选择题的区域就足够，剩下的都可以过滤掉
if len(contours) > 0:
    max_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]

# 求取这个最大轮廓的近似多边形
perimeter = cv2.arcLength(max_contour, True)
# 调整阈值大小，
approxRect = cv2.approxPolyDP(max_contour, 0.01*perimeter, True)
if len(approxRect) == 4:
    contour_image = cv2.drawContours(ori_image.copy(), [approxRect], -1, (0, 255, 0), 2)
    cv_show('contour_image', contour_image)


transformImg = perspectiveTransform(grayImage, approxRect.reshape(4, 2))
cv_show('transform', transformImg)
# 二值处理
threshImage = cv2.threshold(transformImg, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv_show('threshImage', transformImg)
print(transformImg.shape)

# 找到每一个选择题的圆圈轮廓
contours = cv2.findContours(threshImage, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)[1]

choicesCnts = []
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    ratio = w / float(h)
    if ratio > 0.9 and ratio < 1.1 and w >20 and h > 20:
        choicesCnts.append(contour)

print(len(choicesCnts))

