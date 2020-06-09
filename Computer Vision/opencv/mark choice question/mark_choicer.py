#导入工具包
import numpy as np
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

# 对轮廓进行排序
# orderMode=1表示从上到下排序，orderMode=2表示按照从左到右排序
def orderContours(contours, orderMode=1):
    boundrects = [cv2.boundingRect(c) for c in contours]
    keyid = -1
    if orderMode == 1:
        keyid = 1
    if orderMode == 2:
        keyid = 0
    (contours, boundrects) = zip(*sorted(zip(contours, boundrects), key=lambda x: x[1][keyid], reverse=False))
    print("boundrects:",boundrects)
    return (contours, boundrects)

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
# 二值处理,THRESH_OTSU会自动寻找合适的阈值，适合双峰。
threshImage = cv2.threshold(transformImg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv_show('threshImage', transformImg)

# 找到每一个选择题的圆圈轮廓
contours = cv2.findContours(threshImage, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)[1]

choicesCnts = []
#遍历所有外轮廓，找到选择题对应的轮廓
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    ratio = w / float(h)
    #选择题对应的轮廓的过滤条件
    if ratio >= 0.9 and ratio <= 1.1 and w > 20 and h > 20:
        choicesCnts.append(contour)

#先按照由上到下顺序排序，每道选择题的y值是差不多的
questionContours = orderContours(choicesCnts)[0]
choiceCount = len(questionContours)

correctCount = 0 #统计答对个数
max_non_zero = -1 #当前选项最大非0个数
correct_choice = -1 #记录当前选择题选择了哪个选项
#每道选择题有5个选项，所以设置step=5
for i in range(0, choiceCount, 5):
    questionIndex = int(i / 5)
    #得到每个选择题的5个选项,对这5个选项进行排序，按照x坐标由小到大排列，这样就得到了和原图一样的顺序
    choiceCnts = orderContours(questionContours[i:5 * (questionIndex + 1)], 2)[0]
    for j, cnt in enumerate(choiceCnts):
        mask = np.zeros(threshImage.shape, dtype="uint8")
        # -1表示往轮廓里面填充,在轮廓范围内按照color值填充。这样轮廓内的是255，其余是0.构造出过滤掩码。
        #通过这个掩码可以过滤掉其它选项，只保留掩码对应的选项
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mask = cv2.bitwise_and(threshImage, threshImage, mask=mask)
        #统计每个选项非0个数，因为如果被选中的话，这个选项的轮廓区域内非0个数会很多。反之没被选中的轮廓区域非0个数很少
        #我们就根据统计非0个数来判断哪个选项被选中，非0个数最多的那个就是被选中的
        nonZeroCount = cv2.countNonZero(mask)
        if correct_choice == -1 or nonZeroCount > max_non_zero:
            correct_choice = j
            max_non_zero = nonZeroCount

    if correct_answer_dict[questionIndex] == correct_choice:
        correctCount += 1
    print("current choice:", correct_choice)
    max_non_zero = -1
    correct_choice = -1

score = (correctCount / 5.0) * 100
print("score:", score)