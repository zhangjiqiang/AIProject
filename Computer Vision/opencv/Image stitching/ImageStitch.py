import numpy as np
import cv2

def cv_show(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#缩放图片，拼接的图像大小不一定相同
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    size = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        size = (int(w * r), height)
    else:
        r = width / float(w)
        size = (width, int(h * r))
    resized = cv2.resize(image, size, interpolation=inter)
    return resized

# 求单应矩阵
def findHomographyMatrix(featuresL, kpsL, featuresR, kpsR, ratio, ransacReprojThreshold):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(featuresR, featuresL, k=2)
    matchKp_Ids = []
    for match in matches:
        if match[0].distance < match[1].distance * ratio and len(match) == 2:
            matchKp_Ids.append((match[0].queryIdx, match[0].trainIdx)) #queryIdx是featuresR的kp索引，trainIdx是featureR的kp索引

    # 至少需要4个点才能求得单应矩阵
    if len(matchKp_Ids) >= 4:
        despopints = []
        srcpoints = []
        for ids in matchKp_Ids:
            despopints.append(kpsL[ids[1]])
            srcpoints.append(kpsR[ids[0]])
        # 计算视角变换矩阵
        srcpoints = np.array(srcpoints, dtype=np.float)
        despopints = np.array(despopints, dtype=np.float)
        (M, status) = cv2.findHomography(srcpoints, despopints, cv2.RANSAC, ransacReprojThreshold)
        return M
    else:
        return None

#提取SIFT特征
def extractFeature(image):
    descriptor_sift = cv2.xfeatures2d_SIFT.create()
    kps, features = descriptor_sift.detectAndCompute(image, None)
    kps = np.array([kp.pt for kp in kps], dtype=np.float)

    return (kps, features)

#图片合成，根据alpha比例来融合图像
def compositeImage(imgL, warpR, imgTmp):
    imgRes = imgTmp
    alpha = 1
    gray = cv2.cvtColor(warpR, cv2.COLOR_BGR2GRAY)
    rows, cols = np.where(gray != 0) #查找不为0的像素位置
    start = min(cols)
    minrow = min(rows)
    width = imgL.shape[1] - start
    for i in range(minrow, imgL.shape[0]):
        for j in range(int(start), imgL.shape[1]):
            if warpR[i, j, :].all() == 0:
                alpha = 1
            else:
                alpha = (width - (j - start)) / width #按照占据width的比例来分配alpha
            imgRes[i, j, :] = imgL[i, j, :] * alpha + warpR[i, j, :] * (1 - alpha)
    return imgRes

def ImageStitch(imageL, imageR, ratio=0.75, ransacReprojThreshold=4):
    kpsL, featuresL = extractFeature(imageL)
    kpsR, featuresR = extractFeature(imageR)
    M = findHomographyMatrix(featuresL, kpsL, featuresR, kpsR, ratio, ransacReprojThreshold)
    wrap_image = cv2.warpPerspective(imageR, M, (imageL.shape[1] + imageR.shape[1], imageR.shape[0]))
    imgTmp = wrap_image.copy()
    imgTmp[0:imageR.shape[0], 0:imageL.shape[1]] = imageL
    cv_show(imgTmp, 'result')

#/////////////////////////////////////////////////////////////////
    #根据需要使用图片合成
    # rows, cols = np.where(imgTmp[:, :, 0] != 0)
    # min_row, max_row = min(rows), max(rows) + 1
    # min_col, max_col = min(cols), max(cols) + 1
    # imgTmp = imgTmp[min_row:max_row, min_col:max_col, :]
    # print("imageL shape:",imageL.shape)
    # print('wrap_image shape',wrap_image.shape)
    # print("imgTmp shape", imgTmp.shape)
    # imgRes = Optimize(imageL, wrap_image, imgTmp)
    # cv_show(imgRes, 'result')




imageL = resize(cv2.imread('./images/left_01.png'),800)
imageR = resize(cv2.imread('./images/right_01.png'),800)

print(imageL.shape)
print(imageR.shape)
cv2.imshow("Image L", imageL)
cv2.imshow("Image R", imageR)
ImageStitch(imageL, imageR, 0.75, 4.0)
print("finished")





