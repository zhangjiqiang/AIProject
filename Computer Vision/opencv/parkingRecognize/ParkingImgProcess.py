import cv2
import os, glob
import numpy as np
import operator

"""
  识别停车位，确定车位在图中的坐标
  该类处理所有识别车位坐标的相关的图像处理操作
"""
class ParkingImageProcess(object):
    # 过滤掉背景,我们可以通过直方图统计，可以分析得出背景大多是颜色在120以下的。我们可以过滤掉这些像素。
    # 只保留停车场车位有关的信息，这样能提高我们查找车位的效率和准确度
    def filter_background(self, image):
        #过滤掉背景
        lower = np.uint8([120, 120, 120])
        upper = np.uint8([255, 255, 255])
        # 各通道低于(120, 120, 120)的部分分别变成0，(120, 120, 120)～(255, 255, 255)之间的值变成255,相当于过滤背景
        white_mask = cv2.inRange(image, lower, upper)
        # 设置了掩码，即是mask=xxx，如果直接把white_mask放上去，则函数把white_mask当做输出了，所以发现会不起掩码的作用
        masked = cv2.bitwise_and(image, image, mask=white_mask)
        return masked

    def gray_process(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def cv_show(self,name,img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def canny_process(self, image, min_threshold=50, max_threshold=200):
        return cv2.Canny(image, min_threshold, max_threshold)

    def hough_lines(self, image):
        return cv2.HoughLinesP(image, rho=0.1, theta=np.pi/10, threshold=15,
                               minLineLength=25, maxLineGap=4)

    def select_region(self, image):
        # 根据停车场位置图，可以将不必要的区域去掉，仅保留停车场的区域
        # 我们用6个顶点坐标来划分停车场的区域 
        point_left_top = [83, 500]
        point_left_bottom = [85, 648]
        point_mid_top = [414, 367]
        point_right_bottom = [1145, 648]
        point_right_top = [1145, 76]
        point_mid_right_top = [742, 108]
        points = np.array([[point_left_top, point_left_bottom, point_right_bottom, point_right_top,
                            point_mid_right_top, point_mid_top]],
                            dtype=np.int32)
       # rows, cols = image.shape[:2]
       # pt_1  = [cols*0.05, rows*0.90]
       # pt_2 = [cols*0.05, rows*0.70]
       # pt_3 = [cols*0.30, rows*0.55]
       # pt_4 = [cols*0.6, rows*0.15]
       # pt_5 = [cols*0.90, rows*0.15]
       # pt_6 = [cols*0.90, rows*0.90]
       # points = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]],
       #                     dtype=np.int32)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, points, 255)
        newIamge =cv2.bitwise_and(image, mask)
        self.cv_show("roi region", newIamge)
        return newIamge

    #划分停车场的矩形块，将停车场划分为12个列块，这是根据具体停车场的布局得到的。不同停车场的布局要具体问题具体分析
    def select_park_Rect(self, lines_list, image):
        filter_straight_lines = []
        for line in lines_list:
            for x1,y1,x2,y2 in line:
                if abs(x1-x2)<=55 and abs(y1-y2)<=1:
                    filter_straight_lines.append((x1,y1,x2,y2)) #过滤直线

        # 对这些直线按照x1进行升序排列，如果x1相等则按照y1就进行升序排列,这样可以将停车场分出12个矩形块区域
        filter_straight_lines = sorted(filter_straight_lines,
                                       key=operator.itemgetter(0, 1))

        # 将停车长分为12个矩形块
        clusters = {}
        key_int = 0
        x1_diff = 10
        line_count = len(filter_straight_lines)
        for i in range(line_count - 1):
            distance = abs(filter_straight_lines[i][0]-filter_straight_lines[i+1][0])
            if distance <= x1_diff:
                if key_int not in clusters:
                    clusters[key_int] = []

                clusters[key_int].append(filter_straight_lines[i])
                clusters[key_int].append(filter_straight_lines[i+1])
            else:
                key_int += 1

        # 分成12个列组
        print("column count:", key_int+1)
        print("cluster count:",len(clusters))
        # 该字典是每个列组的最大包围矩形框
        rects = {}
        key_int = 0
        for key in clusters:
            column_cluster = clusters[key]
            column_cluster = list(set(column_cluster))
            if (len(column_cluster) > 5):
                #按照y轴大小升序排列
                column_cluster = sorted(column_cluster, key=lambda x: x[1])
                rect_lt_y = column_cluster[0][1]
                rect_rb_y = column_cluster[-1][1]
                column_cluster = sorted(column_cluster, key=lambda x: x[0])
                rect_lt_x = column_cluster[0][0] # 取最小的x1坐标
                column_cluster = sorted(column_cluster, key=lambda x: x[2])
                rect_rb_x = column_cluster[-1][2] # 取最大的x2坐标
                rects[key_int] = (rect_lt_x, rect_lt_y, rect_rb_x, rect_rb_y)
                key_int += 1

        new_image = image.copy()
        buff = 7
        print("rects count:",len(rects))
        for key in rects:
            cv2.rectangle(new_image, (rects[key][0],rects[key][1]),
                          (rects[key][2],rects[key][3]), (0,0,255), 3)
        self.cv_show("rect image:", new_image)

        return rects


    def find_park_space(self, image, rects, is_copy=True, color=[255,0,0],thickness=2, save=True):
        if is_copy:
            new_image = np.copy(image)
        park_space_width = 15.5 #停车位的宽度即是车宽的容纳
        # 对每列车位矩形框进行微调
        adj_y1 = {0:5, 1:1,   2:28, 3:-5, 4:25, 5:25, 6:-1, 7:-2,  8:7,  9:-7, 10:49, 11:-48}
        adj_y2 = {0:1, 1:-13, 2:15, 3:12, 4:2,  5:16, 6:15, 7:-15, 8:29, 9:15, 10:-5, 11:31}

        adj_x1 = {0:3, 1:0,   2:0,  3:16, 4:0,  5:6,  6:4,  7:1,   8:3,  9:13, 10:5,  11:7}
        adj_x2 = {0:1, 1:0,   2:0,  3:0,  4:0,  5:-7, 6:1, 7:-1,  8:-2, 9:-5,  10:-5, 11:0}
        # 存放车位，key是车位的位置，value是索引号
        parking_space_dict = {}
        for key in rects:
            rect = rects[key]
            x1 = int(rect[0] + adj_x1[key])
            x2 = int(rect[2] + adj_x2[key])
            y1 = int(rect[1] + adj_y1[key])
            y2 = int(rect[3] + adj_y2[key])
            cv2.rectangle(new_image, (x1, y1), (x2, y2), (0,255,0), 2)
            # 计算出每列有多少个车位
            column_parkSpace_num = int(abs(y1-y2)//park_space_width)
            # 画出每个车位的矩形框
            for i in range(column_parkSpace_num + 1):
                y = int(y1 + i*park_space_width)
                # 标识出每列的车位的横线
                cv2.line(new_image, (x1, y), (x2, y), color, thickness)
            if key > 0 and key < len(rects) -1:
                # 不是首尾这两列停车区域，它们是双排停车的，所以要用宽度除以2
                x = int((x1 + x2) / 2)
                # 双排车位一分为二，中间绘制一条竖线
                cv2.line(new_image, (x, y1), (x, y2), color, thickness)

            # 将每个车位的坐标记录到字典中去`
            parking_space_count = len(parking_space_dict)
            if key == 0 or key == len(rects) - 1:
                for i in range(column_parkSpace_num):
                    y = y1 + i*park_space_width
                    parking_space_dict[(x1, y, x2, y+park_space_width)] = parking_space_count + 1
            else:
                for i in range(column_parkSpace_num):
                    y = y1 + i*park_space_width
                    x = (x1 + x2) / 2
                    parking_space_dict[(x1, y, x, y + park_space_width)] = parking_space_count + 1
                    parking_space_dict[(x, y, x2, y + park_space_width)] = parking_space_count + 2


        if save:
            cv2.imwrite("parking_space.jpg", new_image)

        self.cv_show("patking_space", new_image)
        return parking_space_dict

    def predict(self, image, model):
        image = image/255.
        image = np.expand_dims(image, axis=0)
        result = model.predict(image)
        return np.argmax(result[0])

    def parking_predict(self, image, model, parking_dict):
        new_image = np.copy(image)
        overlay = np.copy(image)
        for car_pos in parking_dict.keys():
            (x1, y1, x2, y2) = tuple(map(int, car_pos))
            parking_region = image[y1:y2, x1:x2]
            parking_region = cv2.resize(parking_region, (48,48))
            index_id = self.predict(parking_region, model)
            if index_id == 0:
                cv2.rectangle(overlay, (x1,y1), (x2,y2), [0,255,0], -1)
            cv2.addWeighted(overlay, 0.5, new_image, 0.5, 0, new_image)

        self.cv_show("parking car", new_image)


