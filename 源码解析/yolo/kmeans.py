import numpy as np

#对数据集的box长宽做聚类分析，得到若干组长宽不等的anchors。不同的对象的检测box长宽比例不同
#通过聚类可以总结出不同的长宽比例模式，可以更精确的识别分割图片中的对象
class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number  
        self.filename = "2012_train.txt"
    
	#计算当前选中的cluster_number个聚类中心和所有box之间的iou值
    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)#矩阵对应位置相乘

        result = inter_area / (box_area + cluster_area - inter_area)#这里取最小宽度和最小高度作为交叉的面积，和几何意义上的iou不同。我理解是如果box的长宽和某个中心的长宽是一个类别，则说明该box和中心的长宽比较接近，则他们的较小的宽和高的面积与他们两个各自
		                                                             #的面积比较接近。因为我们聚类的目的是找寻长宽相似的群体
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters,随机选取k个center
        while True:

            distances = 1 - self.iou(boxes, clusters)#计算box和center的距离，越小越接近

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():#all的方法判断所有元素都相等为True,这句代码的表达的是:当前每个box归类到某个中心中，这个中心和上一次的中心如果是同一个中心，则停止迭代。这表示中心已经稳定下来，不会变化了。
			                                           #比方说第一个box和3号中心是一类，每次都是和3号一类，则迭代结束
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)#取中位数,每次需要重新计算中心，根据聚类的到的数据，用每个类别的数据集计算新的中心
                    #current_nearest == cluster,表示1号，2号等一直到9号的中心对应的数据集(九个类别的数据聚集在一起,分成9个小团体)
            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

	#读取左上角和右下角坐标，计算每个box的width和height，保存到result中
    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result
    
	#对训练集的box的长宽做聚类，并打印结果
    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9  #设置聚类格式为9
    filename = "2012_train.txt" #读取训练集的路径
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
