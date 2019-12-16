"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

#将模型的每个输出层的输出进行转换
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y]) #grid得到每个单元格与相对左上角的偏移量。
	'''
	(0,0)位置的cell相对于左上角,在X和Y的方向偏移量都是0,(1,0)位置的cell，在Y轴上有一个cell的偏移,在X轴方向上没有。
	以此类推,grid就记录每个box相对于左上角的偏移量,每个cell加上grid就得到每个box相对于cell的坐标
	'''
    grid = K.cast(grid, K.dtype(feats))
    #把模型的输出重新reshape,原始输出shape是[batchsize, grid_shape[0], grid_shape[1],num_anchors*(num_classes + 5)]
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))#计算bx=cx+tx
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats)) #计算转换后的wh 这些计算公式来自yolov2的论文
    box_confidence = K.sigmoid(feats[..., 4:5])#检测出对象的置信程度
    box_class_probs = K.sigmoid(feats[..., 5:])#对象的类别概率

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

#取得真正的坐标信息,根据图片实际大小转换后的box真实坐标
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]#将列的x,y以颠倒
    box_hw = box_wh[..., ::-1]#将列的w,h颠倒
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])
    #乘上图片实际大小,得到关于图片实际大小吻合的真正的box坐标
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

#得到关于实际大小的真box坐标和检测box评分水平
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4]) #调整box的shape,转为二维数组,行数表示有多少个预测box,列数4记录每个box的中心坐标和框的长宽,大小都是基于实际图片大小的绝对数值
    box_scores = box_confidence * box_class_probs #计算筛选盒子的score,用这个score和传递过来的分数阈值比较,列数是类别个数.记录每个box的关于判断是哪个个类别的置信程度
    return boxes, box_scores
	
'''
评估yolo模型的预测值,根据传递的参数score_threshold和iou_threshold这两个阈值,
采用非极大化抑制来筛选出合适的box框
'''
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] #设置每个输出层对应的anchor索引号
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32 #model的输入是32的倍数
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)#[N1+N2+N3,4]如果num_layers==3
    box_scores = K.concatenate(box_scores, axis=0)#[N1+N2+N3,class_num]

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])#选出该类别的box
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
		#假设筛选出N个，即nms_index得到的索引个数是N个
        class_boxes = K.gather(class_boxes, nms_index) #[N,4]显示盒子坐标
        class_box_scores = K.gather(class_box_scores, nms_index)#[N,1]显示分数
        classes = K.ones_like(class_box_scores, 'int32') * c #[N,1]显示类别
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
		
    boxes_ = K.concatenate(boxes_, axis=0)#每个类别对应的box的累计
    scores_ = K.concatenate(scores_, axis=0)#每个类别对应的分数的累计
    classes_ = K.concatenate(classes_, axis=0)#每个类别对应的实际类id的累计

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):

	'''
	构造符合训练model要求的数据格式。
	参数解释:
	true_boxes:s shape:(batchsize,maxbox_num,5)记录每张图片检测对象box的坐标信息和类别
	input_shape:(416,416)取32的倍数
	anchors:shape(N,2)N个anchor，记录每个anchor的width和height。反映出预测框的长宽比例
	num_classes:类别个数
	
	返回:
	返回一个列表。列表元素个数是num_layers = len(anchors)//3，这里个数是3。反映出不同单元网格划分对应的输出值
	即是(13,13),(26,26),(52,52)这三个网格划分对应的输出。
	输出shape:(batchsize,13,13,3,5+num_class),(batchsize,26,26,3,5+num_class),(batchsize,26,26,3,5+num_class)
	记录每个单元格负责检测对象的信息：3表示每个cell有预测三个box，5+num表示每个box含有box框的中心点坐标，框的左上和右下角坐标，是否检测到对象，以及类别信息
	第0和1位是中心点xy，范围是(0~13/26/52)，第2和3位是宽高wh，范围是0~1，
    第4位是置信度1或0，第5~n位是类别为1其余为0。
	'''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # 多少个输出
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2#计算box的中心点坐标
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2] #计算box的宽和高
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1] #归一化坐标,相对于真个图片尺寸
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1] #归一化宽和高,相对于真个图片尺寸

    m = true_boxes.shape[0]#图片数量
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]#网格划分尺寸数据，数据格式:[(13,13),(26,26),(52,52)]
	
	#y_true：真值集合,构造符合计算损失的数据结构。即是返回值
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

	#增加维度用于 broadcasting计算
    anchors = np.expand_dims(anchors, 0)#anchors shape:(1,9,2)9是anchor的个数
    anchor_maxes = anchors / 2. 
    anchor_mins = -anchor_maxes
	#w大于0的box参与计算,因为true_boxes里面有可能图片检测到的对象小于maxbox的数量，导致会有0填充。这部分不必参与计算，可以提高效率
    valid_mask = boxes_wh[..., 0]>0 
    
	#遍历每一张图片，确定每一张图片中检测到的对象都有哪些cell负责
    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]#筛选出非0填充的box,shape:(n,2),n表示多少个w不为0的box
        if len(wh)==0: continue
        #wh增加维度,采取这个方式是为了满足broadcast机制,让所有的box和anchor计算彼此之间的iou
        wh = np.expand_dims(wh, -2)#wh扩展后的shape:(n,1,2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins) #左上角最大坐标取最大
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)#右上角坐标取最小
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.) #靠左上的最大坐标减去口右下的坐标坐标,就是交叉部分的width和height
		'''
		这种计算archor和box的iou方式是将两者的中心都移动到原点，原点在左上角.
		我理解archor其实反映的box的长宽比这一属性，由真实的box坐标来确定位置。和archor的iou侧重的是box符合那个长宽比，因此可以将放到放到中心都是原点
		的位置来比较，这样就去确定了物体的形状，用长宽比来确定c,同时也确定由每个单元格的哪个box来负责预测部分
        '''
		intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1] #交叉部分面积
        box_area = wh[..., 0] * wh[..., 1] #真实box面积
        anchor_area = anchors[..., 0] * anchors[..., 1]#anchor面积
		
		#iou的shape:(n,9)表示n个box,每个box和9个anchor计算iou
        iou = intersect_area / (box_area + anchor_area - intersect_area) #计算出真实box和anchor的iou，

		#获取每个box和9个anchor的i最大iou的索引,索引范围是[0-8]
        best_anchor = np.argmax(iou, axis=-1)#best_anchor shape:(n,)
        
		#t表示的是box的索引,n表示的是best_anchor的值,也就是9个anchor的索引
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:#该anchor索引是在第l组的anchor里
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')#查看该对象的box的中心坐标x落在哪个单元格
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')#查看该对象的box的中心坐标y落在哪个单元格
                    k = anchor_mask[l].index(n) #k是cell中哪个box负责预测
                    c = true_boxes[b,t, 4].astype('int32')#类别
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]#将真实的box的坐标赋给单元格负责预测的第k个预测值
                    y_true[l][b, j, i, k, 4] = 1 #设置检测对象的置信程度,有对象1,否则0
                    y_true[l][b, j, i, k, 5+c] = 1#赋值类别,对应位1,其余位0

    return y_true

#计算预测box和真实box之间的iou
def box_iou(b1, b2):
    '''

    参数
    ----------
    b1: 预测得到的所有box, shape类似(13,13,3,4)or(26,26,3,4)or(52,52,3,4)这样, xywh
    b2: 真实对象的box, shape=(j, 4), xywh,j表示有多少个含有对象的真实box个数

    Returns
    -------
    iou: tensor, 返回每个真实box和预测box之间的iou,shape类似(13,13,3,j),(26,26,3,j),(52,52,3,j)这样的

    '''

    #增加维度利用broadcast机制进行预测和真实box之间的iou计算
    b1 = K.expand_dims(b1, -2)#b1 shape类似这样(13,13,3,1,4)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half #左上角坐标
    b1_maxes = b1_xy + b1_wh_half #右下角坐标

    #与上面同理,增加维度计算两者之间的iou
    b2 = K.expand_dims(b2, 0) #b2 shape类似(1,j,4)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half #左上角坐标
    b2_maxes = b2_xy + b2_wh_half#右下角坐标
     
	#通过增加维度，两者之间可以通过broadcast机制进行彼此之间的iou计算

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1] #交叉部分面积
    b1_area = b1_wh[..., 0] * b1_wh[..., 1] #预测box面积
    b2_area = b2_wh[..., 0] * b2_wh[..., 1] #真实box面积
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou
'''
计算loss。计算loss的公式参考课yolov1的论文的方式。只是在此基础上有细微改动，就是在没检测到对象的预测box,如果这些预测box和实际
box的iou大于给定的ignore_thresh参数阈值,那么这些box关于检测是否有对象的置信程度这一部分,不参与loss的计算.
个人认为或许是真实的box比较大,在它邻近的周围的cell预测得到的box与这个真实box的iou就会很大,说明这些,这种情况也是正常的,虽然这些box不是负责预测的部分,但是也确实没必要对它们进行惩罚
'''
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''返回损失计算结果

    参数
    ----------
	args:包含yolo的输出(yolo_outputs)和真实值(y_true)两部分
        yolo_outputs: list of tensor, yolo的输出
        y_true: list of array, 通过preprocess_true_boxes方法得到的真实值
    anchors: array, shape=(N, 2), 宽和高
    num_classes: 类别个数
    ignore_thresh: float, the iou阈值,用来判断iou大于这个阈值,则不用进行loss的计算

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 #多少个输出层,因为每个cell分配3个box的预测
    yolo_outputs = args[:num_layers]#获所有yolo模型的输出
    y_true = args[num_layers:]#获取真值
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] #所有输出层对应的anchor索引号
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0])) #即(416, 416)
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)] #即[(13, 13), (26, 26), (52, 52)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    
	#逐个输出层计算loss,最终损失所有输出层的累计和
    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]#获取每个box的是否含有对象的置信程度
        true_class_probs = y_true[l][..., 5:] #获取每个box是哪个类别的概率

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
		'''
		调用yolo_head方法返回grid,每个cell对应左上角的偏移量,重新调整shape的输出raw_pred，
		pred_xy:预测的中心坐标,相对于整张图片大小,做了归一化
		pred_wh:预测的box的宽高,相对于整张图片大小,做了归一化
		'''
        pred_box = K.concatenate([pred_xy, pred_wh]) #将预测得到的box的中心坐标和宽高合并为一条记录,shape:[32,13,13,3,4]

        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid #得到相对于cell大小的box真实的中心偏移量,用来和预测的中心最表计算损失
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])#得到和预测wh来计算损失的真实的wh
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')#包含对象为True,否则为False,用来筛选真实值中含有对象的box
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])#返回的tensor shape[,4](无论多少维)
            iou = box_iou(pred_box[b], true_box)
			#best_iou,预测box与真值box中最大的iou
            best_iou = K.max(iou, axis=-1)#keras.backend.max(x, axis=None, keepdims=False) best_iou shape:[13,13,3]
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
		
		#while_loop循环调用loop_body,查询出所有iou小于ignore_mask的box
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask]) #获取所有图片的所有真实box和预测box中小于ignore_thresh的box有哪,这些box参与损失计算		
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

       
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, K.sigmoid(raw_pred[...,0:2]), from_logits=True)#计算检测到对象的cell中负责预测的box和真实box的中心坐标损失       
		wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])#计算检测到对象的cell中负责预测的box和真实box的宽高损失
       
	   confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask #confidence_loss:计算检测到对象的cell中负责预测的box和真实box之间判断是否含有对象的置信程度之间
			#的误差,还要加上不含有对象的不负责预测box的其余部分,并且这部分和真实box之间的iou小于给定的ignore_thresh的部分,这些参与计算对象检测置信程度的损失
       
	   class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)#计算检测到对象的cell中负责预测的box和真实box的class判断之间的误差
        
		#计算各部分损失累计和的平均
        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
