import json
from collections import defaultdict
#解析coco数据集中图像分割相关数据。主要是获得每张图片中所有检测到对象的box，显示其左上和右下坐标，以及对象的类别

name_box_id = defaultdict(list)
id_name = dict()
f = open(
    "mscoco2017/annotations/instances_train2017.json",
    encoding='utf-8')
data = json.load(f)#加载先关数据文件

annotations = data['annotations']
for ant in annotations:
    id = ant['image_id']
    name = 'mscoco2017/train2017/%012d.jpg' % id
    cat = ant['category_id']#获取类别id
    
	#根据获取的category_id来映射真正的类别id
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11

    name_box_id[name].append([ant['bbox'], cat]) #以图片路径作为字典的key存储每张图片检测到的box信息和类别id

f = open('train.txt', 'w')#将图片的box和对象类别写入到train.txt文件中
for key in name_box_id.keys():
    f.write(key)
    box_infos = name_box_id[key]
    for info in box_infos:
        x_min = int(info[0][0])#左上角x坐标
        y_min = int(info[0][1])#左上角y坐标
        x_max = x_min + int(info[0][2])#右下角x坐标
        y_max = y_min + int(info[0][3])#右下角y坐标

        box_info = " %d,%d,%d,%d,%d" % (
            x_min, y_min, x_max, y_max, int(info[1]))#一行一张图片和该图片所有检测对象的box和对应的类别
        f.write(box_info)
    f.write('\n')
f.close()

 '''
 数据写入格式: image_file_path_1 box1 classx box2 classx... boxN classx`
                  image_file_path_2 box1 classx box2 classx... boxN classx`
				  image_file_path_3 box1 classx box2 classx... boxN classx`
				          ::::::::
				          ::::::::
				  image_file_path_n box1 classx box2 classx... boxN classx`
'''