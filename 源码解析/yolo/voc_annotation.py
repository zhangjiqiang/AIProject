import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]#数据集合路径
#类别名称集合
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#从数据集中读取boundbox，每个boundbox是[minx,miny,maxx,maxy]左上角和右下角坐标以及类别id，保存到文件中
def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)#获取类别id
        xmlbox = obj.find('bndbox')#获取对象框选的box信息
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))#获取每张图片的每个box的左上角和右下角坐标信息
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))#将每个box的坐标信息和类别id写入，用逗号分隔。后面读取数据时候就用split(',')来拆分数据

wd = getcwd()

for year, image_set in sets:#遍历训练集，验证集和测试集的路径
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split() #根据训练集，验证集和测试集的路径来读取其图片id数据集
    list_file = open('%s_%s.txt'%(year, image_set), 'w')#保存数据路径
    for image_id in image_ids:#遍历每张图片id
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))#将年份和图片id结合形成训练集，验证集和测试集的图片路径集合，保存到对应文件中
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')#每张图片占据一行
    list_file.close()

