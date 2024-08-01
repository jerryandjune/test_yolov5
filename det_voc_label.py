# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id, dataset_path):
    in_file = open('%s/voc_xml/%s.xml' % (dataset_path, image_id), encoding='UTF-8')
    out_file = open('%s/labels/%s.txt' % (dataset_path, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    print('size', size)
    print('w', w)
    print('h', h)
    # filename = root.find('path').text[-3:]
    # if filename == 'peg':
    #     filename = 'jpeg'
    # elif filename == 'PEG':
    #     filename = 'JPEG'
    print('image_id:', image_id)
    for obj in root.iter('object'):
        print('obj', obj)
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        # return filename


sets = ['train', 'val']
# classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']  # 改成自己的类别   bottle,bias_cap,becate_cap,no_cap,ok_cap
# abs_path = os.getcwd()
# print(abs_path)
# wd = getcwd()
classes = ['fire']

dataset_path = 'E:/task/task3/yolov5-master/VOCData'

for image_set in sets:
    if not os.path.exists('%s/labels' % dataset_path):
        os.makedirs('%s/labels' % dataset_path)
    image_ids = open('%s/ImageSets/Main/%s.txt' % (dataset_path, image_set)).read().strip().split()
    list_file = open('%s/%s.txt' % (dataset_path, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/images/%s.jpg\n' % (dataset_path, image_id))
        convert_annotation(image_id, dataset_path)
    list_file.close()
