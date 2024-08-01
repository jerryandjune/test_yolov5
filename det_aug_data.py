# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:35:45 2023

@author: Administrator
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
# 引入一个新的目标检测库
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm

ia.seed(1)


# 读取出图像中的目标框
def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id), encoding='utf-8')
    print(in_file)
    tree = ET.parse(in_file)
    print(tree)
    root = tree.getroot()
    print(root)
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值
        print(object)
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    # bndbox = root.find('object').find('bndbox')
    return bndboxlist  # 以多维数组的形式保存


# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str(image_id) + "_aug" + '.xml'))


def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
    in_file = open(os.path.join(root, str(image_id) + '.xml'), encoding='utf-8')  # 读取原来的xml文件
    tree = ET.parse(in_file)  # 读取xml文件
    xmlroot = tree.getroot()
    index = 0
    # 将bbox中原来的坐标值换成新生成的坐标值
    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        # xmin = int(bndbox.find('xmin').text)
        # xmax = int(bndbox.find('xmax').text)
        # ymin = int(bndbox.find('ymin').text)
        # ymax = int(bndbox.find('ymax').text)

        # 注意new_target原本保存为高维数组
        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str(image_id) + "_aug_" + str(id) + '.xml'))
    # tree.write(os.path.join(saveroot, str(image_id) + "_baseaug_" + '.xml'))


# 处理文件
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


if __name__ == "__main__":
    root_path = 'D:/1_dataset/gangshen_det'
    IMG_DIR = '%s/ori_image' % root_path
    XML_DIR = '%s/ori_voc_xml' % root_path

    # 存储增强后的影像文件夹路径
    AUG_IMG_DIR = '%s/images_aug' % root_path
    mkdir(AUG_IMG_DIR)
    # 存储增强后的XML文件夹路径
    AUG_XML_DIR = '%s/voc_xml_aug' % root_path
    mkdir(AUG_XML_DIR)

    AUGLOOP = 9  # 每张影像增强的数量

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    # 定义数据增强方式，这里随机使用水平翻转和光暗方式
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 50% 图像水平翻转
        iaa.Flipud(0.5),  # 50% 图像垂直翻转
        iaa.Multiply((0.9, 1.1)),  # 改变明亮度
        iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))),  # 使用高斯白噪声添加随机噪声
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0))),  # 使用sigma介于0.0和3.0之间的高斯核模糊图像
        iaa.Sometimes(0.5, iaa.MultiplyAndAddToBrightness(mul=(0.75, 1.25), add=(-10, 10))),
        # 50% 概率 随机改变对比度的大小，并将改变应用于每个 RGB 通道
        # iaa.Sometimes(0.5, iaa.Affine(translate_px={"x": 10, "y": 10}, scale=(0.8, 0.9), rotate=(-0, 0))),
        # 50% 概率对图像进行 translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
        iaa.Sometimes(0.5, iaa.GammaContrast((0.75, 1.25))),  # 50% 概率对图像进行 Gamma 对比度调整
    ])

    # 得到当前运行的目录和目录当中的文件，其中sub_folders可以为空
    for root, sub_folders, files in os.walk(XML_DIR):
        # 遍历没一张图片
        for name in tqdm(files):
            print(name)
            try:
                bndbox = read_xml_annotation(XML_DIR, name)
                for epoch in range(AUGLOOP):
                    seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机

                    # 读取图片
                    img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                    print(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                    img = np.array(img)

                    # bndbox 坐标增强，依次处理所有的bbox
                    for i in range(len(bndbox)):
                        bbs = ia.BoundingBoxesOnImage([
                            ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                        ], shape=img.shape)

                        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                        boxes_img_aug_list.append(bbs_aug)

                        # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                        new_bndbox_list.append([int(bbs_aug.bounding_boxes[0].x1),
                                                int(bbs_aug.bounding_boxes[0].y1),
                                                int(bbs_aug.bounding_boxes[0].x2),
                                                int(bbs_aug.bounding_boxes[0].y2)])
                    # 存储变化后的图片
                    image_aug = seq_det.augment_images([img])[0]
                    path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                    # path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_baseaug_" + '.jpg')
                    # image_auged = bbs.draw_on_image(image_aug, thickness=0)
                    Image.fromarray(image_aug).save(path, quality=100)

                    # 存储变化后的XML
                    change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR, epoch)
                    # print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                    new_bndbox_list = []
            except:
                print('处理出错：', name)
