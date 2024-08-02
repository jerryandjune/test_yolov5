import json
import numpy as np
from PIL import Image, ImageDraw

def labelme_json_to_mask(json_path, output_path, label_map):
    with open(json_path) as f:
        data = json.load(f)

    # 获取图像的尺寸
    imageWidth = data['imageWidth']
    imageHeight = data['imageHeight']

    # 创建一个新的空白图像
    mask = Image.new('L', (imageWidth, imageHeight), 0)
    draw = ImageDraw.Draw(mask)

    # 绘制每个标注区域
    for shape in data['shapes']:
        points = shape['points']
        label = shape['label']
        if label in label_map:
            color = label_map[label]
            polygon = [(x, y) for x, y in points]
            draw.polygon(polygon, outline=color, fill=color)

    # 将PIL图像转换为NumPy数组
    mask = np.array(mask, dtype=np.uint8)

    # 保存为PNG文件
    Image.fromarray(mask).save(output_path)

# 使用示例
json_path = r"E:\task\task4\before\001.json"
output_path = r'E:\task\task4\before\001_mask.png'

# 定义label到颜色值的映射
label_map = {
    'background': 0,
    'load': 1,
    'person': 2,
    'tree': 3,
    'build': 4,
    # 添加更多的label和对应的颜色值
}

labelme_json_to_mask(json_path, output_path, label_map)

