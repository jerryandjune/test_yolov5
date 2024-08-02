import numpy as np
from PIL import Image
from collections import Counter

# 定义 label 到颜色值的映射
label_map = {
    'background': 0,
    'load': 1,
    'person': 2,
    'tree': 3,
    'build': 4,
    # 添加更多的 label 和对应的颜色值
}

# 读取 PNG 掩码图像
output_path = r"C:\Users\zgdx\Downloads\数据样例\模块二样例\任务2（语义分割）\标注任务样例\提交标注文件示例.png"
mask_image = Image.open(output_path)
mask_array = np.array(mask_image, dtype=np.uint8)

# 创建一个计数器来统计每个像素值
pixel_counts = Counter(mask_array.flatten())

# 打印统计结果
print("Pixel value counts:")
for label_value, count in pixel_counts.items():
    label = None
    # 将像素值映射回标签
    for label_key, value in label_map.items():
        if value == label_value:
            label = label_key
            break
    print(f"Label '{label}': {count} pixels")

