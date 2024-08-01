import torch
import os

# Model
# model = torch.hub.load("ultralytics/yolov5", "yolov5s.pt")  # or yolov5n - yolov5x6, custom
model = torch.hub.load("", "custom", path="yolov5s.pt", source="local")
# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
image_path = "test_image"

image_list = os.listdir(image_path)

model.conf = 0.25  # NMS confidence threshold
iou = 0.45  # NMS IoU threshold
agnostic = False  # NMS class-agnostic
multi_label = False  # NMS multiple labels per box
classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
max_det = 1000  # maximum number of detections per image
amp = False  # Automatic Mixed Precision (AMP) inference

for img in image_list:
    img_path = image_path + "/" + img
    # Inference
    results = model(img_path, size=640)
    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

    df = results.pandas().xyxy[0]  # im1 predictions (pandas)

    print(results.pandas().xyxy[0])
    # 只选择需要的列
    selected_columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name']
    df_selected = df[selected_columns]

    # 将 DataFrame 保存到文本文件，使用逗号分隔
    df_selected.to_csv('text_result/%s.txt'%img.split(".")[0], sep=',', index=False, header=False)







