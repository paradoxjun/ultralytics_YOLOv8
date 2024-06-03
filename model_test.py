from ultralytics.task_bank.predict import BankDetectionPredictor
import cv2
# from ultralytics.utils import yaml_load
from utils_my.det_res_ops import image_show
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import xywh2xyxy


model = '/home/chenjun/code/ultralytics_YOLOv8/runs/detect/train_bank_05_21_m/weights/best.pt'
# yaml_path = '/home/chenjun/code/ultralytics_YOLOv8/ultralytics/cfg/default.yaml'
img_path = '/home/chenjun/code/datasets/bank_monitor/data_without_neg/val/images/87203377_1688924277.jpg'

# cfg = yaml_load(yaml_path)
# print(cfg)

overrides = {"task": "detect",
             "mode": "predict",
             "model": model,
             "verbose": False
             }


predictor = BankDetectionPredictor(overrides=overrides)

img = cv2.imread(img_path)
print(img.shape)
image_show(img)
lb = LetterBox(auto=True)
img_resize = lb(image=img)
print(img_resize.shape)
image_show(img_resize, "LetterBox")
# img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

# image_resized = cv2.resize(img, (640, 640))
# image_display(image_resized)
# print(f"{image_resized.shape}, image_tensor: {image_resized[0, :, :]}")
#
#
# tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0)
# tensor = torch.cat([tensor, tensor], dim=0) / 255.0
# print(f"{tensor.shape, tensor[0].shape}, image_tensor: {tensor[:, :, 0]}")
# image_display(tensor.squeeze(0))

a = predictor(source=img_resize)
print(len(a))
print(dir(a[0]))
print(a[0].boxes)


img = a[0].orig_img
image_show(img)
xywh = a[0].boxes.xywh
xyxy = xywh2xyxy(xywh)


for i, bbox in enumerate(xyxy):
    # 转换为CPU上的NumPy数组
    bbox = bbox.cpu().numpy().astype(int)
    x1, y1, x2, y2 = bbox

    # 确保坐标在图像范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(a[0].orig_shape[1], x2)
    y2 = min(a[0].orig_shape[0], y2)

    # 截取图像区域
    cropped_image = a[0].orig_img[y1:y2, x1:x2]

    # 显示图像区域
    cv2.imshow(f'Cropped Image {i+1}', cropped_image)


# for i, bbox in enumerate(new_boxes):
#     # bbox = xywh2xyxy(bbox)
#     # 转换为CPU上的NumPy数组
#     bbox = bbox.cpu().numpy().astype(int)
#     x1, y1, x2, y2 = bbox
#
#     # 确保坐标在图像范围内
#     x1 = max(0, x1)
#     y1 = max(0, y1)
#     x2 = min(640, x2)
#     y2 = min(640, y2)
#
#     # 截取图像区域
#     cropped_image = img_resize[y1:y2, x1:x2]
#     print(cropped_image.shape)
#
#     # 显示图像区域
#     cv2.imshow(f'Cropped Image {i+1}', cropped_image)
#
# 4. 等待用户按键并关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
