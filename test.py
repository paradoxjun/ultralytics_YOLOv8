from ultralytics import YOLO


#  data_path = r"/home/chenjun/code/datasets/bank_monitor/data_without_neg/val/images/114588_1248155509.jpg"
data_path = r"/home/chenjun/code/datasets/bank_monitor/银行柜台监控_1.mp4"
# Load a model
model = YOLO(task="detect", model='./runs/detect/train_bank_06_14_m/weights/best.pt')  # pretrained YOLOv8n model
# model = YOLO(task="detect", model='./weights/yolov8m.pt')  # pretrained YOLOv8n model
results = model.track(source=data_path, show=True, save=False, stream=True)

"""
查看结果属性：dir(result)
['boxes', 'cpu', 'cuda', 'keypoints', 'masks', 'names', 'new', 'numpy', 'obb',
 'orig_img', 'orig_shape', 'path', 'plot', 'probs', 'save', 'save_crop', 'save_dir',
  'save_txt', 'show', 'speed', 'summary', 'to', 'tojson', 'update', 'verbose']
查看boxes属性：dir(boxes)
['cls', 'conf', 'cpu', 'cuda', 'data', 'id', 'is_track', 'numpy', 'orig_shape',
'shape', 'to', 'xywh', 'xywhn', 'xyxy', 'xyxyn'
"""

for i, result in enumerate(results):
    print(i, result.boxes.xywh)
    print(result.boxes.id)
    print(result.boxes.cls)
