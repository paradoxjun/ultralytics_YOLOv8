from ultralytics.task_bank.predict import BankDetectionPredictor

# from ultralytics.utils import yaml_load


model = '/home/chenjun/code/ultralytics_YOLOv8/weights/yolov8m.pt'
# yaml_path = '/home/chenjun/code/ultralytics_YOLOv8/ultralytics/cfg/default.yaml'
img_path = '/home/chenjun/code/ultralytics_YOLOv8/ultralytics/assets/bus.jpg'

# cfg = yaml_load(yaml_path)
# print(cfg)

overrides = {"task": "detect",
             "mode": "predict",
             "model": model,
             "verbose": False,
             "classes": 0}

predictor = BankDetectionPredictor(overrides=overrides)

a = predictor(source=img_path)
print(dir(a[0]))
print(a[0].boxes)
