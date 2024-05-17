from ultralytics import YOLO

model = YOLO("./weights/yolov8s.pt")

data = "ultralytics/cfg/bank_monitor/data.yaml"

model.train(task="detect", mode="train", data=data, epochs=200, batch=32)

