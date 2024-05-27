from ultralytics import YOLO

model = YOLO("./weights/yolov8m.pt")
data = "ultralytics/cfg/bank_monitor/data.yaml"

model.train(task="detect", mode="train", data=data, epochs=500, batch=12)
