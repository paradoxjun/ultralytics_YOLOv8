from ultralytics import YOLO


data_path = r"/home/chenjun/code/datasets/bank_monitor/data_05_09"
# Load a model
model = YOLO(model='./runs/detect/train_05_09_01/weights/best.pt', task="detect")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(data_path, stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = dir(result)  # Boxes object for bounding box outputs
    print(result.boxes.cls)
    print("*" * 20)
    # result.save(filename='result.jpg')  # save to disk
