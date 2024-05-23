import cv2
from ultralytics import YOLO
import time


# 加载训练好的模型
model = YOLO(model='./runs/detect/train_bank_05_18_s/weights/last.pt', task="detect")  # pretrained YOLOv8n model

# 打开视频文件
video_path = r"/home/chenjun/code/datasets/bank_monitor/银行柜台监控_1.mp4"
cap = cv2.VideoCapture(video_path)


# 获取视频的帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入对象
out = cv2.VideoWriter('output_video_05_18_s.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧转换为RGB格式
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 推理
    results = model(img)

    # 在帧上绘制检测结果
    for result in results:  # 每个结果的格式: [x1, y1, x2, y2, confidence, class]
        for i, xyxy in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = xyxy
            conf = (
                result.boxes.conf)[i]
            cls = result.boxes.cls[i]

            label = f'{result.names[int(cls)]} {float(conf):.2f}'
            color = (0, 255, 0)  # Green color for bounding boxes

            # 绘制边界框和标签
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 写入处理后的帧到输出视频
    out.write(frame)

    # 显示处理后的帧（可选）
    cv2.imshow('Frame', frame)
    # time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
