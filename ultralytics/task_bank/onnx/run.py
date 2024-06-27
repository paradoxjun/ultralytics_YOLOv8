import cv2
from paddle_pplcnetv2 import PPLCNetv2Predictor
import time


img_path_1 = "/home/chenjun/code/paddleclas/dataset/COCO_val2014_000000000285.jpg"      # 熊
img_path_2 = "/home/chenjun/code/paddleclas/dataset/COCO_val2014_000000000757.jpg"      # 大象
img_path_3 = "/home/chenjun/code/paddleclas/dataset/COCO_val2014_000000002255.jpg"      # 大象

model_path = r'/home/chenjun/code/paddleclas/models/PPLCNetv2_onnx/inference.onnx'
# 加载ONNX模型（使用GPU）
predict = PPLCNetv2Predictor(model_path=model_path, use_cuda=True)

outputs = predict(img_path_1)
print(outputs)

# with open('vectors_4.txt', 'w') as f:
#     np.savetxt(f, outputs.reshape(1, -1))  # reshape 为一行
