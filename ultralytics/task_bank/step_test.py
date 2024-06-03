from ultralytics.task_bank.predict import BankDetectionPredictor
from ultralytics.task_bank.utils import transform_and_concat_tensors, split_indices
import torch


overrides_1 = {"task": "detect",
               "mode": "predict",
               "model": r'/home/chenjun/code/ultralytics_YOLOv8/weights/yolov8s.pt',
               "verbose": False,
               "classes": [0]
               }

overrides_2 = {"task": "detect",
               "mode": "predict",
               "model": r'/home/chenjun/code/ultralytics_YOLOv8/runs/detect/train_bank_05_25_m/weights/best.pt',
               "verbose": False,
               "classes": [0, 1, 2, 3]
               }

predictor_1 = BankDetectionPredictor(overrides=overrides_1)
predictor_2 = BankDetectionPredictor(overrides=overrides_2)
predictors = [predictor_1, predictor_2]

img_path = r'/home/chenjun/code/ultralytics_YOLOv8/runs/detect/track_3/image_plot/img_00000.jpg'
p_1 = predictor_1(source=img_path)[0]
p_2 = predictor_2(source=img_path)[0]

print("官方预训练模型预测结果：", p_1.boxes.cls)
print("自己训练的模型预测结果：", p_2.boxes.cls)

class_name_num_str = {
    0: 'ycj',
    1: 'kx',
    2: 'kx_dk',
    3: 'money',
    4: 'person'
}


tensor_list = [p_1.boxes.cls, p_2.boxes.cls]
k1_v2_dict_list = [p_1.names, p_2.names]

bbox_xywh = torch.cat((p_1.boxes.xywh, p_2.boxes.xywh)).cpu()
res = transform_and_concat_tensors(tensor_list, k1_v2_dict_list, class_name_num_str)

indices_1, indices_2 = split_indices(res)

print("二者合并的模型预测结果：", res)
print(indices_1, indices_2)
print(bbox_xywh[indices_1])
print(bbox_xywh[indices_2])
print(res[indices_1])
print(res[indices_2])
