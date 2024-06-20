from ultralytics.task_bank.predict import BankDetectionPredictor
import torch


overrides_1 = {"task": "detect",
               "mode": "predict",
               "model": r'E:\Py_Learning\yolo/ultralytics_YOLOv8/weights/yolov8m.pt',
               "verbose": False,
               "classes": [0]
               }

overrides_2 = {"task": "detect",
               "mode": "predict",
               "model": r'E:\Py_Learning\yolo\ultralytics_YOLOv8//weights/best.pt',
               "verbose": False,
               "classes": [0, 1, 2, 3]
               }

predictor_1 = BankDetectionPredictor(overrides=overrides_1)
predictor_2 = BankDetectionPredictor(overrides=overrides_2)
predictors = [predictor_1, predictor_2]

img_path = r'E:\Py_Learning\yolo\ultralytics_YOLOv8\runs\detect\track\image_plot\img_00000.jpg'
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


def transform_and_concat_tensors(tensor_list, k1_v1_dict_list, k2_v2_dict):
    def transform_tensor(tensor, k1_v1_dict):
        original_dtype = tensor.dtype       # 获取输入tensor的数据类型
        original_device = tensor.device     # 获取输入tensor的device
        v1_to_k2 = {v: k for k, v in k2_v2_dict.items()}    # 创建 v1 到 k2 的映射字典
        transformed_list = []               # 创建一个新的列表来存储转换后的值

        for value in tensor:
            v1 = k1_v1_dict[value.item()]   # 获取原模型的

            if v1 not in v1_to_k2:          # 合并后的类别标签不存在
                raise ValueError(f"label [{v1}] from model not found in new labels.")

            k2 = v1_to_k2[v1]
            transformed_list.append(k2)

        transformed_tensor = torch.tensor(transformed_list, dtype=original_dtype, device=original_device)
        return transformed_tensor

    transformed_tensors = [transform_tensor(tensor, k1_v1_dict) for tensor, k1_v1_dict in
                           zip(tensor_list, k1_v1_dict_list)]   # 对每个 tensor 进行转换

    result_tensor = torch.cat(transformed_tensors, dim=0)       # 将转换后的 tensor 按 dim=1 进行连接

    return result_tensor


tensor_list = [p_1.boxes.cls, p_2.boxes.cls]
k1_v2_dict_list = [p_1.names, p_2.names]

res = transform_and_concat_tensors(tensor_list, k1_v2_dict_list, class_name_num_str)
print("二者合并的模型预测结果：", res)
