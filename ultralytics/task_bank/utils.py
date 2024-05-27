import os
import yaml
import torch
from easydict import EasyDict

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


class YamlParser(EasyDict):
    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.safe_load(fo.read()))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.safe_load(fo.read()))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)


def transform_and_concat_tensors(tensor_list, k1_v1_dict_list, k2_v2_dict):
    """
    >>> tensor_list = [p_1.boxes.cls, p_2.boxes.cls]
    >>> k1_v2_dict_list = [p_1.names, p_2.names]
    >>> res = transform_and_concat_tensors(tensor_list, k1_v2_dict_list, class_name_num_str)
    Args:
        tensor_list: 标签张量列表
        k1_v1_dict_list: 原模型标签字典
        k2_v2_dict: 修改后的标签字典
    Returns: 修改后的标签张量
    """
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


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
