import os
import yaml
import torch
import cv2
import numpy as np
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


def resize_and_pad(frame, target_size=(800, 800), pad_color=(114, 114, 114)):
    h, w, _ = frame.shape
    target_w, target_h = target_size

    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放图像
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建填充后的图像
    padded_frame = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

    # 计算填充位置
    top = (target_h - new_h) // 2
    bottom = top + new_h
    left = (target_w - new_w) // 2
    right = left + new_w

    # 将缩放后的图像放置在填充后的图像上
    padded_frame[top:bottom, left:right] = resized_frame

    return padded_frame


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


def split_indices(labels, group1_classes=(0, 1, 2, 4), group2_classes=(3,)):
    """
    将标签张量根据给定的类别划分成两部分，并返回对应的索引。

    参数：
    labels (torch.Tensor): 标签张量。
    group1_classes (set): 第一组类别值的集合。
    group2_classes (set): 第二组类别值的集合。

    返回：
    tuple: 包含两个列表，分别为group1和group2的索引。
    """
    group1_indices = [i for i, label in enumerate(labels) if label in group1_classes]
    group2_indices = [i for i, label in enumerate(labels) if label in group2_classes]

    return group1_indices, group2_indices


def apply_indices(tensor, indices):
    """
    根据索引返回张量中的元素。
    Args:
        tensor: 原张量
        indices: 索引

    Returns:
        根据索引返回张量中的元素，保持原张量的shape。
    """
    original_shape = tensor.shape
    num_dims = len(original_shape)

    if indices.numel() == 0:
        empty_shape = list(original_shape)
        empty_shape[0] = 0
        return torch.empty(empty_shape, dtype=tensor.dtype, device=tensor.device)

    selected = tensor[indices]

    if selected.ndimension() == num_dims - 1:
        selected = selected.unsqueeze(0)

    return selected


def split_indices_deepsort(deepsort_outputs, labels):
    """
    根据label拆分deepsort输出，并返回索引字典。

    :param deepsort_outputs: numpy数组，形状为 (n, 6)，其中包含 x1, y1, x2, y2, label, track_ID, confs
    :param labels: list，包含可能的label值
    :return: dict，将标签作为key，如果标签存在，value为对应的索引列表；如果不存在，value为None
    """
    # 初始化结果字典
    result = {label: None for label in labels}

    # 遍历所有的输出，按label分类
    for label in labels:
        indices = np.where(deepsort_outputs[:, 4] == label)[0]
        if indices.size > 0:
            result[label] = indices

    return result


def ioa(bbox1, bbox2):
    """
    计算两个检测框的交集面积比上当前检测框的面积(IOA)
    """
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    inter_x1 = np.maximum(x1, x1_)
    inter_y1 = np.maximum(y1, y1_)
    inter_x2 = np.minimum(x2, x2_)
    inter_y2 = np.minimum(y2, y2_)
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    bbox2_area = (x2_ - x1_) * (y2_ - y1_)
    return inter_area / bbox2_area


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
