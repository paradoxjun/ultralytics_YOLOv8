import cv2
import torch
import numpy as np


COCO_keypoint_indices = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_DEFAULT_UPPER_BODY_KEYPOINT_INDICES = (5, 6, 7, 8, 9, 10)          # 上半身的关键点索引
COCO_DEFAULT_CONNECTIONS = ((4, 2), (2, 0), (0, 1), (1, 3), (3, 5))     # 关键点连接顺序（例如：0连接1，1连接2，依此类推）

default_up_body_indices = {
    0: 'left_shoulder',
    1: 'right_shoulder',
    2: 'left_elbow',
    3: 'right_elbow',
    4: 'left_wrist',
    5: 'right_wrist',
}


def get_upper_body_keypoint(data, keypoint_indices=COCO_DEFAULT_UPPER_BODY_KEYPOINT_INDICES):
    # 检查数据是否为空或大小为零
    if data is None or data.shape[0] == 0 or data.shape[1] == 0:
        # 返回一个空的数组，形状为 (0, len(keypoint_indices), 3)
        return np.empty((0, len(keypoint_indices), 3))      # 如果不需要使用置信度，长度设为2

    # 检查 keypoint_indices 是否超出数据的范围
    if max(keypoint_indices) >= data.shape[1]:
        raise IndexError("Keypoint indices are out of bounds for the given data")

    return data[:, keypoint_indices, :]


def image_read(image):
    # 如果 image 是字符串，则尝试读取路径
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"无法读取图像路径: {image}")
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError("image 参数应为字符串路径或 numpy 数组")

    return img


def image_show(image, desc="KeyPoint"):
    cv2.imshow(desc, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_keypoint(image, data, connections=COCO_DEFAULT_CONNECTIONS, point_color=(0, 0, 255), point_radius=4,
                  line_color=(0, 255, 0), line_thickness=2):
    """
    在图片上绘制关键点和连线。
    Args:
        image: 图片源
        data: YOLOv8姿态检测结果
        connections: 连线顺讯
        point_color: 关键点的颜色
        point_radius: 关键点的大小
        line_color: 连线的颜色
        line_thickness: 连线的粗细

    Returns:
        绘制了关键点的图片。
    """
    img = image_read(image)   # 读取图片
    data = data.cpu().numpy() if torch.is_tensor(data) else np.array(data)  # 将张量移动到CPU并转换为numpy数组

    # 绘制关键点
    for person in data:
        # 绘制连接线
        if connections:
            for start_idx, end_idx in connections:
                sta_point = person[start_idx]
                end_point = person[end_idx]
                if (sta_point[0] > 0 or sta_point[1] > 0) and (end_point[0] > 0 and end_point[1] > 0):  # 忽略无效点
                    cv2.line(img, (int(sta_point[0]), int(sta_point[1])),
                             (int(end_point[0]), int(end_point[1])), line_color, line_thickness)

        # 绘制关键点
        for point in person:
            x, y = point[:2]
            if x > 0 or y > 0:  # 忽略无效点
                cv2.circle(img, (int(x), int(y)), point_radius, point_color, -1)

    return img
