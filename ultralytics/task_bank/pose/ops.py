import numpy as np
import torch
from ultralytics.task_bank.compute import ioa_bbox_bbox_xyxy


def is_center_in_box(box1, box2):
    """
    计算一个检测框box1的中心是否在另一个box2中，因为是俯视视角，所以如果在的话，很可能是nms不行的结果。
    Args:
        box1: 待选框1
        box2: 待选框2

    Returns:
        bool
    """
    center = (box1[:2] + box1[2:4]) / 2
    return box2[0] <= center[0] <= box2[2] and box2[1] <= center[1] <= box2[3]


def filter_boxes_ioa(boxes, confidences, confidence_threshold=0.25, ioa_threshold=0.7):
    """
    过滤掉无效的框。
    Args:
        boxes: 一组人的检测框。
        confidences: 人检测框对应的置信度。
        confidence_threshold: 置信度阈值，0.25为极大值抑制默认阈值。
        ioa_threshold: 认为重合的ioa阈值。

    Returns:
        有效人检测框的索引。
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    if isinstance(confidences, torch.Tensor):
        confidences = confidences.cpu().numpy()

    is_valid = np.ones(num_boxes := len(boxes), dtype=bool)   # 标记每个框是否被过滤

    for i in range(num_boxes):
        if confidences[i] < confidence_threshold:
            is_valid[i] = False
            continue

        for j in range(i + 1, num_boxes):
            if confidences[j] < confidence_threshold:
                is_valid[j] = False
                continue

            if is_center_in_box(boxes[i], boxes[j]) or is_center_in_box(boxes[j], boxes[i]):
                #
                ioa1, ioa2, _ = ioa_bbox_bbox_xyxy(boxes[i], boxes[j])
                if ioa1 > ioa_threshold or ioa2 > ioa_threshold:
                    if confidences[i] > confidences[j]:
                        is_valid[j] = False
                    else:
                        is_valid[i] = False
                        break  # 当前框已经被过滤，跳出内循环

    return is_valid


if __name__ == '__main__':
    Boxes = np.array([[337, 188, 503, 391], [68, 241, 178, 337], [668, 208, 779, 281], [60, 204, 179, 337]])
    Confidences = np.array([0.82838, 0.67665, 0.65202, 0.41285])

    a = filter_boxes_ioa(Boxes, Confidences)
    print(a)
    b = Confidences[a]
    print(b)

    box1 = np.array([68, 226, 192, 420])
    box2 = np.array([40, 207, 164, 281])
    res = ioa_bbox_bbox_xyxy(box1, box2)
    print(res)
