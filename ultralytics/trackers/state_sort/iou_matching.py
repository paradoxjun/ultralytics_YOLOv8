import numpy as np
from ultralytics.trackers.state_sort.linear_assignment import INFTY_COST


def iou(bbox, candidates):
    """
    计算交并比（Intersection over Union，IoU）。

    参数
    ----------
    bbox : (ndarray) 一个边界框，格式为 `(top left x, top left y, width, height)`。
    candidates : (ndarray) 候选边界框的矩阵（每行一个），格式与 `bbox` 相同。

    返回
    -------
    (ndarray) 返回 `bbox` 与每个候选边界框之间的交并比，值在 [0, 1] 之间。分数越高表示 `bbox` 被候选框遮挡的比例越大。
    """
    # 计算 bbox 和 candidates 的左上角和右下角坐标
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl, candidates_br = candidates[:, :2], candidates[:, :2] + candidates[:, 2:]

    # 计算相交区域的左上角和右下角坐标
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
    np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
    np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)  # 计算相交区域的宽度和高度，若无交集则为0

    area_intersection = wh.prod(axis=1)  # 区域的面积
    area_bbox = bbox[2:].prod()  # bbox的面积
    area_candidates = candidates[:, 2:].prod(axis=1)  # candidates的面积

    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    交并比距离度量。

    参数
    ----------
    tracks : List[deep_sort.track.Track] 轨迹列表。
    detections : List[deep_sort.detection.Detection] 检测列表。
    track_indices : Optional[List[int]] 要匹配的轨迹索引列表。默认为所有 `tracks`。
    detection_indices : Optional[List[int]] 要匹配的检测索引列表。默认为所有 `detections`。

    返回
    -------
    (ndarray) 返回一个形状为 [len(track_indices), len(detection_indices)] 的成本矩阵，
    其中条目 (i, j) 是 `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`。
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:  # 如果轨迹更新间隔大于1，则设置为无限大成本
            cost_matrix[row, :] = INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()  # 获取当前轨迹的边界框
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])  # 获取所有候选检测的边界框
        cost_matrix[row, :] = 1. - iou(bbox, candidates)  # 计算交并比并转化为成本

    return cost_matrix
