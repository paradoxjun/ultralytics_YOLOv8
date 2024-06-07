from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

INFTY_COST = 1e+5

# 自由度为 N 的卡方分布的 0.95 分位数表（包含 N=1、...、9 的值）。取自 MATLAB/Octave 的 chi2inv 函数并用作马哈拉诺比斯门控阈值。
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    """
    解决线性分配问题。

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        距离度量函数，接受轨迹和检测列表，以及轨迹和检测的索引列表。
        返回NxM维的成本矩阵，其中元素(i, j)表示给定轨迹索引中第i个轨迹和检测索引中第j个检测之间的关联成本。
    max_distance: (float) 门限值，成本大于这个值的关联将被忽略。
    tracks : (List[track.Track]) 当前时间步预测的轨迹列表。
    detections : (List[detection.Detection]) 当前时间步的检测列表。
    track_indices : (List[int]) 将`cost_matrix`中的行映射到`tracks`中的轨迹的轨迹索引列表。
    detection_indices : (List[int]) 将`cost_matrix`中的列映射到`detections`中的检测的检测索引列表。

    Returns
    -------
    (List[(int, int)], List[int], List[int]) 返回一个包含以下三个条目的元组：
        * 匹配的轨迹和检测索引列表。
        * 未匹配的轨迹索引列表。
        * 未匹配的检测索引列表。
    """
    if track_indices is None:  # 如果未指定轨迹索引，默认所有轨迹
        track_indices = np.arange(len(tracks))
    if detection_indices is None:  # 如果未指定检测索引，默认所有检测
        detection_indices = np.arange(len(detections))
    if len(detection_indices) == 0 or len(track_indices) == 0:  # 没有匹配的情况
        return [], track_indices, detection_indices

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)  # 计算成本矩阵
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5  # 对成本矩阵进行门限处理（避免某些值过大）
    row_indices, col_indices = linear_assignment(cost_matrix)  # 线性分配求解，获取行列索引

    matches, unmatched_tracks, unmatched_detections = [], [], []
    # 找出未匹配的检测
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    # 找出未匹配的轨迹
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    # 处理匹配和门限判断
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections,
                     track_indices=None, detection_indices=None):
    """
    进行级联匹配。

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        距离度量函数，接受轨迹和检测列表，以及轨迹和检测的索引列表。
        返回NxM维的成本矩阵，其中元素(i, j)表示给定轨迹索引中第i个轨迹和检测索引中第j个检测之间的关联成本。
    max_distance : (float) 门限值，成本大于这个值的关联将被忽略。
    cascade_depth: (int) 级联深度，应设置为最大轨迹年龄。
    tracks : (List[track.Track]) 当前时间步预测的轨迹列表。
    detections : (List[detection.Detection]) 当前时间步的检测列表。
    track_indices : (Optional[List[int]]) 将`cost_matrix`中的行映射到`tracks`中的轨迹的轨迹索引列表。默认所有轨迹。
    detection_indices : (Optional[List[int]]) 将`cost_matrix`中的列映射到`detections`中的检测的检测索引列表。默认所有检测。

    Returns
    -------
    (List[(int, int)], List[int], List[int]) 返回一个包含以下三个条目的元组：
        * 匹配的轨迹和检测索引列表。
        * 未匹配的轨迹索引列表。
        * 未匹配的检测索引列表。
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # 没有剩余检测
            break

        track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0:  # 当前级别没有需要匹配的轨迹
            continue

        matches_l, _, unmatched_detections = min_cost_matching(distance_metric, max_distance, tracks, detections,
                                                               track_indices_l, unmatched_detections)  # 进行最小成本匹配
        matches += matches_l

    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))  # 找出未匹配的轨迹

    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices,
                     gated_cost=INFTY_COST, only_position=False):
    """
    根据卡尔曼滤波的状态分布无效化成本矩阵中不可行的条目。

    参数
    ----------
    kf : 卡尔曼滤波器。
    cost_matrix : (ndarray) NxM维的成本矩阵，其中N是轨迹索引的数量，M是检测索引的数量。
        条目(i, j)表示`tracks[track_indices[i]]`和`detections[detection_indices[j]]`之间的关联成本。
    tracks : (List[track.Track]) 当前时间步预测的轨迹列表。
    detections : (List[detection.Detection]) 当前时间步的检测列表。
    track_indices : (List[int]) 将`cost_matrix`中的行映射到`tracks`中的轨迹的轨迹索引列表。
    detection_indices : (List[int]) 将`cost_matrix`中的列映射到`detections`中的检测的检测索引列表。
    gated_cost : (Optional[float]) 成本矩阵中对应于不可行关联的条目将被设置为此值。默认为一个很大的值。
    only_position : (Optional[bool]) 如果为True，则在进行门限判断时只考虑状态分布的x、y位置。默认为False。

    返回
    -------
    (ndarray) 返回修改后的成本矩阵。
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])  # 将检测转换为测量值数组

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)  # 计算门限距离
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost  # 超过门限的条目设置为gated_cost

    return cost_matrix
