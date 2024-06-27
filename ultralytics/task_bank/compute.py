import numpy as np


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


def ioa_bbox_candidates_xyxy(bbox, candidates):
    """
    计算交集比（Intersection over Area，IOA）。
    参数
    ----------
    bbox : (ndarray) 一个边界框，格式为 `(xmin, ymin, xmax, ymax)`。
    candidates : (ndarray) 候选边界框的矩阵（每行一个），格式与 `bbox` 相同。

    返回
    -------
    (ndarray, ndarray)
        返回两个数组：
        - 第一个数组是交集区域相对于 `bbox` 面积的比值。
        - 第二个数组是交集区域相对于每个候选框面积的比值。
    """
    # 计算 bbox 和 candidates 的左上角和右下角坐标
    bbox_tl, bbox_br = bbox[:2], bbox[2:4]
    candidates_tl, candidates_br = candidates[:, :2], candidates[:, 2:4]

    # 计算相交区域的左上角和右下角坐标
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
    np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
    np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)  # 计算相交区域的宽度和高度，若无交集则为0

    area_intersection = wh.prod(axis=1)  # 相交区域的面积
    area_bbox = (bbox_br[0] - bbox_tl[0]) * (bbox_br[1] - bbox_tl[1])  # bbox的面积
    area_candidates = (candidates_br[:, 0] - candidates_tl[:, 0]) * (
                candidates_br[:, 1] - candidates_tl[:, 1])  # candidates的面积

    # 计算 IOA
    ioa_bbox = area_intersection / area_bbox
    ioa_candidates = area_intersection / area_candidates
    iou = area_intersection / (area_bbox + area_candidates - area_intersection)

    return ioa_bbox, ioa_candidates, iou


def ioa_bbox_bbox_xyxy(bbox1, bbox2):
    """
    计算两个边界框之间的交集比（Intersection over Area，IOA）。

    参数
    ----------
    bbox1 : (ndarray) 第一个边界框，格式为 `(xmin, ymin, xmax, ymax)`。
    bbox2 : (ndarray) 第二个边界框，格式为 `(xmin, ymin, xmax, ymax)`。

    返回
    -------
    (float, float, float)
        返回三个值：
        - 第一个值是交集区域相对于 `bbox1` 面积的比值。
        - 第二个值是交集区域相对于 `bbox2` 面积的比值。
        - 第三个值是交并比（IoU）。
    """
    # 计算 bbox1 和 bbox2 的左上角和右下角坐标
    bbox1_tl, bbox1_br = bbox1[:2], bbox1[2:4]
    bbox2_tl, bbox2_br = bbox2[:2], bbox2[2:4]

    # 计算相交区域的左上角和右下角坐标
    tl = np.maximum(bbox1_tl, bbox2_tl)
    br = np.minimum(bbox1_br, bbox2_br)
    wh = np.maximum(0., br - tl)  # 计算相交区域的宽度和高度，若无交集则为0

    area_intersection = np.prod(wh)  # 相交区域的面积
    area_bbox1 = (bbox1_br[0] - bbox1_tl[0]) * (bbox1_br[1] - bbox1_tl[1])  # bbox1的面积
    area_bbox2 = (bbox2_br[0] - bbox2_tl[0]) * (bbox2_br[1] - bbox2_tl[1])  # bbox2的面积

    # 计算 IOA 和 IoU
    ioa_bbox1 = area_intersection / area_bbox1
    ioa_bbox2 = area_intersection / area_bbox2
    iou = area_intersection / (area_bbox1 + area_bbox2 - area_intersection)

    return ioa_bbox1, ioa_bbox2, iou


if __name__ == '__main__':
    # 示例数据
    bbox1 = np.array([50, 50, 150, 150])  # (xmin, ymin, xmax, ymax)
    bbox2 = np.array([100, 100, 200, 200])  # (xmin, ymin, xmax, ymax)

    # 计算IOA
    ioa_bbox1, ioa_bbox2, iou = ioa_bbox_bbox_xyxy(bbox1, bbox2)

    # 打印结果
    print("IOA relative to bbox1:", ioa_bbox1)
    print("IOA relative to bbox2:", ioa_bbox2)
    print("IoU:", iou)
