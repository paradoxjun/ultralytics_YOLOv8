import numpy as np


class FilterPerson:
    def __init__(self, max_history=5, max_missing_time=75, need_valid=False):
        # 初始化信息
        self.max_history = max_history              # 保留最近的帧数
        self.max_missing_time = max_missing_time    # 人可以离开的最大时间
        self.need_valid = need_valid                # 人的出现需要有效（检测更加严格）
        # 历史状态信息
        self.time_since_update = 0          # 最近一次更新（检测到人）
        self.time_since_update_valid = 0    # 最近一次更新（存在有效的人）
        self.person_valid = 0               # 视频中认为有效的人数（占据中心的主要验钞人员，排除误检、边缘无效情况）
        self.last_person_valid_box = np.empty((0, 6), dtype=np.float32)   # 最后一次存在有效人的记录

    def __call__(self, det_person, min_conf=0.4, hwa=(3.5, 3.5, 6.0)):
        """
        用于人的历史信息检测。
        Args:
            det_person: (ultralytics.Result)当前帧Yolov8检测人的结果；
            min_conf: 检测框有效的最小置信度；
            hwa: （高度是宽度的上限倍数， 宽度是高度的上限倍数， 最大面积 / 当前面积的上限比例）
        Returns:
            [有效人的检测框，是否人长时间离开]
        """
        xyxy, conf, cls = det_person.boxes.xyxy.numpy(), det_person.boxes.conf.numpy(), det_person.boxes.cls.numpy()

        self.time_since_update = self.time_since_update + 1 if xyxy.shape[0] == 0 else 0    # 当前帧未检测到人，更新时间+1。
        valid_boxes = self._filter_valid_person(xyxy, conf, cls, hwa[0], hwa[1], hwa[2], min_conf)   # 获取有效的人的检测框
        self.time_since_update_valid = self.time_since_update_valid + 1 if valid_boxes.shape[0] == 0 else 0

        # TODO: 何时返回上一次结果？比如检测有效人数结果[1, 1, 1, 1, 0], [3, 3, 4, 3, 2]，目前只承认一次误检。
        if self.person_valid > valid_boxes.shape[0]:    # 前一次有效人数 > 当前次有效人数
            self.person_valid = valid_boxes.shape[0]    # 有效人数修改为本次人数
            valid_boxes, self.last_person_valid_box = self.last_person_valid_box, valid_boxes   # 本次改上次，上次更新为本次
        else:   # 当前有效人数 >= 前一次的，只需更新就行
            self.person_valid = valid_boxes.shape[0]    # 虽然都有这行，但是不能放判断外，因为会影响判断
            self.last_person_valid_box = valid_boxes

        return valid_boxes, self._check_missing_person()

    @staticmethod
    def _filter_valid_person(xyxy, conf, cls, height_ratio=3.0, width_ratio=3.0, area_ratio=6.0, confidence=0.4):
        """
        返回有效的文本框。
        首先，没有达到confidence的排除，然后超过height_ratio和width_ratio的排除,最后面积小于最大面积1/4的排除
        Args:
            xyxy: ndarray, (N, 4) 检测框
            conf: ndarray, (N,) 检测框的置信度
            cls: ndarray, (N,) 检测框的类别
            height_ratio: 高度是宽度的上限倍数
            width_ratio: 宽度是高度的上限倍数
            area_ratio: 最大面积 / 当前面积的上限比例
            confidence: 检测框的置信度

        Returns:
            valid_boxes: ndarray, (M, 6) 包含[x1, y1, x2, y2, conf, cls]的有效检测框
        """
        if xyxy.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)

        # 筛选置信度
        valid_conf_indices = conf > confidence
        # 计算宽度和高度，并筛选高度和宽度比例
        widths, heights = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
        valid_height_indices, valid_width_indices = heights <= widths * height_ratio, widths <= heights * width_ratio
        # 计算面积
        areas = widths * heights
        max_area = areas.max()
        valid_area_indices = areas >= max_area / area_ratio
        # 综合所有条件
        valid_indices = valid_conf_indices & valid_height_indices & valid_width_indices & valid_area_indices

        if np.sum(valid_indices) == 0:  # 返回空数组，形状为(0, 6)
            return np.empty((0, 6), dtype=np.float32)

        # 筛选有效的检测框和置信度
        valid_xyxy, valid_conf, valid_cls = xyxy[valid_indices], conf[valid_indices], cls[valid_indices]
        # 组合结果
        valid_boxes = np.hstack((valid_xyxy, valid_conf[:, np.newaxis], valid_cls[:, np.newaxis])).astype(np.float32)

        return valid_boxes

    def _check_missing_person(self):
        # 判断指定时间内，人是否消失了
        return (self.time_since_update_valid if self.need_valid else self.time_since_update) > self.max_missing_time
