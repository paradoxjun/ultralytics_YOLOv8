import numpy as np
from collections import deque
from ultralytics.task_bank.compute import ioa_bbox_candidates_xyxy, ioa_bbox_bbox_xyxy


class KX:
    def __init__(self, idx_frame, xyxy, track_id, conf, label,
                 label_open=2, min_conf=0.3, max_history=10, max_interval=75):
        # 初始化信息
        self.last_update_frame = idx_frame  # 最后更新帧，bytetrack_output: x1,y1,x2,y2,track_id,conf,label,order
        self.xyxy = xyxy                    # 检测框
        self.track_id = track_id            # 当前跟踪ID
        self.conf = conf                    # 置信度
        self.label = label                  # 标签
        self.label_open = label_open        # 表示款箱打开的标签值
        self.min_conf = min_conf            # 可加入历史信息的置信度下限
        self.max_history = max_history      # 最大保留的历史帧数
        self.max_interval = max_interval    # 最大间隔
        # 历史状态信息
        self.desc = ""                      # 描述信息
        self.time_since_update = 0          # 距离上次更新的间隔
        self.time_since_occlusion = 0       # 遮挡持续时间
        self.occlusion = False              # 是否发生遮挡
        self.is_open = label == label_open  # 打开(true) or 关闭(False)
        self.history = deque([[idx_frame, xyxy, conf, label, self.is_open]], maxlen=self.max_history)     # 保存的历史信息

    def update(self, idx_frame=None, xyxy=None, conf=None, label=None):
        is_info_integrate = all(param is not None for param in [idx_frame, xyxy, conf, label])
        if is_info_integrate and conf > self.min_conf:    # 置信度高，历史信息作用小，当前结果加入历史信息
            self._add_to_history(idx_frame, xyxy, conf, label)  # 输入参数完整，并且置信度高，则加入历史信息

            self.last_update_frame = idx_frame      # 最后更新的帧
            self.time_since_update = 0              # 更新时间设置为0
            self.time_since_occlusion = 0           # 遮挡时间设置为0
            self.occlusion = False                  # 没有遮挡
        else:
            self.last_update_frame = idx_frame  # 最后更新的帧
            self.time_since_update += 1

        # TODO: 开闭状态暂时不加入
        return self.xyxy, self.track_id, self.conf, self.label  # , self.is_open

    def is_occlusion(self, output_object=None, occlusion_threshold=0.5):
        if output_object is None or output_object.shape[0] == 0:
            self.time_since_occlusion = 0   # 遮挡时间设置为0
            self.occlusion = False          # 没有遮挡
        else:
            ioa_kx, ioa, _ = ioa_bbox_candidates_xyxy(self.xyxy, output_object)  # (款箱∩人) / 款箱

            if max(ioa_kx) > occlusion_threshold:       # 重合比例过高，说明很可能遮挡了
                self.time_since_occlusion += 1
                self.occlusion = True
            else:
                self.time_since_occlusion = 0
                self.occlusion = False

        return self.occlusion

    @property
    def is_valid(self):
        # TODO: 设计判定为无效的规则
        return self.min_conf <= self.conf     # 置信度小于最低阈值

    def need_delete(self):
        # 需要删除的情况，没有遮挡，而且无效
        return not self.occlusion and self.is_valid

    def _add_to_history(self, idx_frame, xyxy, conf, label):
        while self.history and idx_frame - self.history[0][0] > self.max_interval:  # 删除超过15帧的历史信息
            self.history.popleft()

        self.history.append([idx_frame, xyxy, conf, label, label == self.label_open])
        max_confidence_index = max(range(len(self.history)), key=lambda i: self.history[i][2])     # 返回最大置信度检测结果
        self.xyxy = self.history[max_confidence_index][1]
        self.conf = self.history[max_confidence_index][2]
        self.label = self.history[max_confidence_index][3]
        self.is_open = self.history[-1][4]

    def add_desc(self, desc):
        self.desc = desc


class KXTracker:
    def __init__(self):
        self.nums = 0                   # 历史追踪到的款箱数量
        self.track_id_map = dict()      # 将追踪ID映射成款箱的序号，字典：{track_id: real_id}
        self.money_boxes = dict()       # 款箱的实际ID，字典: {real_id: KX类}
        self.id_set = set()             # 当前检测到的ID，集合: {real_id}

    def update(self, idx_frame, kx_results, output_person, ioa_threshold=0.5):
        for kx in kx_results:
            x1, y1, x2, y2, track_id, conf, label, _ = kx  # bytetrack_output: x1,y1,x2,y2,track_id,conf,label,order
            xyxy = [x1, y1, x2, y2]

            # 使用目标追踪算法获取的track_id已经存在，且是有效目标，直接更新
            if track_id in self.track_id_map and self.track_id_map[track_id] in self.id_set:
                real_id = self.track_id_map[track_id]
                self.money_boxes[real_id].update(idx_frame, xyxy, conf, label)
            else:
                # 不存在，则遍历self.money_boxes已经存在的验钞机，计算和他们的IOA，判断是否是同一个。
                # 是，则同一台则将 {track_id: real_id}加入self.track_id_map，并update这个款箱。
                # 不是，则建立一个新的款箱对象，self.nums+1，real_id=self.nums，id_set里加入real_id
                found = False
                for real_id, boxes in self.money_boxes.items():
                    ioa_1, ioa_2, _ = ioa_bbox_bbox_xyxy(xyxy, boxes.xyxy)
                    if ioa_1 > ioa_threshold or ioa_2 > ioa_threshold:  # 假设IOA大于0.7认为是同一个款箱
                        self.track_id_map[track_id] = real_id
                        boxes.update(idx_frame, xyxy, label, conf)
                        found = True
                        break

                if not found:   # 新的款箱
                    if track_id in self.track_id_map:
                        real_id = self.track_id_map[track_id]
                    else:
                        self.nums += 1
                        real_id = self.nums
                    self.track_id_map[track_id] = real_id
                    self.money_boxes[real_id] = KX(idx_frame, xyxy, track_id, conf, label)
                    self.id_set.add(real_id)

        boxes_valid, boxes_occluded = self.get_money_boxes(idx_frame, output_person, ioa_threshold)

        return boxes_valid, boxes_occluded

    def get_money_boxes(self, idx_frame, output_person, ioa_threshold):
        # 清理未在本帧中检测到的ID，需要检测是否存在人的遮挡，即人和未检测到的款箱的ioa，不存在遮挡则超过15帧删除，否则保留
        boxes_valid = []         # 检测到的有效款箱（没遮挡+置信度高）
        boxes_occluded = []      # 被遮挡的有效款箱（有遮挡+置信度高）
        boxes_to_remove = []     # 需要删除的款箱（无遮挡+置信度低）

        for real_id, boxes in self.money_boxes.items():
            if boxes.last_update_frame == idx_frame:  # 当前更新过的款箱
                if boxes.is_valid:   # 置信度高的
                    boxes_valid.append([*boxes.xyxy, boxes.track_id, boxes.conf, boxes.label, real_id])
            else:   # 没更新的
                occluded = boxes.is_occlusion(output_object=output_person, occlusion_threshold=ioa_threshold)  # 遮挡
                valid = boxes.is_valid   # 实际还要考虑更新时间（因为最多保存1秒，这里就不判断了）

                if not occluded:    # 无遮挡
                    # TODO: 长时间更新也没删除
                    # if idx_frame - boxes.last_update_frame > boxes.max_interval:  # 长时间没更新就删除
                    #     boxes_to_remove.append(real_id)
                    if valid:     # 置信度够高（可能是漏检）
                        boxes_valid.append([*boxes.xyxy, boxes.track_id, boxes.conf, boxes.label, real_id])

                elif valid:     # 有遮挡且有效
                    boxes_occluded.append([*boxes.xyxy, boxes.track_id, boxes.conf, boxes.label, real_id])

        for real_id in boxes_to_remove:     # 移除实际ID无效的条目
            del self.money_boxes[real_id]
            self.id_set.discard(real_id)

        if boxes_to_remove:     # 移除没追踪到的ID条目
            track_id_to_remove = []

            for k, v in self.track_id_map.items():
                if v not in self.id_set:
                    track_id_to_remove.append(k)

            for k in track_id_to_remove:
                del self.track_id_map[k]

        if len(boxes_valid) == 0:
            boxes_valid = np.empty((0, 8), dtype=np.float32)
        else:
            boxes_valid = np.array(boxes_valid, dtype=np.float32)

        if len(boxes_occluded) == 0:
            boxes_occluded = np.empty((0, 8), dtype=np.float32)
        else:
            boxes_occluded = np.array(boxes_occluded, dtype=np.float32)

        return boxes_valid, boxes_occluded
