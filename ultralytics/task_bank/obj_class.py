import torch
import numpy as np
from collections import deque
from ultralytics.task_bank.utils import ioa


def filter_person_boxes(xyxy, conf, height_ratio=3.0, width_ratio=3.0, area_ratio=5.0, confidence=0.35):
    """
    返回有效的文本框索引。
    首先，没有达到confidence的排除，然后超过height_ratio和width_ratio的排除,最后面积小于最大面积1/4的排除
    Args:
        xyxy: tensor, (N, 4) 检测框
        conf: tensor, (N, 1) 检测框的置信度
        height_ratio: 高度是宽度的上限倍数
        width_ratio: 宽度是高度的上限倍数
        area_ratio: 最大面积 / 当前面积的上限比例
        confidence: 检测框的置信度

    Returns:
        valid_indices: tensor, (M,) 有效的文本框索引
    """
    valid_conf_indices = conf >= confidence         # 筛选置信度

    widths, heights = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]      # 筛选高度和宽度比例
    valid_height_indices = heights <= widths * height_ratio
    valid_width_indices = widths <= heights * width_ratio

    areas = widths * heights    # 筛选面积比例
    max_area = areas.max()
    valid_area_indices = areas >= max_area / area_ratio

    # 综合所有条件
    valid_indices = valid_conf_indices & valid_height_indices & valid_width_indices & valid_area_indices

    return torch.nonzero(valid_indices).squeeze()


class BankDetObject:
    """
    返回目标的 [检测框, 置信度, 追踪ID]
    """
    def __init__(self, idx_frame, track_id, xyxy, label, confidence, min_confidence=0.5, max_history=5):
        self.last_update_frame = idx_frame  # 最后更新帧
        self.track_id = track_id            # 当前跟踪ID
        self.xyxy = xyxy                    # 检测框
        self.label = label                  # 标签
        self.confidence = confidence        # 置信度
        self.max_history = max_history      # 最大保留的历史帧数
        self.desc = [idx_frame, None]       # 描述信息
        self.min_confidence = min(min_confidence, confidence)       # 可加入历史信息的置信度下限
        self.history = deque([[idx_frame, xyxy, label, confidence]], maxlen=self.max_history)     # 保存的历史信息

    def update(self, idx_frame=None, xyxy=None, label=None, confidence=None):
        if not (idx_frame is None or confidence < self.min_confidence):    # 置信度高，历史信息作用小，当前结果加入历史信息
            self.add_to_history(idx_frame, xyxy, label, confidence)
            self.last_update_frame = idx_frame

        return self.xyxy, self.label, self.confidence, self.track_id                    # 返回最大置信度对应的检测结果

    def add_to_history(self, idx_frame, xyxy, label, confidence):      # 删除超过15帧的历史信息，相当于记录1秒内最有用的几帧
        self.history = deque([info for info in self.history if idx_frame - info[0] <= 15], maxlen=self.max_history)

        if len(self.history) >= self.max_history:
            min_confidence_index = self.get_min_confidence_index()      # 找到置信度最小的历史信息索引
            if confidence > self.history[min_confidence_index][-1]:     # 如果最小置信度也比当前的大，当前帧的置信度就无效，不加入
                self.history[min_confidence_index] = [idx_frame, xyxy, label, confidence]
        else:
            self.history.append([idx_frame, xyxy, label, confidence])

        max_confidence_index = self.get_max_confidence_index()
        self.xyxy = self.history[max_confidence_index][1]
        self.label = self.history[max_confidence_index][2]
        self.confidence = self.history[max_confidence_index][3]

    def get_max_confidence_index(self):  # 返回最大置信度索引
        return max(range(len(self.history)), key=lambda i: self.history[i][-1])

    def get_min_confidence_index(self):  # 返回最小置信度索引
        return min(range(len(self.history)), key=lambda i: self.history[i][-1])

    def add_desc(self, desc):
        self.desc = desc


class MoneyCounterTracker:
    def __init__(self):
        self.nums = 0                   # 历史追踪到的点钞机数量
        self.track_id_map = dict()      # 将追踪ID映射成点钞机的序号，字典：{track_id: real_id}
        self.money_counters = dict()    # 点钞机的实际ID，字典: {real_id: MoneyCounter类}
        self.id_set = set()             # 当前检测到的ID，集合：{real_id}

    def update(self, idx_frame, ycj_results, ren_result, ioa_threshold=0.5):
        for ycj in ycj_results:
            x1, y1, x2, y2, label, track_id, conf = ycj
            xyxy = [x1, y1, x2, y2]

            # 使用deepsort获取的track_id已经存在，且是有效目标，直接更新
            if track_id in self.track_id_map and self.track_id_map[track_id] in self.id_set:
                real_id = self.track_id_map[track_id]
                self.money_counters[real_id].update(idx_frame, xyxy, label, conf)
            else:
                # 不存在，则遍历self.money_counters已经存在的验钞机，计算和他们的IOU或者距离，判断是否是同一台。
                # 是，则同一台则将 {track_id: real_id}加入self.track_id_map，并update这台验钞机
                # 不是，则建立一个新的验钞机对象，self.nums+1，real_id=self.nums，id_set里加入real_id
                found = False
                for real_id, counter in self.money_counters.items():
                    if ioa(counter.xyxy, xyxy) > ioa_threshold:  # 假设IOA大于0.5认为是同一台验钞机
                        self.track_id_map[track_id] = real_id
                        counter.update(idx_frame, xyxy, label, conf)
                        found = True
                        break

                if not found:   # 新的验钞机
                    if track_id in self.track_id_map:
                        real_id = self.track_id_map[track_id]
                    else:
                        self.nums += 1
                        real_id = self.nums
                    self.track_id_map[track_id] = real_id
                    self.money_counters[real_id] = BankDetObject(idx_frame, track_id, xyxy, label, conf)
                    self.id_set.add(real_id)

        self.cleanup(idx_frame, ren_result, ioa_threshold)

        valid_counters = []     # 返回所有有效验钞机
        for real_id, counter in self.money_counters.items():
            valid_counters.append([*counter.xyxy, counter.label, real_id, counter.confidence])

        if len(valid_counters) == 0:
            return np.empty((0, 7), dtype=np.int32)

        return np.array(valid_counters, dtype=np.int32)

    def cleanup(self, idx_frame, ren_result, ioa_threshold=0.5):
        # 清理未在本帧中检测到的ID，需要检测是否存在人的遮挡，即人和未检测到的验钞机的ioa，不存在遮挡则超过15帧删除，否则保留
        to_remove = []
        for real_id, counter in self.money_counters.items():
            if idx_frame - counter.last_update_frame > 15:
                is_occluded = False
                for ren in ren_result:
                    *ren_xyxy, _, _, _ = ren
                    if ioa(ren_xyxy, counter.xyxy) > ioa_threshold:  # 假设IOA大于0.5认为被遮挡
                        is_occluded = True
                        break
                if not is_occluded:
                    to_remove.append(real_id)

        for real_id in to_remove:
            del self.money_counters[real_id]
            self.id_set.discard(real_id)

        # 清理track_id_map中对应的track_id，由于间隔问题，历史信息可能有用，所以不删除。
        # track_ids_to_remove = [track_id for track_id, real_id in self.track_id_map.items() if real_id in to_remove]
        # for track_id in track_ids_to_remove:
        #     del self.track_id_map[track_id]


class BoxTracker:
    def __init__(self):
        self.nums = 0                   # 历史追踪到的箱子数量
        self.track_id_map = dict()      # 将追踪ID映射成箱子的序号，字典：{track_id: real_id}
        self.boxes = dict()             # 箱子的实际ID，字典: {real_id: BankDetObject类}
        self.id_set = set()             # 当前检测到的ID，集合：{real_id}

    def update(self, idx_frame, kx_results, ren_result, ioa_threshold=0.5):
        for kx in kx_results:
            x1, y1, x2, y2, label, track_id, conf = kx
            xyxy = [x1, y1, x2, y2]

            if track_id in self.track_id_map and self.track_id_map[track_id] in self.id_set:
                real_id = self.track_id_map[track_id]
                old_label = self.boxes[real_id].label
                self.boxes[real_id].update(idx_frame, xyxy, label, conf)
                self.check_state_change(idx_frame, old_label, real_id)
            else:
                found = False
                for real_id, box in self.boxes.items():
                    if ioa(box.xyxy, xyxy) > ioa_threshold or ioa(xyxy, box.xyxy) > ioa_threshold:  # 打开和关闭两种情况
                        self.track_id_map[track_id] = real_id
                        box.update(idx_frame, xyxy, label, conf)
                        found = True

                        old_label = box.label
                        box.update(idx_frame, xyxy, label, conf)
                        self.check_state_change(idx_frame, old_label, real_id)
                        # print(f"标签变化：{old_label}, {box.label}")

                if not found:  # 新的箱子
                    if track_id in self.track_id_map:
                        real_id = self.track_id_map[track_id]
                    else:
                        self.nums += 1
                        real_id = self.nums
                    self.track_id_map[track_id] = real_id
                    self.boxes[real_id] = BankDetObject(idx_frame, track_id, xyxy, label, conf)
                    self.id_set.add(real_id)

        self.cleanup(idx_frame, ren_result, ioa_threshold)

        valid_boxes = []    # 返回所有有效的箱子
        valid_desc = []     # 箱子的描述
        for real_id, box in self.boxes.items():
            valid_boxes.append([*box.xyxy, box.label, real_id, box.confidence])
            valid_desc.append(box.desc[1])

        if len(valid_boxes) == 0:
            return np.empty((0, 7), dtype=np.int32), []

        return np.array(valid_boxes, dtype=np.int32), valid_desc

    def cleanup(self, idx_frame, ren_result, ioa_threshold=0.5):
        to_remove = []
        for real_id, box in self.boxes.items():
            if idx_frame - box.last_update_frame > 15:
                is_occluded = False
                for ren in ren_result:
                    ren_xyxy = ren[:4]
                    if ioa(ren_xyxy, box.xyxy) > ioa_threshold:  # 假设IOA大于0.5认为被遮挡
                        is_occluded = True
                        break
                if not is_occluded:
                    to_remove.append(real_id)

        for real_id in to_remove:
            del self.boxes[real_id]
            self.id_set.discard(real_id)

        # 清理track_id_map中对应的track_id，由于间隔问题，历史信息可能有用，所以不删除。
        # track_ids_to_remove = [track_id for track_id, real_id in self.track_id_map.items() if real_id in to_remove]
        # for track_id in track_ids_to_remove:
        #     del self.track_id_map[track_id]

    def check_state_change(self, idx_frame, old_label, real_id):
        if old_label - self.boxes[real_id].label < 0:  # 款箱的标签发生变化，说明进行了打开或者关闭
            self.boxes[real_id].add_desc([idx_frame, "打开款箱"])
            return
        if old_label - self.boxes[real_id].label > 0:
            self.boxes[real_id].add_desc([idx_frame, "关闭款箱"])
            return
        if self.boxes[real_id].desc[1] and idx_frame - self.boxes[real_id].desc[0] > 15:
            self.boxes[real_id].add_desc([idx_frame, None])
        return


if __name__ == '__main__':
    # 测试用例
    from ultralytics.task_bank.utils import apply_indices
    xyxy = torch.tensor([
        [100, 50, 200, 150],
        [100, 50, 200, 150],
        [50, 50, 70, 130],
        [30, 30, 100, 200],
        [10, 10, 15, 30]
    ])
    confidences = torch.tensor([0.8, 0.8, 0.3, 0.8, 0.7])

    valid_indices = filter_person_boxes(xyxy, confidences, height_ratio=2, width_ratio=2.5, area_ratio=4,
                                        confidence=0.5)
    selected_empty = apply_indices(xyxy, valid_indices)
    print("Selected Empty:", selected_empty)

    # 测试索引为空的情况
    empty_indices = torch.tensor([], dtype=torch.long)
    selected_empty = apply_indices(xyxy, empty_indices)
    print("Selected Empty:", selected_empty)
