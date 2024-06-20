import numpy as np
from collections import deque
from ultralytics.task_bank.compute import ioa_bbox_bbox_xyxy, ioa_bbox_candidates_xyxy


class MoneyState:
    """
    钱的验钞状态。
    """
    UnChecked = 1   # 未验钞
    InChecking = 2  # 验钞中
    Checked = 3     # 已验钞


class PositionState:
    """
    钱的位置状态。
    """
    UnKnown = 0     # 未知
    InBox = 1       # 在款箱中
    InDetector = 2  # 在验钞机中
    InDesk = 3      # 在桌上
    InHands = 4     # 在手中
    Occlusion = 5   # 被遮挡


class OperationState:
    """
    对钱的操作状态。
    """
    UnKnown = 0      # 未知
    LeaveBox = 1     # 离开款箱
    ReturnBox = 2    # 返回款箱
    ForgetCheck = 3  # 未验钞


class StateInfo:
    def __init__(self, state_type, state_keep_time, state_confirm_time, last_state):
        self.state_type = state_type
        self.state_keep_time = state_keep_time
        self.state_confirm_time = state_confirm_time
        self.last_state = last_state


class Money:
    def __init__(self, idx_frame, xyxy, track_id, conf, label, check_state=MoneyState.UnChecked,
                 pos_state=PositionState.UnKnown, state_confirm_time=5, min_conf=0.3, max_history=5, max_interval=15):
        # 追踪信息
        self.last_update_frame = idx_frame  # 最后更新帧，bytetrack_output: x1,y1,x2,y2,track_id,conf,label,order
        self.xyxy = xyxy                    # 检测框
        self.track_id = track_id            # 当前跟踪ID
        self.conf = conf                    # 置信度
        self.label = label                  # 标签
        # 状态共用信息
        self.state_confirm_time = state_confirm_time    # 确认一个状态所需的次数
        # 验钞状态信息
        self.check_state = check_state                  # 默认未进行验钞
        self.check_state_keep_time = 0                  # 上个验钞状态已经已经持续的时间
        self.last_check_state = self.check_state        # 上一个验钞状态
        # 位置状态信息
        self.pos_state = pos_state                      # 默认钱未知
        self.pos_state_keep_time = 0                    # 上个位置状态已经已经持续的时间
        self.last_pos_state = self.pos_state            # 上一个位置状态
        # 操作状态信息
        self.operation_state = OperationState.UnKnown   # 操作状态信息，由其余状态确定，所以不需要增加确认时间
        # 历史保存信息
        self.desc = ""  # 描述信息
        self.min_conf = min_conf                        # 可加入历史信息的置信度下限
        self.max_history = max_history                  # 最大保留的历史帧数
        self.history = deque(maxlen=max_history)        # 保存的历史信息
        self.history.append((idx_frame, xyxy, conf))    # 加入初始化信息
        self.max_interval = max_interval                # 最大间隔

    def update(self, idx_frame=None, xyxy=None, conf=None, check_state=None, pos_state=None):
        is_info_integrate = all(param is not None for param in [idx_frame, xyxy, conf])
        if is_info_integrate and conf > self.min_conf:    # 置信度高，历史信息作用小，当前结果加入历史信息
            self._add_to_history(idx_frame, xyxy, conf, check_state, pos_state)

    def _add_to_history(self, idx_frame, xyxy, conf, check_state, pos_state):
        self.last_update_frame = idx_frame  # 最后更新的帧
        while self.history and idx_frame - self.history[0][0] > self.max_interval:  # 删除超过15帧的历史信息
            self.history.popleft()

        if check_state is not None and check_state != self.check_state:     # 钱的验钞状态信息修改
            if check_state == self.last_check_state:
                self.check_state_keep_time += 1
                if self.check_state_keep_time >= self.state_confirm_time:
                    self.check_state = check_state
            else:
                self.last_check_state = check_state
                self.check_state_keep_time = 0

        if pos_state is not None and pos_state != self.pos_state:           # 钱的位置状态信息修改
            if pos_state == self.last_pos_state:
                self.pos_state_keep_time += 1
                if self.pos_state_keep_time >= self.state_confirm_time:
                    self.pos_state = pos_state
            else:
                self.last_pos_state = pos_state
                self.pos_state_keep_time = 0

        self.history.append((idx_frame, xyxy, conf))
        max_confidence_index = max(range(len(self.history)), key=lambda i: self.history[i][2])     # 返回最大置信度检测结果
        max_confidence_index = -1
        self.xyxy = self.history[max_confidence_index][1]
        self.conf = self.history[max_confidence_index][2]

    @property
    def result(self):
        state_array = np.concatenate([
            self.xyxy,
            np.array([self.track_id, self.conf, self.label, self.check_state, self.pos_state, self.operation_state],
                     dtype=np.float32)])
        return state_array

    def add_desc(self, desc):
        self.desc = desc

    def reset_track_id(self, new_track_id):
        self.track_id = new_track_id

    def set_operation_state(self, state):
        self.operation_state = state

    def set_check_state(self, state):
        self.check_state = state

    def get_check_state(self):
        return self.check_state

    def set_pos_state(self, state):
        self.pos_state = state

    def get_pos_state(self):
        return self.pos_state

    def get_operation_state(self):
        return self.operation_state


class MoneyTracker:
    def __init__(self):
        self.money = dict()             # 钱的实际ID，字典: {track_id: MoneyDetector类}
        self.id_now = set()             # 当前帧检测到的ID，集合: {track_id}
        self.id_history = set()         # 前面帧检测到的ID，集合: {track_id}

    def update(self, idx_frame, output_money, output_person, output_ycj, output_kx,
               ioa_in_kx=0.8, ioa_in_ycj=0.6, ioa_same_money=0.6):
        self.id_history.update(self.id_now)
        self.id_now.clear()

        have_update = [False] * output_money.shape[0]

        print(f"beginning history: {self.id_history}")
        self.check_in_ycj(idx_frame, output_money, output_ycj, have_update, ioa_in_ycj, ioa_same_money)
        print(f"after check money: money_num: {len(self.money)}, now: {self.id_now}, history: {self.id_history}")
        print(have_update)
        self.check_in_kx(idx_frame, output_money, output_kx, have_update, ioa_in_kx)
        print(f"after check box: money_num: {len(self.money)}, now: {self.id_now}, history: {self.id_history}")
        print(have_update)
        self.check_now_residual(idx_frame, output_money, have_update, ioa_same_money)
        print(f"after check res: money_num: {len(self.money)}, now: {self.id_now}, history: {self.id_history}")
        print(have_update)
        self.check_history(idx_frame, output_person, ioa_same_money)
        print(f"finally: money_num: {len(self.money)}, now: {self.id_now}, history: {self.id_history}")
        print(have_update)

        if len(self.id_now) == 0:
            return np.empty((0, 10), dtype=np.float32)
        else:
            output_money = []
            for id_n in self.id_now:
                output_money.append(self.money[id_n].result)

            return np.asarray(output_money, dtype=np.float32)

    def check_history(self, idx_frame, output_person, ioa_same_money):
        if output_person.shape[0] == 0:
            return

        remove_history_id = []
        for history_id in self.id_history:
            found_id = -1
            for now_id in self.id_now:
                ioa_n, ioa_h, iou = ioa_bbox_bbox_xyxy(self.money[now_id].xyxy, self.money[history_id].xyxy)

                if ioa_n > ioa_same_money or ioa_h > ioa_same_money:
                    found_id = now_id
                    break

            if found_id != -1:
                if (self.money[history_id].check_state == MoneyState.UnChecked or
                        self.money[found_id].check_state == MoneyState.UnChecked):
                    self.money[found_id].set_check_state(MoneyState.UnChecked)

                remove_history_id.append(history_id)

        self._remove_invalid_history_id(remove_history_id)

        for history_id in self.id_history:
            ioa_money, _, _ = ioa_bbox_candidates_xyxy(self.money[history_id].xyxy, output_person)
            if max(ioa_money) > ioa_same_money:     # 发生遮挡
                self.money[history_id].set_pos_state(PositionState.Occlusion)
            else:
                # 没遮挡且没检测到
                if (self.money[history_id].pos_state != PositionState.Occlusion and
                        idx_frame - self.money[history_id].last_update_frame > self.money[history_id].max_interval):
                    remove_history_id.append(history_id)

        self._remove_invalid_history_id(remove_history_id)

    def check_now_residual(self, idx_frame, output_money, have_update, ioa_same_money):
        track_id_set = {money[4] for money in output_money}  # 当前帧的追踪ID哈希表，方便反向检索

        for i, money in enumerate(output_money):
            if have_update[i]:
                continue

            x1, y1, x2, y2, track_id, conf, label, _ = money
            xyxy = [x1, y1, x2, y2]

            if track_id in self.id_history:  # 该ID已经存在
                have_update[i] = True
                self.money[track_id].update(idx_frame, xyxy, conf, None, PositionState.InDesk)  # 不在箱子也不在验钞机，设置在桌子上
                self.id_now.add(track_id)
                self.id_history.remove(track_id)
            else:
                found_id = -1
                for history_id in self.id_history:
                    pos_state = self.money[history_id].get_pos_state
                    if pos_state not in {PositionState.InDetector, PositionState.InBox}:
                        ioa_now, ioa_history, _ = ioa_bbox_bbox_xyxy(xyxy, self.money[history_id].xyxy)
                        if ioa_now > ioa_same_money or ioa_history > ioa_same_money:
                            found_id = history_id
                            break

                check_state = MoneyState.UnChecked
                pos_state = PositionState.UnKnown

                if found_id != -1:  # 找到了不同ID的同一个目标
                    check_state = self.money[found_id].get_check_state()
                    pos_state = self.money[found_id].get_pos_state()
                    self._remove_invalid_history_id([found_id])

                self.id_now.add(track_id)
                self.money[track_id] = Money(idx_frame, xyxy, track_id, conf, label, check_state, pos_state)

    def check_in_kx(self, idx_frame, output_money, output_kx, have_update, ioa_in_kx):
        if output_kx.shape[0] == 0:
            return

        for i, money in enumerate(output_money):
            if have_update[i]:     # 已经更新过则跳过
                continue

            x1, y1, x2, y2, track_id, conf, label, _ = money
            xyxy = [x1, y1, x2, y2]
            ioa_money, _, _ = ioa_bbox_candidates_xyxy(money, output_kx)    # 计算钱和款箱的ioa

            if max(ioa_money) > ioa_in_kx:  # 钱在款箱中
                have_update[i] = True

                if track_id in self.id_history:     # 该ID已经存在
                    check_state = self.money[track_id].get_check_state()
                    pos_state = self.money[track_id].get_check_state()

                    # 先前的状态不是在款箱中，且不是已经验钞状态
                    if pos_state != PositionState.InBox and check_state != MoneyState.Checked:
                        self.money[track_id].set_operation_state(OperationState.ForgetCheck)

                    self.money[track_id].update(idx_frame, xyxy, conf, check_state, PositionState.InBox)
                    self.id_now.add(track_id)
                    self.id_history.remove(track_id)
                else:
                    self.id_now.add(track_id)
                    self.money[track_id] = Money(idx_frame, xyxy, track_id, conf, label,
                                                 MoneyState.UnChecked, PositionState.InBox)

    def check_in_ycj(self, idx_frame, output_money, output_ycj, have_update, ioa_in_ycj, ioa_same_money):
        if output_ycj.shape[0] == 0:
            return

        ycj_dict_now = {i: [] for i in range(len(output_ycj))}      # 当前帧追踪结果在每个验钞机的情况
        ycj_dict_history = {i: [] for i in range(len(output_ycj))}  # 历史丢失帧在每个验钞机的情况（不含当前的track_id）
        track_id_set = {money[4] for money in output_money}         # 当前帧的追踪ID哈希表，方便反向检索

        for i, money in enumerate(output_money):    # 获取当前帧，每个验钞机中含有的钱
            if not have_update[i]:
                ioa_money, _, _ = ioa_bbox_candidates_xyxy(money, output_ycj)
                ioa_max = np.max(ioa_money)

                if ioa_max > ioa_in_ycj:            # 钱在验钞机中，则加入对应的验钞机列表中
                    ycj_dict_now[np.argmax(ioa_money)].append(i)

        remove_history_id = []  # 需要移除的历史ID
        for history_id in self.id_history:      # 获取当前帧，每个验钞机历史含有的钱
            # 对于历史信息中，不在本次追踪ID中，且在验钞机中的钱操作
            if history_id not in track_id_set and self.money[history_id].get_check_state() != MoneyState.UnChecked:
                ioa_money, _, _ = ioa_bbox_candidates_xyxy(self.money[history_id].xyxy, output_ycj)
                ioa_max = np.max(ioa_money)

                if ioa_max > ioa_in_ycj:
                    # TODO:此处判断需要加上遮挡判断？长时间遮挡后，ID切换了、更新时间超时了，但是还是原来的钱。不过在验钞机里还好。
                    # 先验知识：放入的钱会消失，验过的钱（新生成的目标，只有手去拿了才会消失）
                    if (self.money[history_id].check_state != MoneyState.Checked and
                            idx_frame - self.money[history_id].last_update_frame > self.money[history_id].max_interval):
                        remove_history_id.append(history_id)
                    else:
                        ycj_dict_history[np.argmax(ioa_money)].append(history_id)   # 一个验钞机上漏检的（放入or路过）的钱

        self._remove_invalid_history_id(remove_history_id)

        for i, (ycj, money_list) in enumerate(ycj_dict_now.items()):
            if len(money_list) == 0:
                continue
            elif len(money_list) == 1:  # 当前只检测到一个
                have_update[money_list[0]] = True

                x1, y1, x2, y2, track_id, conf, label, _ = output_money[money_list[0]]
                xyxy = [x1, y1, x2, y2]

                check_state = MoneyState.InChecking
                pos_state = PositionState.InDetector

                if len(ycj_dict_history[i]) > 0:
                    for history_id in ycj_dict_history[i]:
                        ioa_now, ioa_history, _ = ioa_bbox_bbox_xyxy(xyxy, self.money[history_id].xyxy)

                        if ioa_now > ioa_same_money or ioa_history > ioa_same_money:    # 是同一捆钱，只需要认定已验钞的部分
                            remove_history_id.append(history_id)    # 删除重复
                            check_state = MoneyState.Checked if (
                                    self.money[history_id].check_state == MoneyState.Checked) else MoneyState.InChecking

                    self._remove_invalid_history_id(remove_history_id)

                if track_id in self.id_history:
                    if check_state != MoneyState.Checked and self.money[track_id].check_state == MoneyState.Checked:
                        check_state = MoneyState.Checked

                    self.money[track_id].update(idx_frame, xyxy, conf, check_state, pos_state)
                    self.id_now.add(track_id)
                    self.id_history.remove(track_id)
                else:
                    self.money[track_id] = Money(idx_frame, xyxy, track_id, conf, label, check_state, pos_state)
                    self.id_now.add(track_id)

            else:
                for money_index in money_list:
                    # TODO: 这里简化只检测一次。实际需要遍历两次，先匹配存在的ID；再匹配不存在的，匹配不上再分配新的对象
                    x1, y1, x2, y2, track_id, conf, label, _ = output_money[money_index]
                    xyxy = [x1, y1, x2, y2]

                    if track_id in self.id_history:  # 存在：完整：刚放入验钞机，验钞完，路过验钞机；漏检、遮挡：另一个没检测
                        have_update[money_index] = True
                        check_state = MoneyState.InChecking if self.money[track_id].check_state != MoneyState.Checked \
                            else MoneyState.Checked
                        pos_state = PositionState.InDetector
                        self.money[track_id].update(idx_frame, xyxy, conf, check_state, pos_state)

                        self.id_now.add(track_id)
                        self.id_history.remove(track_id)

                for ycj, money_list in ycj_dict_now.items():
                    # 第二次匹配没有匹配上的
                    for money_index in money_list:
                        if have_update[money_index]:
                            continue
                        else:  # 不存在：漏检验钞入口、入口形状变化、遮挡产生新ID
                            have_update[money_index] = True
                            x1, y1, x2, y2, track_id, conf, label, _ = output_money[money_index]
                            xyxy = [x1, y1, x2, y2]

                            found_id = -1   # 找到的ID（同一个目标被分配了不同ID）
                            for history_id in self.id_history:
                                if self.money[history_id].get_pos_state == PositionState.InDetector:
                                    ioa_now, ioa_history, _ = ioa_bbox_bbox_xyxy(xyxy, self.money[history_id].xyxy)
                                    if ioa_now > ioa_same_money or ioa_history > ioa_same_money:
                                        found_id = history_id
                                        break

                            if found_id != -1:  # 找到了不同ID的同一个目标
                                self.money[found_id].reset_track_id(track_id)       # 原钱对象的追踪ID修改
                                self.money[track_id] = self.money.pop(found_id)     # 追踪对象的追踪ID修改
                                check_state = MoneyState.InChecking \
                                    if self.money[track_id].get_check_state() != MoneyState.Checked else MoneyState.Checked
                                pos_state = PositionState.InDetector
                                self.money[track_id].update(idx_frame, xyxy, conf, check_state, pos_state)
                                self.id_now.add(track_id)
                                self.id_history.remove(found_id)
                            else:   # 没有找到
                                self.id_now.add(track_id)
                                self.money[track_id] = Money(idx_frame, xyxy, track_id, conf, label,
                                                             MoneyState.Checked, PositionState.InDetector)

    def _remove_invalid_history_id(self, remove_history_id: list):
        while remove_history_id:
            hid = remove_history_id.pop()
            self.id_history.remove(hid)
            del self.money[hid]
