import time
import cv2
import os

import torch
import numpy as np
from datetime import datetime
from ultralytics.utils.torch_utils import time_sync
from ultralytics.utils.plotting import colors as set_color
from ultralytics.task_bank.predict import BankDetectionPredictor
from ultralytics.trackers.tracker_deep_sort import VideoTracker
from ultralytics.task_bank.utils import resize_and_pad
from ultralytics.utils.metrics import box_iou
from collections import deque


class GetPos:
    def __init__(self):
        self.confs = -1
        self.xyxy = (-1, -1, -1, -1)
        self.xywh = (-1, -1, -1, -1)
        # self.xyv = (0, 0)
        # self.t = 0
        # self.status_q = deque()
        # self.valid_status_num = 0

    def get_status(self, xyxy, xywh, confs):
        if confs > self.confs:
            self.confs = confs
            self.xyxy = xyxy
            self.xywh = xywh
        #     self.t = time.time()
        #     self.status_q.append(self.xyxy)
        #     self.valid_status_num += 1


class MoneyStateMachine:
    def __init__(self):
        self.state = "IN_BOX"

    def update(self, current_detections):
        money_coords = current_detections['money']
        person_coords = current_detections['person']
        box_coords = current_detections['box']
        counting_machine_coords = current_detections['counting_machine']

        # 更新动作
        action = self.detect_action(money_coords, person_coords, box_coords, counting_machine_coords)

        # 更新状态
        if action == "UNKNOWN":
            pass
        else:
            self.state = action
        return self.state

    def detect_action(self, money_coords, person_coords, box_coords, counting_machine_coords):
        if money_coords is None:
            return "UNKNOWN"

        if self.is_in_person(money_coords, person_coords):
            return "IN_PERSON"
        elif self.is_in_box(money_coords, box_coords):
            return "IN_BOX"
        elif self.is_in_counting_machine(money_coords, counting_machine_coords):
            return "IN_COUNTING_MACHINE"
        else:
            return "UNKNOWN"

    def is_in_person(self, money_coords, person_coords):
        return self.is_in_region(money_coords, person_coords)

    def is_in_box(self, money_coords, box_coords):
        return self.is_in_region(money_coords, box_coords)

    def is_in_counting_machine(self, money_coords, counting_machine_coords):
        return self.is_in_region(money_coords, counting_machine_coords)

    def is_on_table(self, money_coords, table_coords):
        return self.is_in_region(money_coords, table_coords)

    def is_near(self, coords1, coords2, threshold=20):
        x1, y1, x2, y2 = coords1
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        bx1, by1, bx2, by2 = coords2
        cx2, cy2 = (bx1 + bx2) / 2, (by1 + by2) / 2
        distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        return distance < threshold

    def is_in_region(self, money_coords, region_coords):
        x_center = (money_coords[0] + money_coords[2]) / 2
        y_center = (money_coords[1] + money_coords[3]) / 2
        rx1, ry1, rx2, ry2 = region_coords
        return rx1 <= x_center <= rx2 and ry1 <= y_center <= ry2


money_state = MoneyStateMachine()
current_detections = {
    'counting_machine': None,
    'box': None,
    'money': None,
    'person': None,
}


def plot_track_fix(self, img, deepsort_output, ycj: GetPos, kx: GetPos):      # 在一帧上绘制检测结果（类别+置信度+追踪ID）
    now_state = 'UNKNOWN'
    for i, box in enumerate(deepsort_output):
        x1, y1, x2, y2, label, track_id, confidence = list(map(int, box))       # 将结果均映射为整型
        if label == 0:
            if confidence > ycj.confs:
                ycj.confs = confidence
                ycj.xyxy = x1, y1, x2, y2
                current_detections['counting_machine'] = [x1, y1, x2, y2]
            continue
        elif label == 1:
            if confidence > kx.confs:
                kx.confs = confidence
                kx.xyxy = x1, y1, x2, y2
                current_detections['box'] = [x1, y1, x2, y2]
            continue
        elif label == 3:
            current_detections['money'] = [x1, y1, x2, y2]
        elif label == 4:
            current_detections['person'] = [x1, y1, x2, y2]

        money_state.update(current_detections)
        now_state = money_state.state

        # 设置显示内容：文本框左上角为“标签名：置信度”，右上角为“跟踪id”，文本框颜色由类别决定
        color = set_color(label * 4)    # 设置颜色
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)    # 基本矩形检测框
        label_text = f'{self.track_cfg["class_name"][label]}:{round(confidence / 100, 2)}'  # 左上角标签+置信度文字
        cv2.putText(img, label_text, (x1 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        track_text = f"ID: {track_id}"  # 右上角追踪ID文字
        cv2.putText(img, track_text, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if ycj.confs > 0:
            label = 0
            confidence = ycj.confs
            x1, y1, x2, y2 = ycj.xyxy
            color = set_color(label * 4)  # 设置颜色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # 基本矩形检测框
            label_text = f'{self.track_cfg["class_name"][label]}:{round(confidence / 100, 2)}'  # 左上角标签+置信度文字
            cv2.putText(img, label_text, (x1 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            track_text = f"ID: ycj"  # 右上角追踪ID文字
            cv2.putText(img, track_text, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if kx.confs > 0:
            label = 1
            confidence = kx.confs
            x1, y1, x2, y2 = kx.xyxy
            color = set_color(label * 4)  # 设置颜色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # 基本矩形检测框
            label_text = f'{self.track_cfg["class_name"][label]}:{round(confidence / 100, 2)}'  # 左上角标签+置信度文字
            cv2.putText(img, label_text, (x1 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            track_text = f"ID: kx"  # 右上角追踪ID文字
            cv2.putText(img, track_text, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, now_state


track_cfg = '/home/chenjun/code/ultralytics_YOLOv8/ultralytics/cfg/bank_monitor/track.yaml'
overrides_1 = {"task": "detect",
               "mode": "predict",
               "model": '/home/chenjun/code/ultralytics_YOLOv8/weights/yolov8s.pt',
               "verbose": False,
               "classes": [0]
               }

overrides_2 = {"task": "detect",
               "mode": "predict",
               "model": '/home/chenjun/code/ultralytics_YOLOv8/runs/detect/train_bank_05_21_m/weights/best.pt',
               "verbose": False,
               }

predictor_1 = BankDetectionPredictor(overrides=overrides_1)
predictor_2 = BankDetectionPredictor(overrides=overrides_2)
predictors = [predictor_1, predictor_2]

ycj = GetPos()
kx = GetPos()
ren = GetPos()

vt = VideoTracker(track_cfg=track_cfg, predictors=predictors)
cap = vt.get_video()

if not cap.isOpened():
    print("INFO: 无法获取视频，退出！")
    exit()

# 获取视频的宽度、高度和帧率
if vt.track_cfg["save_option"]["save"]:
    if vt.track_cfg["video_shape"][0] > 32 and vt.track_cfg["video_shape"][1] > 32:
        width = vt.track_cfg["video_shape"][0]
        height = vt.track_cfg["video_shape"][1]
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    video_plot_save_path = os.path.join(vt.save_dir, "video_plot_" + current_time + ".mp4")
    out = cv2.VideoWriter(video_plot_save_path, fourcc, fps, (width, height))  # 初始化视频写入器

yolo_time, sort_time, avg_fps = [], [], []
t_start = time.time()

idx_frame = 0
last_deepsort = None  # 跳过的帧不绘制，会导致检测框闪烁


while True:
    ret, frame = cap.read()
    t0 = time.time()

    if not ret or cv2.waitKey(1) & 0xFF == ord('q'):  # 结束 或 按 'q' 键退出
        break

    if vt.track_cfg["video_shape"][0] > 32 and vt.track_cfg["video_shape"][1] > 32:
        frame = resize_and_pad(frame, vt.track_cfg["video_shape"])
        plot_img = frame

    # frame = apply_gaussian_blur(frame)

    if idx_frame % vt.track_cfg["vid_stride"] == 0:
        deep_sort, det_res, cost_time = vt.image_track(frame)       # 追踪结果，检测结果，消耗时间
        last_deepsort = deep_sort
        yolo_time.append(cost_time[0])          # yolo推理时间
        sort_time.append(cost_time[1])          # deepsort跟踪时间

        if vt.track_cfg["verbose"]:
            print('INFO: Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, *cost_time))

        plot_img, now_state = plot_track_fix(vt, frame, deep_sort, ycj, kx)                  # 绘制加入追踪框的图片
        vt.save_track(idx_frame, plot_img, deep_sort, det_res)      # 保存跟踪结果
    else:
        plot_img, now_state = plot_track_fix(vt, frame, last_deepsort, ycj, kx)              # 帧间隔小，物体运动幅度小，就用上一次结果

    t1 = time.time()
    avg_fps.append(t1 - t0)     # 第1帧包含了模型加载时间要删除

    # add FPS information on output video
    text_scale = max(1, plot_img.shape[1] // 1000)
    cv2.putText(plot_img, 'frame: %d fps: %.2f ' % (idx_frame, (len(avg_fps) - 1) / (sum(avg_fps[1:]) + 1e-6)),
                (10, 20 + text_scale), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=1)
    cv2.putText(plot_img, "Now State: " + now_state,
                (260, 20 + text_scale), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 255), thickness=2)

    if vt.track_cfg["save_option"]["save"]:
        out.write(plot_img)         # 将处理后的帧写入输出视频

    cv2.imshow('Frame', plot_img)
    idx_frame += 1

cap.release()   # 释放读取资源
if vt.track_cfg["save_option"]["save"]:
    out.release()   # 释放写入资源
cv2.destroyAllWindows()

avg_yolo_t, avg_sort_t = sum(yolo_time[1:]) / (len(yolo_time) - 1), sum(sort_time[1:]) / (len(sort_time) - 1)
print(f'INFO: Avg YOLO time ({avg_yolo_t:.3f}s), Sort time ({avg_sort_t:.3f}s) per frame')
total_t, avg_fps = time.time() - t_start, (len(avg_fps) - 1) / (sum(avg_fps[1:]) + 1e-6)
print('INFO: Total Frame: %d, Total time (%.3fs), Avg fps (%.3f)' % (idx_frame, total_t, avg_fps))

