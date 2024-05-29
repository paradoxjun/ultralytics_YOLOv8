import time
import cv2
import os

from datetime import datetime
from ultralytics.task_bank.predict import BankDetectionPredictor
from ultralytics.trackers.tracker_deep_sort import VideoTracker
from ultralytics.task_bank.utils import resize_and_pad
from ultralytics.utils.metrics import box_iou
from collections import deque


class GetPos:
    def __init__(self):
        self.confs = 0
        self.xyxy = (-1, -1, -1, -1)
        # self.xyv = (0, 0)
        # self.t = 0
        # self.status_q = deque()
        # self.valid_status_num = 0

    def get_status(self, xyxy, confs):
        if confs > self.confs:
            self.confs = confs
            self.xyxy = xyxy
        #     self.t = time.time()
        #     self.status_q.append(self.xyxy)
        #     self.valid_status_num += 1


class MoneyStateMachine:
    def __init__(self):
        self.state = "IN_BOX"

    def update(self, detection):
        if self.state == "IN_BOX":
            if detection['location'] == 'out_of_box':
                self.state = "OUT_BOX"
        elif self.state == "OUT_BOX":
            if detection['location'] == 'counting_machine':
                self.state = "COUNTING"
            elif detection['location'] == 'in_box':
                self.state = "RETURN_BOX"
        elif self.state == "COUNTING":
            if detection['location'] == 'out_of_counting_machine':
                self.state = "RETURN_BOX"

    def has_passed_counting(self):
        return self.state == "RETURN_BOX"


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

vt = VideoTracker(track_cfg=track_cfg, predictors=predictors)
cap = vt.get_video()

if not cap.isOpened():
    print("INFO: 无法获取视频，退出！")
    exit()

# 获取视频的宽度、高度和帧率
if vt.track_cfg["save_option"]["save"]:
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

    # frame = apply_gaussian_blur(frame)

    if idx_frame % vt.track_cfg["vid_stride"] == 0:
        deep_sort, det_res, cost_time = vt.image_track(frame)       # 追踪结果，检测结果，消耗时间
        last_deepsort = deep_sort
        yolo_time.append(cost_time[0])          # yolo推理时间
        sort_time.append(cost_time[1])          # deepsort跟踪时间

        if vt.track_cfg["verbose"]:
            print('INFO: Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, *cost_time))

        plot_img = vt.plot_track(frame, deep_sort)                  # 绘制加入追踪框的图片
        vt.save_track(idx_frame, plot_img, deep_sort, det_res)      # 保存跟踪结果
    else:
        plot_img = vt.plot_track(frame, last_deepsort)              # 帧间隔小，物体运动幅度小，就用上一次结果

    if vt.track_cfg["save_option"]["save"]:
        out.write(plot_img)         # 将处理后的帧写入输出视频

    t1 = time.time()
    avg_fps.append(t1 - t0)     # 第1帧包含了模型加载时间要删除

    # add FPS information on output video
    text_scale = max(1, plot_img.shape[1] // 1000)
    cv2.putText(plot_img, 'frame: %d fps: %.2f ' % (idx_frame, (len(avg_fps) - 1) / (sum(avg_fps[1:]) + 1e-6)),
                (10, 20 + text_scale), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=1)
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

