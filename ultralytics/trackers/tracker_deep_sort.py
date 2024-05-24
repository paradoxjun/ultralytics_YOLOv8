"""
代码参考DeepSORT_YOLOv5_Pytorch
"""
from ultralytics.utils.torch_utils import time_sync
from ultralytics.utils import yaml_load
from ultralytics.utils.plotting import colors as set_color
from ultralytics.trackers.deep_sort import build_tracker
from ultralytics.task_bank.predict import BankDetectionPredictor
from ultralytics.task_bank.utils import get_config

import os
import sys
import time
import cv2
import torch
import torch.backends.cudnn as cudnn

currentUrl = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(currentUrl)))

cudnn.benchmark = True


class VideoTracker:
    def __init__(self, track_cfg, predictors):
        self.track_cfg = yaml_load(track_cfg)       # v8内置方法读取track.yaml文件为字典
        self.deepsort_arg = get_config(self.track_cfg["config_deep_sort"])  # 读取deep_sort.yaml为EasyDict类
        self.predictors = predictors                # 检测器列表
        use_cuda = self.track_cfg["device"] != "cpu" and torch.cuda.is_available()
        self.deepsort = build_tracker(self.deepsort_arg, use_cuda=use_cuda)

        print("Tracker init finished...")

    def get_video(self, video_path=None):                   # 获取视频流
        if video_path is None:                              # 读取输入
            if self.track_cfg["camera"] != -1:              # 使用摄像头获取视频
                print("Using webcam " + str(self.track_cfg["camera"]))
                v_cap = cv2.VideoCapture(self.track_cfg["camera"])
            else:
                assert os.path.isfile(self.track_cfg["input_path"]), "Video path in *.yaml is error. "
                v_cap = cv2.VideoCapture(self.track_cfg["input_path"])
        else:
            assert os.path.isfile(video_path), "Video path in method get_video() is error. "
            v_cap = cv2.VideoCapture(video_path)

        return v_cap

    def resize_video(self, v_cap, shape=None):              # 重新调整帧的尺寸
        if shape is None:
            width, height = self.track_cfg["video_shape"]
        else:
            width, height = shape
        if width >= 32 and height >= 32:
            v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)      # 设置帧宽度
            v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)    # 设置帧高度

            if not (v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) and v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)):
                print(f"Fail to set video size, can not use cv2.VideoCapture().set() to set. ")
                return False
            print(f"Success to set video size, width: {width}, height: {height}. ")
            return True

        print(f"Fail to set video size, width: {width} or height: {height} is illegal, which need to larger than 32. ")
        return False

    def image_track(self, img):     # 生成追踪目标的id
        t1 = time_sync()
        det_person = self.predictors[0](source=img)[0]     # 官方预训练权重，检测人的位置
        det_things = self.predictors[1](source=img)[0]     # 自己训练的权重，检测物的位置
        t2 = time_sync()

        bbox_xywh = torch.cat((det_person.boxes.xywh, det_things.boxes.xywh)).cpu()     # 目标框
        confs = torch.cat((det_person.boxes.conf, det_things.boxes.conf)).cpu()         # 置信度
        cls = torch.cat((det_person.boxes.cls + 4, det_things.boxes.cls)).cpu()         # 标签，多检测器需要调整类别标签

        if len(cls) > 0:
            deepsort_outputs = self.deepsort.update(bbox_xywh, confs, img, cls)   # x1,y1,x2,y2,label,track_ID,confs
            # print(f"bbox_xywh: {bbox_xywh}, confs: {confs}, cls: {cls}, outputs: {outputs}")
        else:
            deepsort_outputs = torch.zeros((0, 6))

        t3 = time.time()
        return deepsort_outputs, [bbox_xywh, cls, confs], [t2 - t1, t3 - t2]

    def plot_track(self, img, offset=(0, 0)):
        deepsort_output, _, _ = self.image_track(img)

        for i, box in enumerate(deepsort_output):
            print(box)
            x1, y1, x2, y2, label, track_id, confidence = list(map(int, box))
            x1, y1, x2, y2 = x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1]

            # 设置显示内容：文本框左上角为“标签名：置信度”，右上角为“跟踪id”，文本框颜色由类别决定
            color = set_color(label * 4)    # 设置颜色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)    # 基本矩形检测框
            label_text = f'{self.track_cfg["class_name"][label]}:{round(confidence / 100, 2)}'      # 左上角标签文字
            cv2.putText(img, label_text, (x1 - 30, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            track_text = f"ID: {track_id}"
            cv2.putText(img, track_text, (x2 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    def save_track(self):
        pass


if __name__ == '__main__':
    track_cfg = '/home/chenjun/code/ultralytics_YOLOv8/ultralytics/cfg/bank_monitor/track.yaml'
    overrides_1 = {"task": "detect",
                   "mode": "predict",
                   "model": '/home/chenjun/code/ultralytics_YOLOv8/weights/yolov8m.pt',
                   "verbose": False,
                   "classes": 0}

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
        print("无法打开摄像头")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 显示视频帧
        plot_img = vt.plot_track(frame)

        cv2.imshow('Frame', plot_img)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
