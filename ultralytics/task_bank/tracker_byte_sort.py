from ultralytics.utils.torch_utils import time_sync
from ultralytics.utils import yaml_load
from ultralytics.utils.plotting import colors as set_color
from ultralytics.task_bank.byte_tracker_modify import BYTETracker
from ultralytics.task_bank.predict import BankDetectionPredictor
from ultralytics.task_bank.utils import get_config, resize_and_pad, get_bytetrack_input_by_label, draw_dashed_rectangle
from ultralytics.task_bank.process_ren import FilterPerson
from ultralytics.task_bank.process_ycj import MoneyDetectorTracker
from ultralytics.task_bank.process_kx import KXTracker
from ultralytics.task_bank.process_money import MoneyTracker
from pathlib import Path
from datetime import datetime

import os
import time
import cv2
import numpy as np
import torch


class VideoTracker:
    def __init__(self, track_cfg, predictors):
        self.track_cfg = yaml_load(track_cfg)       # v8内置方法读取track.yaml文件为字典
        byte_sort_config = get_config(self.track_cfg["config_byte_sort"])      # 读取byte_sort.yaml为EasyDict类
        self.predictors = predictors                # 检测器列表
        self.save_dir = self.make_save_dir()
        self.filter_person = FilterPerson()
        self.bytesort_ycj = BYTETracker(byte_sort_config)
        self.track_ycj = MoneyDetectorTracker()
        self.bytesort_kx = BYTETracker(byte_sort_config)
        self.track_kx = KXTracker()
        self.bytesort_money = BYTETracker(byte_sort_config)
        self.track_money = MoneyTracker()

        print("INFO: Tracker init finished...")

    def make_save_dir(self):    # 创建保存文件的文件夹
        root_dir = Path(self.track_cfg["save_option"]["root"])      # 保存根路径

        if not root_dir.exists():   # 根路径一定要自己指定
            raise ValueError(f"设置存储根目录失败，不存在根路径：{root_dir}")

        save_dir = os.path.join(root_dir, self.track_cfg["save_option"]["dir"])     # 实际保存路径

        if os.path.exists(save_dir):    # 存在也保存到这里
            print(f"INFO: 当前保存路径 [{save_dir}] 已经存在。")
        else:
            os.makedirs(save_dir)
            print(f"INFO: 当前保存路径 [{save_dir}] 不存在，已创建。")

        for sub_dir in ["image_plot", "txt_track", "txt_xyxy", "txt_xywh"]:     # 分目录保存不同结果
            sub = os.path.join(save_dir, sub_dir)
            if not os.path.exists(sub):
                os.makedirs(sub)

        return save_dir

    def get_video(self, video_path=None):           # 获取视频流（优先级：摄像头 > 指定文件路径 > 配置文件路径）
        if video_path is None:                      # 读取输入
            if self.track_cfg["camera"] != -1:      # 使用摄像头获取视频
                print("INFO: Using webcam " + str(self.track_cfg["camera"]))
                v_cap = cv2.VideoCapture(self.track_cfg["camera"])
            else:                                           # 使用文件路径获取
                assert os.path.isfile(self.track_cfg["input_path"]), "Video path in *.yaml is error. "
                v_cap = cv2.VideoCapture(self.track_cfg["input_path"])
        else:
            assert os.path.isfile(video_path), "Video path in method get_video() is error. "
            v_cap = cv2.VideoCapture(video_path)

        return v_cap

    def image_track(self, img, idx_frame):     # 生成追踪目标的id
        t1 = time_sync()
        det_person = self.predictors[0](source=img)[0].cpu()     # 官方预训练权重，检测人的位置
        det_things = self.predictors[1](source=img)[0].cpu()     # 自己训练的权重，检测物的位置
        t2 = time_sync()

        # outputs_person = torch.cat([det_person.boxes.xyxy, det_person.boxes.conf.unsqueeze(1)], dim=1).numpy()
        ycj = get_bytetrack_input_by_label(det_things, label_need=(0,))
        kx = get_bytetrack_input_by_label(det_things, label_need=(1, 2))
        money = get_bytetrack_input_by_label(det_things, label_need=(3,))

        outputs_person, _ = self.filter_person(det_person)

        outputs_ycj = self.bytesort_ycj.update(*ycj)  # x1,y1,x2,y2,track_id,confs,label,order
        outputs_ycj = self.track_ycj.update(idx_frame, outputs_ycj, outputs_person)
        outputs_kx = self.bytesort_kx.update(*kx)
        outputs_kx = self.track_kx.update(idx_frame, outputs_kx, outputs_person)
        outputs_money = self.bytesort_money.update(*money)
        print("*" * 100)
        print(f"Inputs_money: {outputs_money}")
        outputs_money = self.track_money.update(idx_frame, outputs_money, outputs_person, outputs_ycj[0], outputs_kx[0])
        print(f"Outputs_money: {outputs_money}")

        t3 = time.time()
        return [outputs_person, outputs_ycj, outputs_kx, outputs_money],  [t2 - t1, t3 - t2]

    def plot_track(self, img, track_outputs):      # 在一帧上绘制检测结果（类别+置信度+追踪ID）
        outputs_person, outputs_ycj, outputs_kx, outputs_money = track_outputs
        for i, box in enumerate(outputs_person):
            x1, y1, x2, y2 = list(map(int, box[:4]))
            conf = float(box[4])
            color = set_color(24)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # 基本矩形检测框
            label_text = f'person:{round(conf, 2)}'  # 左上角标签+置信度文字
            cv2.putText(img, label_text, (x1 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for i, box in enumerate(outputs_ycj[0]):
            x1, y1, x2, y2, track_id, _, label, real_id = list(map(int, box))       # 将结果均映射为整型
            confidence = float(box[5])

            # 设置显示内容：文本框左上角为“标签名：置信度”，右上角为“跟踪id”，文本框颜色由类别决定
            color = set_color(label * 4)    # 设置颜色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)    # 基本矩形检测框
            label_text = f'{self.track_cfg["class_name"][label]}:{round(confidence, 2)}'  # 左上角标签+置信度文字
            cv2.putText(img, label_text, (x1 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            track_text = f"ID: {real_id}"  # 右上角追踪ID文字
            cv2.putText(img, track_text, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for i, box in enumerate(outputs_ycj[1]):
            x1, y1, x2, y2, track_id, _, label, real_id = list(map(int, box))  # 将结果均映射为整型
            confidence = float(box[5])

            # 设置显示内容：文本框左上角为“标签名：置信度”，右上角为“跟踪id”，文本框颜色由类别决定
            color = set_color(label * 4)  # 设置颜色
            draw_dashed_rectangle(img, x1, y1, x2, y2, color, 2)  # 虚线矩形检测框
            label_text = f'{self.track_cfg["class_name"][label]}:{round(confidence, 2)}'  # 左上角标签+置信度文字
            cv2.putText(img, label_text, (x1 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            track_text = f"ID: {real_id}"  # 右上角追踪ID文字
            cv2.putText(img, track_text, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for i, box in enumerate(outputs_kx[0]):
            x1, y1, x2, y2, track_id, _, label, real_id = list(map(int, box))       # 将结果均映射为整型
            confidence = float(box[5])

            # 设置显示内容：文本框左上角为“标签名：置信度”，右上角为“跟踪id”，文本框颜色由类别决定
            color = set_color(2 * 4)    # 设置颜色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)    # 基本矩形检测框
            label_text = f'{self.track_cfg["class_name"][label]}:{round(confidence, 2)}'  # 左上角标签+置信度文字
            cv2.putText(img, label_text, (x1 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            track_text = f"ID: {real_id}"  # 右上角追踪ID文字
            cv2.putText(img, track_text, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for i, box in enumerate(outputs_kx[1]):
            x1, y1, x2, y2, track_id, _, label, real_id = list(map(int, box))  # 将结果均映射为整型
            confidence = float(box[5])

            # 设置显示内容：文本框左上角为“标签名：置信度”，右上角为“跟踪id”，文本框颜色由类别决定
            color = set_color(2 * 4)  # 设置颜色
            draw_dashed_rectangle(img, x1, y1, x2, y2, color, 2)  # 虚线矩形检测框
            label_text = f'{self.track_cfg["class_name"][label]}:{round(confidence, 2)}'  # 左上角标签+置信度文字
            cv2.putText(img, label_text, (x1 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            track_text = f"ID: {real_id}"  # 右上角追踪ID文字
            cv2.putText(img, track_text, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for i, box in enumerate(outputs_money):
            x1, y1, x2, y2, track_id, conf, label, check_state, pos_state, operation_state = list(map(int, box))       # 将结果均映射为整型
            confidence = float(box[5])

            if int(check_state) == 1:
                color = (0, 0, 255)
            elif int(check_state) == 2:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            # 设置显示内容：文本框左上角为“标签名：置信度”，右上角为“跟踪id”，文本框颜色由类别决定
            # color = set_color(check_state)    # 设置颜色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)    # 基本矩形检测框
            label_text = f'ch:{int(check_state)} po:{int(pos_state)}'  # 左上角标签+置信度文字
            cv2.putText(img, label_text, (x1 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            track_text = f"ID: {track_id}"  # 右上角追踪ID文字
            cv2.putText(img, track_text, (x2 + 20, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img

    def save_track(self, i=0, img=None, track_output=None, det_res=None):    # 传入帧数，绘制结果，追踪结果，检测结果
        if not self.track_cfg["save_option"]["save"]:
            return

        if img is not None and self.track_cfg["save_option"]["img"]:
            img_save = os.path.join(self.save_dir, "image_plot", "img_" + str(i).zfill(5) + ".jpg")
            cv2.imwrite(img_save, img)

            if self.track_cfg["verbose"]:
                print(f"INFO: 已经保存[{img_save}].")

        if track_output is not None and self.track_cfg["save_option"]["txt"]:
            bytesort_save = os.path.join(self.save_dir, "txt_track", "bytesort_" + str(i).zfill(5) + ".txt")
            np.savetxt(bytesort_save, track_output, fmt='%d')

            if self.track_cfg["verbose"]:
                print(f"INFO: 已经保存[{bytesort_save}].")

        if det_res is not None and self.track_cfg["save_option"]["txt"]:
            xywh, xyxy, cls, confs = det_res    # torch.Size([n, 4]) torch.Size([n, 4]) torch.Size([n]) torch.Size([n])
            xywh_save = os.path.join(self.save_dir, "txt_xywh", "xywh_" + str(i).zfill(5) + ".txt")
            xyxy_save = os.path.join(self.save_dir, "txt_xyxy", "xyxy_" + str(i).zfill(5) + ".txt")
            xywh_np = torch.cat([xywh, cls.view(-1, 1), confs.view(-1, 1)], dim=1).numpy()
            xyxy_np = torch.cat([xyxy, cls.view(-1, 1), confs.view(-1, 1)], dim=1).numpy()
            np.savetxt(xywh_save, xywh_np, fmt='%.6f')

            if self.track_cfg["verbose"]:
                print(f"INFO: 已经保存[{xywh_save}].")
            np.savetxt(xyxy_save, xyxy_np, fmt='%.6f')
            if self.track_cfg["verbose"]:
                print(f"INFO: 已经保存[{xyxy_save}].")

    def det_track_pipline(self, video_path=None):    # 读取视频，检测，追踪，绘制，保存全流程
        cap = self.get_video(video_path=video_path)
        if not cap.isOpened():
            print("INFO: 无法获取视频，退出！")
            exit()

        # 获取视频的宽度、高度和帧率
        if self.track_cfg["save_option"]["save"]:
            if self.track_cfg["video_shape"][0] > 32 and self.track_cfg["video_shape"][1] > 32:
                width = self.track_cfg["video_shape"][0]
                height = self.track_cfg["video_shape"][1]
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
            current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
            video_plot_save_path = os.path.join(self.save_dir, "video_plot_" + current_time + ".mp4")
            out = cv2.VideoWriter(video_plot_save_path, fourcc, fps, (width, height))   # 初始化视频写入器

        yolo_time, sort_time, avg_fps = [], [], []
        t_start = time.time()

        idx_frame = 0
        last_bytesort = None    # 跳过的帧不绘制，会导致检测框闪烁

        while True:
            ret, frame = cap.read()
            t0 = time.time()

            if not ret or cv2.waitKey(1) & 0xFF == ord('q'):    # 结束 或 按 'q' 键退出
                break

            if self.track_cfg["video_shape"][0] > 32 and self.track_cfg["video_shape"][1] > 32:
                frame = resize_and_pad(frame, self.track_cfg["video_shape"])

            if idx_frame % self.track_cfg["vid_stride"] == 0:
                track_output, cost_time = vt.image_track(frame, idx_frame)       # 追踪结果，检测结果，消耗时间
                last_bytesort = track_output
                yolo_time.append(cost_time[0])          # yolo推理时间
                sort_time.append(cost_time[1])          # bytesort跟踪时间

                if self.track_cfg["verbose"]:
                    print('INFO: Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, *cost_time))

                plot_img = vt.plot_track(frame, track_output)                  # 绘制加入追踪框的图片
                vt.save_track(idx_frame, plot_img, track_output)      # 保存跟踪结果
            else:
                plot_img = vt.plot_track(frame, last_bytesort)              # 帧间隔小，物体运动幅度小，就用上一次结果

            if self.track_cfg["save_option"]["save"]:
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
        if self.track_cfg["save_option"]["save"]:
            out.release()  # 释放写入资源
        cv2.destroyAllWindows()

        avg_yolo_t, avg_sort_t = sum(yolo_time[1:]) / (len(yolo_time) - 1), sum(sort_time[1:]) / (len(sort_time) - 1)
        print(f'INFO: Avg YOLO time ({avg_yolo_t:.3f}s), Sort time ({avg_sort_t:.3f}s) per frame')
        total_t, avg_fps = time.time() - t_start, (len(avg_fps) - 1) / (sum(avg_fps[1:]) + 1e-6)
        print('INFO: Total Frame: %d, Total time (%.3fs), Avg fps (%.3f)' % (idx_frame, total_t, avg_fps))


if __name__ == '__main__':
    track_cfg = r'/home/chenjun/code/ultralytics_YOLOv8/ultralytics/cfg/bank_monitor/track.yaml'
    overrides_1 = {"task": "detect",
                   "mode": "predict",
                   "model": r'/home/chenjun/code/ultralytics_YOLOv8/weights/yolov8s.pt',
                   "verbose": False,
                   "classes": [0]
                   }

    overrides_2 = {"task": "detect",
                   "mode": "predict",
                   "model": r'/home/chenjun/code/ultralytics_YOLOv8/runs/detect/train_bank_06_14_m/weights/best.pt',
                   "verbose": False,
                   "classes": [0, 1, 2, 3]
                   }

    predictor_1 = BankDetectionPredictor(overrides=overrides_1)
    predictor_2 = BankDetectionPredictor(overrides=overrides_2)
    predictors = [predictor_1, predictor_2]

    vt = VideoTracker(track_cfg=track_cfg, predictors=predictors)
    vt.det_track_pipline()
