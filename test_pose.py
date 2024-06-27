from ultralytics.task_bank.pose.predict import PosePredictor
from ultralytics.task_bank.pose.utils import get_upper_body_keypoint, plot_keypoint, image_show

video_path = r"/home/chenjun/下载/bank2406-柜台垂直视角1/城东柜员1/城东柜员1全景_20240201161000-20240201162000_1.mp4"
img_path = r"./ultralytics/assets/73597451_32441835.jpg"

overrides = {"task": "pose",
             "mode": "predict",
             "model": r'./weights/yolov8m-pose.pt',
             "verbose": False,
             "classes": [0]
             }

pose_predictor = PosePredictor(overrides=overrides)
res = pose_predictor(source=img_path)
data = get_upper_body_keypoint(res[0].keypoints.data)
# print(res[0].keypoints)
# print(res[0].boxes)
# res[0].show()
image_show(plot_keypoint(img_path, data))
