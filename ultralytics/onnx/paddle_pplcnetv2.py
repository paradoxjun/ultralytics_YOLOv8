import torch
from ultralytics.onnx.paddle_utils import load_onnx_model
from ultralytics.onnx.paddle_utils import preprocess_image as preprocess
from ultralytics.onnx.paddle_utils import spherical_normalize as postprocess
import time


class PPLCNetv2Predictor:
    def __init__(self, model_path, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.session = load_onnx_model(model_path, self.use_cuda)
        self._preprocess = preprocess
        self._postprocess = postprocess

    def __call__(self, img):
        img_batch = self._preprocess(img)
        # start = time.time()
        onnx_output = self.session.run(None, {self.session.get_inputs()[0].name: img_batch})[0]     # ONNX模型推理
        feature = self._postprocess(onnx_output)
        # print(f"推理{img_batch.shape[0]}张图片，用时：{time.time() - start}")

        return feature
