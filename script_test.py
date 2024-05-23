# from pathlib import Path
#
# FILE = Path(__file__).resolve()
# print(FILE.parents[0])
# ROOT = FILE.parents[1]  # YOLO
# print(ROOT)
# DEFAULT_CFG_PATH = ROOT / "cfg/bank_monitor/detect_predict.yaml"


from ultralytics.data.augment import LetterBox
import cv2


img_path = r'/home/chenjun/code/ultralytics_YOLOv8/ultralytics/assets/zidane.jpg'

img = cv2.imread(img_path)

print("*******")
cv2.imshow("123", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("*******")
lb = LetterBox()
img_resize = lb(image=img)
cv2.imshow("123", img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
