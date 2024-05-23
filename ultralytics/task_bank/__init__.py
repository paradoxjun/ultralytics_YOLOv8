from ultralytics.utils import yaml_load, IterableSimpleNamespace
from pathlib import Path

# Default configuration
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
BANK_DEFAULT_CFG_PATH = ROOT / "cfg/bank_monitor/detect_predict.yaml"
DEFAULT_CFG_DICT = yaml_load(BANK_DEFAULT_CFG_PATH)

for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None

DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
