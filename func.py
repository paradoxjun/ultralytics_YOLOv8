from pathlib import Path
import os
import glob
from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS


def get_img_files(img_path, prefix="train"):
    """Read image files."""
    try:
        f = []  # image files
        for p in img_path if isinstance(img_path, list) else [img_path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                # F = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                    # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise FileNotFoundError(f"{prefix}{p} does not exist")
        im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
        # img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
        assert im_files, f"{prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
    except Exception as e:
        raise FileNotFoundError(f"{prefix}Error loading data from {img_path}\n{HELP_URL}") from e
    return im_files


if __name__ == '__main__':
    img_path = [#"/home/chenjun/code/datasets/bank_monitor/seed_img/image_list.txt",
                #"/home/chenjun/code/datasets/bank_monitor/seed_img/image_list_2.txt",
                "/home/chenjun/code/datasets/bank_monitor/seed_img"]
    im = get_img_files(img_path)
    print(len(im))
    for e in im:
        print(e)
