"""示例脚本：使用 Ultralytics YOLO 直接基于 COCO JSON 训练

用法：
    python tools/train_coco_json.py --config configs/yolo26_0.yaml
或覆盖参数：
    python tools/train_coco_json.py --train_json /full/path/instances_train.json --val_json /full/path/instances_val.json --epochs 50

脚本会读取配置中的 training 字段并使用自定义 COCO Trainer（基于 docs/guides/coco-json-training.md 的实现）。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import TQDM, colorstr


class COCODataset(YOLODataset):
    def __init__(self, *args, json_file="", **kwargs):
        self.json_file = json_file
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        x = {"labels": []}
        with open(self.json_file) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        categories = {cat["id"]: i for i, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}

        from collections import defaultdict

        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in TQDM(coco["images"], desc="reading annotations"):
            h, w = img_info["height"], img_info["width"]
            im_file = Path(self.img_path) / img_info["file_name"]
            if not im_file.exists():
                continue

            self.im_files.append(str(im_file))
            bboxes = []
            for ann in img_to_anns.get(img_info["id"], []):
                if ann.get("iscrowd", False):
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = categories[ann["category_id"]]
                bboxes.append([cls, *box.tolist()])

            lb = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        x["hash"] = get_hash([self.json_file, str(self.img_path)])
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([self.json_file, str(self.img_path)])
            self.im_files = [lb["im_file"] for lb in cache["labels"]]
        except (FileNotFoundError, AssertionError, AttributeError, KeyError, ModuleNotFoundError):
            cache = self.cache_labels(cache_path)
        cache.pop("hash", None)
        cache.pop("version", None)
        return cache["labels"]


class COCOTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        json_file = self.data.get("train_json") if mode == "train" else self.data.get("val_json", self.data.get("train_json"))
        return COCODataset(
            img_path=img_path,
            json_file=json_file,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.model.stride.max()) if hasattr(self, "model") and self.model else 32,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/yolo26_0.yaml", help="path to config yaml")
    parser.add_argument("--train_json", type=str, help="override train json path")
    parser.add_argument("--val_json", type=str, help="override val json path")
    parser.add_argument("--epochs", type=int, help="override epochs")
    parser.add_argument("--imgsz", type=int, help="override imgsz")
    parser.add_argument("--batch", type=int, help="override batch")
    return parser.parse_args()


def main():
    args = parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg = cfg.get("training", {})

    train_json = args.train_json or tcfg.get("train_json")
    val_json = args.val_json or tcfg.get("val_json")
    dataset_root = tcfg.get("dataset_root")
    epochs = args.epochs or tcfg.get("epochs", 100)
    imgsz = args.imgsz or tcfg.get("imgsz", 640)
    batch = args.batch or tcfg.get("batch", 16)

    if not train_json and dataset_root:
        train_json = str(Path(dataset_root) / "annotations" / "instances_train.json")
    if not val_json and dataset_root:
        val_json = str(Path(dataset_root) / "annotations" / "instances_val.json")

    assert train_json and Path(train_json).exists(), f"train_json not found: {train_json}"

    # 构造 dataset.yaml 临时文件
    data_yaml = {
        "path": str(Path(dataset_root) if dataset_root else Path(train_json).parents[1]),
        "train": "images/train",
        "val": "images/val",
        "train_json": train_json,
        "val_json": val_json or train_json,
        "nc": tcfg.get("nc", 0),
    }

    tmp_data_yaml = Path(".tmp_coco_dataset.yaml")
    import yaml
    tmp_data_yaml.write_text(yaml.safe_dump(data_yaml))

    model_path = cfg.get("model") or "yolo26n.pt"
    model = YOLO(model_path)
    model.train(data=str(tmp_data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, trainer=COCOTrainer)


if __name__ == "__main__":
    main()
