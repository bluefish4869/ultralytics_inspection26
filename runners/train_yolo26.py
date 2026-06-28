#!/usr/bin/env python3
"""
YOLO26 检测训练程序（连续训练 + 每轮自定义评估回调）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml
from ultralytics import YOLO

from utils.evaluator import COCOEvaluator


def load_config(config_path: Path) -> Dict:
    """加载 YAML 配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_device(device_value):
    """兼容配置里的 cpu/cuda/数字设备写法。"""
    if isinstance(device_value, int):
        return device_value
    if isinstance(device_value, str):
        value = device_value.strip().lower()
        if value.isdigit():
            return int(value)
        if value in {"cpu", "mps"}:
            return value
        if value == "cuda":
            return 0
        return device_value
    return 0


def load_val_images(val_cfg: str, project_root: Path) -> List[Path]:
    """支持从 txt/目录/单文件收集验证图片列表。"""
    val_path = Path(val_cfg)
    if not val_path.is_absolute():
        val_path = (project_root / val_path).resolve()

    if val_path.suffix.lower() == ".txt" and val_path.exists():
        lines = [ln.strip() for ln in val_path.read_text(encoding="utf-8").splitlines()]
        return [Path(ln) for ln in lines if ln]

    if val_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([p for p in val_path.rglob("*") if p.suffix.lower() in exts])

    if val_path.is_file():
        return [val_path]

    return []


def collect_predictions(result) -> List[Dict]:
    """将单张图像预测结果转为 COCOEvaluator 需要的结构。"""
    predictions = []
    boxes = result.boxes
    if boxes is None:
        return predictions

    for box in boxes:
        xyxy = box.xyxy[0].cpu().tolist()
        predictions.append(
            {
                "bbox": xyxy,
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0]),
            }
        )
    return predictions


def build_image_name_mapping(coco_data: Dict) -> Dict[str, int]:
    """建立 file_name 与 image_id 的映射，优先按 basename 兜底。"""
    mapping = {}
    for img in coco_data.get("images", []):
        file_name = str(img.get("file_name", ""))
        image_id = int(img.get("id", -1))
        if not file_name or image_id < 0:
            continue
        mapping[file_name] = image_id
        mapping[Path(file_name).name] = image_id
    return mapping


def run_custom_eval(
    model: YOLO,
    config: Dict,
    project_root: Path,
    epoch: int,
    eval_max_images: int,
    device,
    run_dir: Path,
) -> Dict:
    """在每个 epoch 后执行自定义评估。"""
    gt_file = str(config.get("gt_annotation_file", "")).strip()
    val_cfg = str(config.get("val", "")).strip()
    if not gt_file:
        return {"skipped": True, "reason": "gt_annotation_file 为空，跳过自定义评估"}
    if not val_cfg:
        return {"skipped": True, "reason": "配置缺少 val 字段，跳过自定义评估"}

    gt_path = Path(gt_file)
    if not gt_path.is_absolute():
        gt_path = (project_root / gt_path).resolve()
    if not gt_path.exists():
        return {"skipped": True, "reason": f"标注文件不存在: {gt_path}"}

    val_images = load_val_images(val_cfg, project_root)
    if not val_images:
        return {"skipped": True, "reason": "未找到可用于评估的验证图片"}

    evaluator = COCOEvaluator(str(gt_path))
    image_id_mapping = build_image_name_mapping(evaluator.coco_data)

    conf_thres = float(config.get("conf_threshold", 0.25))
    iou_thres = float(config.get("iou_threshold", 0.45))

    used_images = 0
    for img_path in val_images:
        if eval_max_images > 0 and used_images >= eval_max_images:
            break
        if not img_path.exists():
            continue

        image_id = image_id_mapping.get(str(img_path), image_id_mapping.get(img_path.name))
        if image_id is None:
            continue

        results = model.predict(
            source=str(img_path),
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            verbose=False,
        )
        preds = collect_predictions(results[0])
        evaluator.add_predictions(image_id, preds)
        used_images += 1

    metrics = evaluator.compute()
    metrics["epoch"] = epoch
    metrics["used_images"] = used_images

    eval_dir = run_dir / "custom_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_file = eval_dir / f"epoch_{epoch:03d}.json"
    out_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="YOLO26 检测训练")
    parser.add_argument("--config", "-c", type=str, default="configs/yolo26_det.yaml", help="配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="总训练轮数，覆盖配置")
    parser.add_argument("--imgsz", type=int, default=None, help="输入尺寸，覆盖配置")
    parser.add_argument("--batch", type=int, default=None, help="batch size，覆盖配置")
    parser.add_argument("--device", type=str, default=None, help="训练设备，覆盖配置")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径，覆盖配置")
    parser.add_argument("--eval", action="store_true", help="强制开启自定义评估")
    parser.add_argument("--eval-max-images", type=int, default=100, help="每轮最多评估图片数，0 表示不限制")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.resolve()
    config_path = project_root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    config = load_config(config_path)

    # 命令行覆盖
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.imgsz is not None:
        config["imgsz"] = args.imgsz
    if args.batch is not None:
        config["batch"] = args.batch
    if args.device is not None:
        config["device"] = args.device
    if args.model is not None:
        config["model"] = args.model
    if args.eval:
        config["eval_enabled"] = True

    # 训练参数默认值
    total_epochs = int(config.get("epochs", 10))
    imgsz = int(config.get("imgsz", 640))
    batch = int(config.get("batch", 2))
    workers = int(config.get("workers", 8))
    device = parse_device(config.get("device", 0))
    model_path = str(config.get("model", "")).strip()
    if not model_path:
        raise ValueError("配置缺少 model 字段")
    amp = bool(config.get("amp", False))
    eval_enabled = bool(config.get("eval_enabled", False))

    # 使用配置文件自身作为数据集描述（包含 path/train/val/nc/names）
    data_cfg_path = str(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = str(config.get("experiment_name", "yolo26_det"))
    run_name = f"train_{experiment_name}_{timestamp}"
    run_dir = project_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("YOLO26 连续训练")
    print("=" * 70)
    print(f"配置文件: {config_path}")
    print(f"模型路径: {model_path}")
    print(f"数据配置: {data_cfg_path}")
    print(f"总轮数: {total_epochs}")
    print(f"设备: {device}")
    print(f"实验目录: {run_dir}")
    print(f"自定义评估: {'开启' if eval_enabled else '关闭'}")
    print("=" * 70)

    model = YOLO(model_path)

    history = []
    history_path = run_dir / "train_history.json"

    def on_fit_epoch_end(trainer):
        """每轮结束后记录信息，并可选执行自定义评估。"""
        epoch = int(getattr(trainer, "epoch", -1)) + 1

        # final_eval 会额外触发一次 on_fit_epoch_end，这里只保留真实训练轮次。
        if epoch < 1 or epoch > total_epochs:
            return

        # 防止同一轮被重复记录。
        if history and int(history[-1].get("epoch", -1)) == epoch:
            return

        print(f"\n[Train] Epoch {epoch}/{total_epochs} 完成")
        epoch_record = {
            "epoch": epoch,
            "fitness": float(trainer.fitness) if getattr(trainer, "fitness", None) is not None else None,
            "lr": {k: float(v) for k, v in getattr(trainer, "lr", {}).items()},
        }

        if eval_enabled:
            print(f"[Eval] Epoch {epoch}: 运行自定义评估...")
            try:
                eval_metrics = run_custom_eval(
                    model=model,
                    config=config,
                    project_root=project_root,
                    epoch=epoch,
                    eval_max_images=args.eval_max_images,
                    device=device,
                    run_dir=run_dir,
                )
            except Exception as e:
                eval_metrics = {"skipped": True, "reason": f"自定义评估异常: {e}"}

            epoch_record["custom_eval"] = eval_metrics
            if eval_metrics.get("skipped"):
                print(f"[Eval] 跳过: {eval_metrics.get('reason')}")
            else:
                print(f"[Eval] mAP: {eval_metrics.get('mAP', 0):.4f}")

        history.append(epoch_record)
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    train_result = model.train(
        data=data_cfg_path,
        epochs=total_epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=str(project_root / "runs"),
        name=run_name,
        exist_ok=True,
        resume=False,
        amp=amp,
        val=False,
        verbose=True,
    )

    summary = {
        "train_result": str(train_result),
        "epochs": total_epochs,
        "history_file": str(history_path),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n训练完成")
    print(f"历史记录: {run_dir / 'train_history.json'}")


if __name__ == "__main__":
    main()