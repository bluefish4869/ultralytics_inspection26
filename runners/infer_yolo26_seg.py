#!/usr/bin/env python3
"""
YOLO26-Seg 分割推理 Demo
支持：图片 / 视频 / 文件夹
输出：可视化图片/视频、JSON结果、mask像素图.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from ultralytics import YOLO


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_image(path):
    return path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def is_video(path):
    return path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".flv"]


def save_mask_images(result, mask_dir, frame_id):
    """保存每个实例的二值mask图."""
    mask_paths = []

    if result.masks is None:
        return mask_paths

    masks = result.masks.data.cpu().numpy()

    for i, mask in enumerate(masks):
        mask_img = (mask * 255).astype(np.uint8)
        mask_path = mask_dir / f"frame_{frame_id:06d}_mask_{i}.png"
        cv2.imwrite(str(mask_path), mask_img)
        mask_paths.append(str(mask_path))

    return mask_paths


def parse_seg_result(result):
    """解析YOLO26-Seg输出."""
    predictions = []

    boxes = result.boxes
    masks = result.masks

    if boxes is None:
        return predictions

    for i, box in enumerate(boxes):
        pred = {
            "bbox": box.xyxy[0].cpu().numpy().tolist(),
            "confidence": float(box.conf[0]),
            "class_id": int(box.cls[0]),
            "mask_area": 0,
            "polygon": [],
        }

        if masks is not None:
            # mask面积
            mask_np = masks.data[i].cpu().numpy()
            pred["mask_area"] = int(mask_np.sum())

            # polygon坐标，原图尺度
            if masks.xy is not None and len(masks.xy) > i:
                pred["polygon"] = masks.xy[i].tolist()

        predictions.append(pred)

    return predictions


def infer_image(model, image_path, output_dir, config):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"读取图片失败: {image_path}")
        return None

    results = model.predict(
        frame,
        conf=config["conf_threshold"],
        iou=config["iou_threshold"],
        imgsz=config["imgsz"],
        device=config["device"],
        verbose=False,
    )

    result = results[0]
    annotated = result.plot()

    out_img = output_dir / f"{image_path.stem}_seg.jpg"
    cv2.imwrite(str(out_img), annotated)

    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    mask_paths = save_mask_images(result, mask_dir, 0)

    return {
        "image": str(image_path),
        "output": str(out_img),
        "predictions": parse_seg_result(result),
        "mask_files": mask_paths,
    }


def infer_video(model, video_path, output_dir, config):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = output_dir / f"{video_path.stem}_seg.mp4"

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    mask_dir = output_dir / "masks"
    if config.get("save_masks", True):
        mask_dir.mkdir(exist_ok=True)

    all_results = []
    inference_times = []

    frame_id = 0

    for _ in tqdm(range(total), desc="YOLO26-Seg 视频推理"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % config.get("frame_skip", 1) != 0:
            frame_id += 1
            continue

        t0 = time.time()

        results = model.predict(
            frame,
            conf=config["conf_threshold"],
            iou=config["iou_threshold"],
            imgsz=config["imgsz"],
            device=config["device"],
            verbose=False,
        )

        infer_time = time.time() - t0
        inference_times.append(infer_time)

        result = results[0]
        annotated = result.plot()

        writer.write(annotated)

        mask_paths = []
        if config.get("save_masks", True):
            mask_paths = save_mask_images(result, mask_dir, frame_id)

        frame_result = {
            "frame_id": frame_id,
            "timestamp": frame_id / fps if fps > 0 else 0,
            "predictions": parse_seg_result(result),
            "mask_files": mask_paths,
        }

        all_results.append(frame_result)

        frame_id += 1

    cap.release()
    writer.release()

    return {
        "video": str(video_path),
        "output_video": str(out_video),
        "total_frames": len(all_results),
        "avg_inference_time_ms": float(np.mean(inference_times) * 1000) if inference_times else 0,
        "avg_fps": float(1 / np.mean(inference_times)) if inference_times else 0,
        "frames": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO26-Seg 分割推理 Demo")
    parser.add_argument("--config", "-c", default="configs/seg_config.yaml")
    parser.add_argument("--input", "-i", default=None)
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.input:
        config["input_path"] = args.input
    if args.model:
        config["model"] = args.model
    if args.device:
        config["device"] = args.device

    input_path = Path(config["input_path"])

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("output_dir", "runs/segment_demo")) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"模型: {config['model']}")
    print(f"输入: {input_path}")
    print(f"输出目录: {output_dir}")

    model = YOLO(config["model"])

    results = []

    if input_path.is_file() and is_image(input_path):
        results.append(infer_image(model, input_path, output_dir, config))

    elif input_path.is_file() and is_video(input_path):
        results.append(infer_video(model, input_path, output_dir, config))

    elif input_path.is_dir():
        for p in sorted(input_path.iterdir()):
            if is_image(p):
                results.append(infer_image(model, p, output_dir, config))
            elif is_video(p):
                results.append(infer_video(model, p, output_dir, config))

    else:
        raise ValueError(f"不支持的输入类型: {input_path}")

    json_path = output_dir / "seg_results.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n推理完成")
    print(f"结果JSON: {json_path}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
