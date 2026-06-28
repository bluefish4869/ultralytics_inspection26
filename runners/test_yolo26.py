#!/usr/bin/env python3
"""
YOLO26 视频推理测试程序
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import yaml
import cv2

from ultralytics import YOLO
from utils.data_loader import create_loader, VideoDataLoader, ImageLoader
from utils.evaluator import COCOEvaluator, VideoEvaluator


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_annotated_image(image: np.ndarray, output_path: Path, results):
    """保存标注后的图片"""
    annotated = results[0].plot()
    cv2.imwrite(str(output_path), annotated)
    return output_path


def parse_task_type(config: dict) -> str:
    """解析任务类型，兼容 task_type/type 两种字段。"""
    task_type = str(config.get("task_type", config.get("type", "infer"))).strip().lower()
    aliases = {
        "infer": "infer",
        "det": "infer",
        "detect": "infer",
        "detection": "infer",
        "seg": "seg",
        "segment": "seg",
        "segmentation": "seg",
    }
    return aliases.get(task_type, "infer")


def save_mask_images(result, mask_dir: Path, frame_id: int):
    """保存每个实例的二值 mask 图。"""
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


def parse_predictions(result, task_type: str):
    """按任务类型解析预测结果。"""
    predictions = []
    boxes = result.boxes

    if boxes is None:
        return predictions

    for i, box in enumerate(boxes):
        pred = {
            "bbox": box.xyxy[0].cpu().numpy().tolist(),
            "confidence": float(box.conf[0]),
            "class_id": int(box.cls[0]),
        }

        if task_type == "seg":
            pred["mask_area"] = 0
            pred["polygon"] = []
            if result.masks is not None:
                mask_np = result.masks.data[i].cpu().numpy()
                pred["mask_area"] = int(mask_np.sum())
                if result.masks.xy is not None and len(result.masks.xy) > i:
                    pred["polygon"] = result.masks.xy[i].tolist()

        predictions.append(pred)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="YOLO26 视频推理测试")
    parser.add_argument("--config", "-c", type=str, 
                       default="configs/config.yaml",
                       help="配置文件路径")
    parser.add_argument("--input", "-i", type=str, help="输入路径（图片/视频/文件夹）")
    parser.add_argument("--output", "-o", type=str, help="输出路径（可选）")
    parser.add_argument("--device", type=str, help="设备")
    parser.add_argument("--eval", action="store_true", help="启用评估")
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 加载配置
    config_path = project_root / args.config
    config = load_config(config_path)
    task_type = parse_task_type(config)
    
    # 读取保存标志
    is_save = config.get('is_save', True)       
    save_video = config.get('save_video', False) 
    save_json = config.get('save_json', True)   
    save_pic = config.get('save_pic', False)   

    # 命令行覆盖
    if args.input:
        config['input_path'] = args.input
    if args.output:
        config['output_path'] = args.output
    if args.device:
        config['device'] = args.device
    if args.eval:
        config['eval_enabled'] = True

    # 允许通过 type/task_type 在配置中区分检测和分割
    task_type = parse_task_type(config)
    
    # 检查输入
    if not config.get('input_path'):
        print("❌ 请指定输入路径")
        return 1
    
    input_path = Path(config['input_path'])
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return 1
    
    # 确定输出路径
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('experiment_name', 'test_yolo26')
    runs_dir = project_root / 'runs' / f"test_{experiment_name}_{timestamp}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定输出路径
    if args.output:
        # 如果用户指定了输出路径，使用用户指定的
        output_path = Path(args.output)
    else:
        # 根据输入类型自动生成输出路径
        if input_path.is_file():
            # 单张图片或单个视频
            output_path = runs_dir / f"{input_path.stem}_output{input_path.suffix}"
        else:
            # 文件夹
            output_path = runs_dir / f"{input_path.name}_output"
    
    print(f"\n📁 输入: {input_path}")
    print(f"📁 输出: {output_path}")
    print(f"📁 实验目录: {runs_dir}")
    print(f"🧩 任务类型: {task_type}")
    
    # 加载模型
    print(f"\n🚀 加载模型: {config['model']}")
    model = YOLO(config['model'])
    device = config.get('device', 'cpu')
    if isinstance(device, str) and device.lower() != 'cpu':
        model.to(device)
    print("✅ 模型加载完成")
    
    # 创建数据加载器
    loader = create_loader(config)
    
    # 判断输入类型
    is_video = isinstance(loader, VideoDataLoader)
    is_image_folder = isinstance(loader, ImageLoader)
    
    # 创建评估器
    coco_evaluator = None
    video_evaluator = None
    
    if config.get('eval_enabled'):
        if config.get('gt_annotation_file'):
            gt_path = project_root / config['gt_annotation_file']
            coco_evaluator = COCOEvaluator(str(gt_path))
        video_evaluator = VideoEvaluator()
    
    # 输出视频（仅视频输入时）
    video_writer = None
    if is_save and is_video and save_video:  
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, loader.fps,
                                       (loader.width, loader.height))
        print(f"📹 输出视频: {output_path}")
    
    # 统计
    inference_times = []
    detections_per_frame = []
    mask_counts_per_frame = []
    saved_files = []
    
    # 处理
    print("\n开始处理...")
    start_time = time.time()
    
    for frame, frame_id, timestamp in tqdm(loader, total=len(loader), desc="推理进度"):
        # 推理
        inf_start = time.time()
        infer_kwargs = {
            'conf': config['conf_threshold'],
            'iou': config['iou_threshold'],
            'verbose': False,
            'device': device,
        }
        if config.get('imgsz'):
            infer_kwargs['imgsz'] = config['imgsz']
        model_results = model.predict(frame, **infer_kwargs)
        inf_time = time.time() - inf_start
        
        inference_times.append(inf_time)
        
        # 解析预测
        predictions = parse_predictions(model_results[0], task_type)
        
        detections_per_frame.append(len(predictions))
        if task_type == 'seg':
            mask_count = int(model_results[0].masks.data.shape[0]) if model_results[0].masks is not None else 0
            mask_counts_per_frame.append(mask_count)
        
        # 评估
        if coco_evaluator:
            coco_evaluator.add_predictions(frame_id, predictions)
        if video_evaluator:
            video_evaluator.add_frame(frame_id, predictions, timestamp)
        
        # 保存输出
        if not is_save: 
            continue

        # 单张图片输入：输出单张图片
        if save_pic:
            # 创建统一的图片子文件夹
            pic_dir = output_path.parent / f"{output_path.stem}_pics"
            pic_dir.mkdir(parents=True, exist_ok=True)
            
            # 确定保存文件名
            if is_video:
                out_file = pic_dir / f"frame_{frame_id:06d}.jpg"
            elif is_image_folder:
                if hasattr(loader, 'image_files') and frame_id < len(loader.image_files):
                    original_name = Path(loader.image_files[frame_id]).stem
                    out_file = pic_dir / f"{original_name}_output.jpg"
                else:
                    out_file = pic_dir / f"frame_{frame_id:06d}.jpg"
            else:
                # 单张图片输入
                if task_type == 'seg':
                    out_file = pic_dir / f"{input_path.stem}_seg.jpg"
                else:
                    out_file = pic_dir / f"{input_path.stem}_output.jpg"
            save_annotated_image(frame, out_file, model_results)
            saved_files.append(out_file)

            if task_type == 'seg' and config.get('save_masks', True):
                mask_dir = output_path.parent / f"{output_path.stem}_masks"
                mask_dir.mkdir(parents=True, exist_ok=True)
                save_mask_images(model_results[0], mask_dir, frame_id)
        
        if is_video and video_writer and save_video:
            # 视频输入：输出视频
            annotated = model_results[0].plot()
            video_writer.write(annotated)
        
    # 清理
    loader.release()
    if video_writer:
        video_writer.release()
    
    # 统计结果
    total_time = time.time() - start_time
    total_frames = len(inference_times)
    
    summary = {
        'input': str(input_path),
        'output': str(output_path),
        'task_type': task_type,
        'total_frames': total_frames,
        'total_time': total_time,
        'avg_fps': total_frames / total_time if total_time > 0 else 0,
        'avg_inference_time_ms': np.mean(inference_times) * 1000 if inference_times else 0,
        'std_inference_time_ms': np.std(inference_times) * 1000 if inference_times else 0,
        'p95_inference_time_ms': np.percentile(inference_times, 95) * 1000 if inference_times else 0,
        'total_detections': sum(detections_per_frame),
        'avg_detections_per_frame': np.mean(detections_per_frame) if detections_per_frame else 0,
    }
    if task_type == 'seg':
        summary['total_masks'] = int(sum(mask_counts_per_frame))
        summary['avg_masks_per_frame'] = float(np.mean(mask_counts_per_frame)) if mask_counts_per_frame else 0.0
    
    # 评估结果
    if coco_evaluator:
        eval_results = coco_evaluator.compute()
        summary['evaluation'] = eval_results
    
    if video_evaluator:
        temporal_results = video_evaluator.compute()
        summary['temporal'] = temporal_results
    
    # 打印结果
    print("\n" + "="*50)
    print("测试结果")
    print("="*50)
    print(f"处理帧数: {summary['total_frames']}")
    print(f"总耗时: {summary['total_time']:.2f} 秒")
    print(f"平均FPS: {summary['avg_fps']:.2f}")
    print(f"平均推理时间: {summary['avg_inference_time_ms']:.2f} ms")
    print(f"P95推理时间: {summary['p95_inference_time_ms']:.2f} ms")
    print(f"总检测数: {summary['total_detections']}")
    print(f"平均每帧检测: {summary['avg_detections_per_frame']:.2f}")
    if task_type == 'seg':
        print(f"总Mask数: {summary['total_masks']}")
        print(f"平均每帧Mask: {summary['avg_masks_per_frame']:.2f}")
    
    if coco_evaluator:
        print(f"\n评估指标:")
        print(f"  AP@0.5: {summary['evaluation'].get('AP@0.5', 0):.4f}")
        print(f"  AP@0.75: {summary['evaluation'].get('AP@0.75', 0):.4f}")
        print(f"  mAP: {summary['evaluation'].get('mAP', 0):.4f}")
    
    if video_evaluator:
        print(f"\n时序指标:")
        print(f"  一致性: {summary['temporal'].get('consistency', 0):.4f}")
    
    print("="*50)
    print(f"\n✅ 输出已保存到: {output_path}")
    if saved_files and len(saved_files) <= 10:
        for f in saved_files:
            print(f"   - {f}")
    elif saved_files:
        print(f"   ... 共 {len(saved_files)} 个文件")
    
    # 保存JSON结果
    if save_json:
        result_file = output_path.parent / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if output_path.is_dir():
            result_file = output_path / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ 结果已保存: {result_file}")
    
    print(f"✅ 测试完成！")
    return 0


if __name__ == "__main__":
    exit(main())