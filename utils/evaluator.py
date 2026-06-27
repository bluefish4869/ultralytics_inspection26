"""评估器 - 真值评价接口."""

import json
from collections import defaultdict

import numpy as np


class COCOEvaluator:
    """COCO格式评估器."""

    def __init__(self, annotation_file: str):
        """初始化评估器.

        Args:
            annotation_file: COCO格式标注文件路径
        """
        self.annotation_file = annotation_file
        self.reset()

        # 加载真值
        with open(annotation_file) as f:
            self.coco_data = json.load(f)

        # 构建图片ID到标注的映射
        self.gt_by_image = defaultdict(list)
        for ann in self.coco_data["annotations"]:
            self.gt_by_image[ann["image_id"]].append(ann)

        # 类别信息
        self.categories = {cat["id"]: cat["name"] for cat in self.coco_data.get("categories", [])}

        print("\n📊 评估器初始化:")
        print(f"   标注文件: {annotation_file}")
        print(f"   图片数: {len(self.coco_data['images'])}")
        print(f"   标注数: {len(self.coco_data['annotations'])}")
        print(f"   类别数: {len(self.categories)}")

    def reset(self):
        """重置预测结果."""
        self.predictions = []

    def add_predictions(self, image_id: int, predictions: list[dict]):
        """添加预测结果.

        Args:
            image_id: 图片ID
            predictions: 预测列表，每个元素包含 bbox, confidence, class_id
        """
        for pred in predictions:
            self.predictions.append(
                {
                    "image_id": image_id,
                    "bbox": pred["bbox"],
                    "score": pred["confidence"],
                    "category_id": pred["class_id"],
                }
            )

    def _compute_iou(self, bbox1: list[float], bbox2: list[float]) -> float:
        """计算两个边界框的IoU."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def compute_ap(self, iou_threshold: float = 0.5) -> dict[str, float]:
        """计算平均精度.

        Args:
            iou_threshold: IoU阈值

        Returns:
            包含AP和mAP的字典
        """
        if not self.predictions:
            return {"AP": 0.0, "mAP": 0.0}

        # 按类别分组
        all_classes = set()
        for pred in self.predictions:
            all_classes.add(pred["category_id"])
        for img_id, gts in self.gt_by_image.items():
            for gt in gts:
                all_classes.add(gt["category_id"])

        ap_list = []
        class_results = {}

        for class_id in all_classes:
            # 获取该类别的预测
            class_preds = [p for p in self.predictions if p["category_id"] == class_id]
            class_preds.sort(key=lambda x: x["score"], reverse=True)

            # 统计该类别的真值数量
            gt_count = 0
            gt_matched = {}
            for img_id, gts in self.gt_by_image.items():
                for gt in gts:
                    if gt["category_id"] == class_id:
                        key = f"{img_id}_{gt['id']}"
                        gt_matched[key] = False
                        gt_count += 1

            if gt_count == 0 or not class_preds:
                ap_list.append(0.0)
                class_results[self.categories.get(class_id, str(class_id))] = 0.0
                continue

            # 计算TP/FP
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))

            for i, pred in enumerate(class_preds):
                best_iou = 0
                best_gt_key = None

                if pred["image_id"] in self.gt_by_image:
                    for gt in self.gt_by_image[pred["image_id"]]:
                        if gt["category_id"] != class_id:
                            continue

                        key = f"{pred['image_id']}_{gt['id']}"
                        if gt_matched[key]:
                            continue

                        iou = self._compute_iou(pred["bbox"], gt["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_key = key

                if best_iou >= iou_threshold and best_gt_key:
                    tp[i] = 1
                    gt_matched[best_gt_key] = True
                else:
                    fp[i] = 1

            # 计算precision和recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recalls = tp_cumsum / gt_count
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

            # 计算AP（11点插值）
            ap = 0.0
            for t in np.linspace(0, 1, 11):
                if np.sum(recalls >= t) > 0:
                    ap += np.max(precisions[recalls >= t])
            ap /= 11

            ap_list.append(ap)
            class_results[self.categories.get(class_id, str(class_id))] = ap

        return {
            "AP": np.mean(ap_list),
            "mAP": np.mean(ap_list),
            "per_class": class_results,
            "num_classes": len(ap_list),
        }

    def compute(self) -> dict:
        """计算所有评估指标."""
        results = {}

        # 计算不同IoU阈值的AP
        for iou in [0.5, 0.75]:
            ap_result = self.compute_ap(iou)
            results[f"AP@{iou}"] = ap_result["AP"]

        # 计算mAP
        results["mAP"] = results.get("AP@0.5", 0)

        # 统计信息
        results["total_predictions"] = len(self.predictions)

        return results


class VideoEvaluator:
    """视频时序评估器."""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置."""
        self.frame_results = []

    def add_frame(self, frame_id: int, detections: list[dict], timestamp: float = 0):
        """添加单帧结果."""
        self.frame_results.append(
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "num_detections": len(detections),
            }
        )

    def compute(self) -> dict:
        """计算时序指标."""
        if len(self.frame_results) < 2:
            return {"consistency": 0.0}

        det_counts = [f["num_detections"] for f in self.frame_results]
        mean_count = np.mean(det_counts)
        std_count = np.std(det_counts)

        # 时序一致性：检测数量的稳定性
        consistency = 1.0 - (std_count / (mean_count + 1e-6))
        consistency = max(0.0, min(1.0, consistency))

        return {
            "consistency": consistency,
            "mean_detections": mean_count,
            "std_detections": std_count,
            "total_frames": len(self.frame_results),
        }
