from ultralytics import YOLO

# 加载模型
model = YOLO("/mnt/pfs-baidu/public/muyuan.zhang/ultralytics-main/ckpts/yolo26n-seg.pt")

# 训练
model.train(
    data="/mnt/pfs-baidu/public/muyuan.zhang/ultralytics-main/configs/data_coco_seg.yaml",
    epochs=300,
    imgsz=640,
    batch=2,
    device=0,
)
