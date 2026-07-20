import cv2

from ultralytics import YOLO

# 1. 权重和视频路径
model_path = "./yolo26n.pt"
video_path = "./webdamages/AL4-AL98A5-M1-25010-20250716135316.mp4"
save_path = "./runs/result_yolo26.mp4"

# 2. 加载模型
model = YOLO(model_path)

# 3. 打开视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 4. 创建输出视频
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 5. 推理
    results = model.predict(source=frame, conf=0.25, iou=0.45, device=0, verbose=False)

    # 6. 绘制结果
    annotated_frame = results[0].plot()

    # 7. 保存帧
    out.write(annotated_frame)

cap.release()
out.release()

print(f"推理完成，结果保存到: {save_path}")
