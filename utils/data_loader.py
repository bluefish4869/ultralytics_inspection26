"""数据加载器 - 解耦视频/图片解析."""

from __future__ import annotations

from pathlib import Path

import cv2


class VideoDataLoader:
    """视频数据加载器."""

    def __init__(self, video_path: str, max_frames: int | None = None, frame_skip: int = 1):
        # ... 保持不变 ...
        self.video_path = video_path
        self.max_frames = max_frames
        self.frame_skip = frame_skip

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames and max_frames < self.total_frames:
            self.total_frames = max_frames

        self.num_frames = (self.total_frames + frame_skip - 1) // frame_skip

        print("\n📹 视频加载器:")
        print(f"   路径: {video_path}")
        print(f"   分辨率: {self.width}x{self.height}")
        print(f"   FPS: {self.fps:.2f}")
        print(f"   总帧数: {self.total_frames}")
        print(f"   跳帧: {self.frame_skip}")
        print(f"   将处理: {self.num_frames} 帧")

    def __len__(self) -> int:
        return self.num_frames

    def __iter__(self):
        self.current_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self):
        if self.current_idx >= self.num_frames:
            raise StopIteration

        frame_pos = self.current_idx * self.frame_skip
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = self.cap.read()

        if not ret:
            raise StopIteration

        frame_id = self.current_idx
        timestamp = frame_pos / self.fps if self.fps > 0 else 0

        self.current_idx += 1
        return frame, frame_id, timestamp

    def release(self):
        if self.cap:
            self.cap.release()


class ImageLoader:
    """单张图片或图片文件夹加载器."""

    def __init__(self, path: str):
        self.path = Path(path)

        # 判断是单张图片还是文件夹
        if self.path.is_file():
            # 单张图片
            self.image_files = [self.path]
        else:
            # 文件夹
            self.image_files = sorted(
                [f for f in self.path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
            )

        if not self.image_files:
            raise ValueError(f"没有找到图片文件: {path}")

        # 获取第一张图片的尺寸
        first_img = cv2.imread(str(self.image_files[0]))
        if first_img is not None:
            self.height, self.width = first_img.shape[:2]
        else:
            self.width, self.height = 0, 0

        self.fps = 1.0

        print("\n🖼️ 图片加载器:")
        print(f"   路径: {path}")
        print(f"   图片数: {len(self.image_files)}")
        if self.width > 0:
            print(f"   分辨率: {self.width}x{self.height}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.image_files):
            raise StopIteration

        img_path = self.image_files[self.current_idx]
        img = cv2.imread(str(img_path))

        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")

        frame_id = self.current_idx
        self.current_idx += 1

        return img, frame_id, 0.0

    def release(self):
        pass


def create_loader(config: dict):
    """工厂方法：创建数据加载器."""
    input_path = config.get("input_path", "")

    if not input_path:
        raise ValueError("未指定输入路径")

    path = Path(input_path)

    # 检查文件或目录是否存在
    if not path.exists():
        raise ValueError(f"路径不存在: {input_path}")

    # 视频文件
    if path.is_file() and path.suffix.lower() in [".mp4", ".avi", ".mov", ".mpg", ".mpeg"]:
        return VideoDataLoader(str(path), max_frames=config.get("max_frames"), frame_skip=config.get("frame_skip", 1))

    # 图片文件或图片文件夹
    elif path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        return ImageLoader(str(path))

    # 文件夹
    elif path.is_dir():
        return ImageLoader(str(path))

    else:
        raise ValueError(f"不支持的输入类型: {input_path}，请使用图片(.jpg/.png)或视频(.mp4/.avi)文件")


# data_loader.py 末尾，确保导出这两个类
__all__ = ["ImageLoader", "VideoDataLoader", "create_loader"]
