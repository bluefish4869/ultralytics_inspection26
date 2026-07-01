import os
import json
import argparse
from pathlib import Path
from datetime import datetime

def extract_frames(video_path, output_dir, frame_interval=1, image_format='jpg', quality=95):
    """
    从视频中提取帧图片，并生成Label Studio导入用的data.json文件
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出图片的文件夹路径
        frame_interval: 提取间隔，1表示每帧都提取，2表示隔一帧提取一帧
        image_format: 输出图片格式 (jpg, png)
        quality: 图片质量 (1-100)，仅对jpg有效
    """
    try:
        import cv2
    except ImportError:
        print("错误: 需要安装 opencv-python")
        print("请运行: pip install opencv-python")
        return
    
    # 检查视频文件是否存在
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    # 如果output_dir为None，使用默认的runs/时间戳路径
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name =  "extra_" + timestamp
        output_dir = Path("./runs") / dir_name
    else:
        output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取绝对路径
    output_dir_abs = output_dir.absolute()
    frames_dir = output_dir_abs / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"  - 视频文件: {video_path.name}")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 帧率: {fps:.2f} FPS")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - 提取间隔: 每 {frame_interval} 帧提取一帧")
    print(f"  - 预计提取: ~{total_frames // frame_interval} 张图片")
    print(f"  - 输出目录: {output_dir_abs}")
    print("-" * 50)
    
    frame_count = 0
    saved_count = 0
    error_count = 0
    file_name_base = video_path.stem  # 获取文件名（不含扩展名）
    
    # 存储所有图片路径，用于生成data.json
    image_paths = []
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 按间隔提取帧
        if frame_count % frame_interval == 0:
            try:
                # 生成文件名，用数字填充以保持排序正确
                file_name = f"{file_name_base}_frame_{saved_count:06d}.{image_format}"
                file_path = frames_dir / file_name
                
                # 保存图片
                if image_format.lower() == 'jpg':
                    success = cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                else:
                    success = cv2.imwrite(str(file_path), frame)
                
                if success:
                    # 保存绝对路径
                    image_paths.append(str(file_path))
                    saved_count += 1
                else:
                    error_count += 1
                    
                # 每100张显示一次进度
                if saved_count % 100 == 0:
                    print(f"进度: 已保存 {saved_count} 张图片")
                    
            except Exception as e:
                error_count += 1
                print(f"保存第 {saved_count} 帧时出错: {e}")
        
        frame_count += 1
    
    # 释放资源
    cap.release()
    
    print("-" * 50)
    print(f"提取完成!")
    print(f"  - 处理总帧数: {frame_count}")
    print(f"  - 成功保存: {saved_count} 张图片")
    print(f"  - 保存目录: {output_dir_abs}")
    if error_count > 0:
        print(f"  - 保存失败: {error_count} 张")
    
    # 生成data.json文件
    if saved_count > 0:
        generate_data_json(image_paths, output_dir_abs, file_name_base, image_format)
    
    # 生成标注配置提示
    print("\n" + "=" * 50)
    print("📋 建议的标注配置（用于图片标注）：")
    print("""<View>
  <Image name="img" value="$image"/>
  <RectangleLabels name="label" toName="img">
    <Label value="目标类别1" background="#FF0000"/>
    <Label value="目标类别2" background="#00FF00"/>
  </RectangleLabels>
</View>""")

def generate_data_json(image_paths, output_dir, file_name_base, image_format):
    """
    生成Label Studio导入用的data.json文件
    """
    # 构建JSON数据
    data = []
    for img_path in image_paths:
        data.append({"image": img_path})
    
    # 保存到文件
    json_path = output_dir / "data.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 50)
        print("✅ 已生成 data.json 文件")
        print(f"  文件路径: {json_path}")
        print(f"  包含 {len(data)} 张图片的记录")
        
        # 显示前3条记录示例
        print("\n📝 data.json 内容示例 (前3条):")
        for i, item in enumerate(data[:3]):
            print(f"  {i+1}. {item['image']}")
        if len(data) > 3:
            print(f"  ... 还有 {len(data) - 3} 条记录")
            
        # 提示如何使用
        print("\n💡 使用方法:")
        print(f"  1. 在Label Studio中创建新项目")
        print(f"  2. 导入数据时选择 'JSON' 格式")
        print(f"  3. 上传 {json_path} 文件")
        print(f"  4. 图片文件在 {output_dir} 目录下")
        print(f"  5. 注意: 确保Label Studio有权限访问这些图片的绝对路径")
        
        # 创建方便导入的说明文件
        readme_content = f"""# 视频帧提取结果

## 文件说明
- `data.json`: Label Studio导入文件
- `*.{image_format}`: 提取的视频帧图片

## 导入Label Studio
1. 创建新项目
2. 导入数据时选择 'JSON' 格式
3. 上传 data.json 文件
4. 确保Label Studio可以访问以下路径:
   {output_dir}

## 统计信息
- 总帧数: {len(data)}
- 输出格式: {image_format.upper()}
- 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        readme_path = output_dir / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"\n📄 已生成说明文件: {readme_path}")
        
    except Exception as e:
        print(f"错误: 生成 data.json 失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='从视频中提取帧图片，并生成Label Studio导入文件')
    parser.add_argument('video_path', help='视频文件路径')
    parser.add_argument('-o', '--output', default=None, 
                       help='输出文件夹路径 (默认: ./runs/时间戳)')
    parser.add_argument('-i', '--interval', type=int, default=1,
                       help='帧提取间隔，1为每帧都提取 (默认: 1)')
    parser.add_argument('-f', '--format', choices=['jpg', 'png'], default='jpg',
                       help='输出图片格式 (默认: jpg)')
    parser.add_argument('-q', '--quality', type=int, default=95,
                       help='图片质量，仅对jpg有效，范围1-100 (默认: 95)')
    
    args = parser.parse_args()
    
    # 参数验证
    if args.interval < 1:
        print("错误: 提取间隔必须 >= 1")
        return
    
    if args.quality < 1 or args.quality > 100:
        print("错误: 图片质量必须在 1-100 之间")
        return
    
    # 执行提取
    extract_frames(
        video_path=args.video_path,
        output_dir=args.output,
        frame_interval=args.interval,
        image_format=args.format,
        quality=args.quality
    )

if __name__ == "__main__":
    main()