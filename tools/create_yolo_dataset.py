# create_yolo_subset_simple.py
"""
YOLO数据集子集创建工具 - 极简版
只需指定数据集根目录，自动识别 images/ 和 labels/ 子目录
输出按子集名称组织在 dataset_root 下的子目录中
"""

import os
import random
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
from datetime import datetime


class YOLOSubsetCreator:
    """YOLO数据集子集创建器 - 极简版"""
    
    def __init__(
        self,
        dataset_root: str,
        subset_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        初始化
        
        Args:
            dataset_root: 数据集根目录 (包含 images/ 和 labels/ 子目录)
            subset_name: 子集名称 (None则自动生成时间戳)
            output_dir: 输出目录 (None则自动创建在 dataset_root/subsets/subset_name/)
            random_seed: 随机种子
        """
        self.dataset_root = Path(dataset_root)
        self.image_dir = self.dataset_root / "images"
        self.label_dir = self.dataset_root / "labels"
        self.random_seed = random_seed
        
        # 生成子集名称
        if subset_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.subset_name = f"subset_{timestamp}"
        else:
            self.subset_name = subset_name
        
        # 设置输出目录
        if output_dir is None:
            self.output_dir = self.dataset_root / "subsets" / self.subset_name
        else:
            self.output_dir = Path(output_dir)
        
        # 支持的图片格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 检查目录
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"数据集根目录不存在: {self.dataset_root}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"标签目录不存在: {self.label_dir}")
        
        print(f"✅ 数据集根目录: {self.dataset_root}")
        print(f"✅ 子集名称: {self.subset_name}")
        print(f"✅ 输出目录: {self.output_dir}")
        print(f"✅ 图片目录: {self.image_dir}")
        print(f"✅ 标签目录: {self.label_dir}")
    
    def load_all_data(self) -> Dict[str, List[int]]:
        """
        加载所有数据
        
        Returns:
            dict: {图片名: [类别列表]}
        """
        data = {}
        
        # 遍历所有标签文件
        label_files = list(self.label_dir.glob("*.txt"))
        print(f"📖 找到 {len(label_files)} 个标签文件")
        
        for label_file in label_files:
            image_name = label_file.stem
            
            # 检查是否存在对应的图片
            has_image = False
            for ext in self.image_extensions:
                if (self.image_dir / f"{image_name}{ext}").exists():
                    has_image = True
                    break
            
            if not has_image:
                print(f"⚠️ 跳过 {image_name}：找不到图片文件")
                continue
            
            # 读取标签
            classes = []
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if parts:
                                classes.append(int(parts[0]))
            except Exception as e:
                print(f"⚠️ 读取标签失败 {label_file}: {e}")
                continue
            
            if classes:
                data[image_name] = classes
        
        print(f"✅ 加载完成: {len(data)} 个有效数据")
        return data
    
    def get_class_distribution(self, data: Dict[str, List[int]]) -> Dict[int, int]:
        """统计类别分布"""
        counts = defaultdict(int)
        for classes in data.values():
            for cls in classes:
                counts[cls] += 1
        return dict(counts)
    
    def filter_by_classes(
        self,
        data: Dict[str, List[int]],
        target_classes: Optional[List[int]] = None,
        include_all: bool = True
    ) -> Dict[str, List[int]]:
        """
        按类别筛选
        
        Args:
            data: 输入数据
            target_classes: 目标类别列表
            include_all: True=必须包含所有，False=包含任意
            
        Returns:
            筛选后的数据
        """
        if target_classes is None:
            return data
        
        target_set = set(target_classes)
        filtered = {}
        
        for image_name, classes in data.items():
            class_set = set(classes)
            
            if include_all:
                if target_set.issubset(class_set):
                    filtered[image_name] = classes
            else:
                if class_set.intersection(target_set):
                    filtered[image_name] = classes
        
        print(f"🎯 类别筛选完成: {len(filtered)} 个数据")
        return filtered
    
    def sample_data(
        self,
        data: Dict[str, List[int]],
        max_count: Optional[int] = None,
        random_sample: bool = False
    ) -> Dict[str, List[int]]:
        """
        采样数据
        
        Args:
            data: 输入数据
            max_count: 最大数量
            random_sample: 是否随机采样
            
        Returns:
            采样后的数据
        """
        if max_count is None or len(data) <= max_count:
            return data
        
        items = list(data.items())
        
        if random_sample:
            random.seed(self.random_seed)
            items = random.sample(items, max_count)
        else:
            items = items[:max_count]
        
        print(f"📊 采样完成: {len(items)} 个数据")
        return dict(items)
    
    def split_dataset(
        self,
        data: Dict[str, List[int]],
        val_ratio: float = 0.2,
        random_split: bool = True
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """
        分割数据集
        
        Args:
            data: 所有数据
            val_ratio: 验证集比例
            random_split: 是否随机分割
            
        Returns:
            (train_data, val_data)
        """
        items = list(data.items())
        
        if random_split:
            random.seed(self.random_seed)
            random.shuffle(items)
        
        split_idx = int(len(items) * (1 - val_ratio))
        train_items = items[:split_idx]
        val_items = items[split_idx:]
        
        print(f"📊 数据分割完成:")
        print(f"   训练集: {len(train_items)} 张")
        print(f"   验证集: {len(val_items)} 张")
        
        return dict(train_items), dict(val_items)
    
    def create_list_file(
        self,
        data: Dict[str, List[int]],
        output_file: str
    ):
        """
        创建列表文件
        
        Args:
            data: 数据
            output_file: 输出文件路径
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建完整路径
        image_paths = []
        missing = []
        
        for image_name in data.keys():
            found = False
            for ext in self.image_extensions:
                img_path = self.image_dir / f"{image_name}{ext}"
                if img_path.exists():
                    image_paths.append(str(img_path.absolute()))
                    found = True
                    break
            
            if not found:
                missing.append(image_name)
        
        if missing:
            print(f"⚠️ 警告: {len(missing)} 张图片找不到，已跳过")
            if len(missing) <= 5:
                for name in missing:
                    print(f"   - {name}")
        
        # 写入文件
        with open(output_path, 'w') as f:
            for img_path in sorted(image_paths):
                f.write(f"{img_path}\n")
        
        print(f"✅ 创建列表文件: {output_path}")
        print(f"   包含 {len(image_paths)} 张图片")
    
    def create_data_yaml(
        self,
        train_file: str,
        val_file: str,
        output_file: str,
        nc: int,
        names: List[str]
    ):
        """
        创建 data.yaml 配置文件
        
        Args:
            train_file: 训练列表文件名
            val_file: 验证列表文件名
            output_file: 输出文件路径
            nc: 类别数量
            names: 类别名称列表
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用相对路径或绝对路径
        # 推荐使用相对路径，方便移动
        yaml_content = f"""# YOLO数据集配置文件
# 子集名称: {self.subset_name}
# 创建时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# 数据集根目录（用于解析相对路径）
path: {self.dataset_root.absolute()}

# 训练和验证列表文件（相对于path）
train: subsets/{self.subset_name}/{train_file}
val: subsets/{self.subset_name}/{val_file}

# 类别信息
nc: {nc}
names: {names}
"""
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ 创建配置文件: {output_path}")
    
    def create_summary(
        self,
        all_data: Dict,
        train_data: Dict,
        val_data: Dict,
        output_file: str
    ):
        """
        创建摘要报告
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        def count_classes(data):
            counts = defaultdict(int)
            for classes in data.values():
                for cls in classes:
                    counts[cls] += 1
            return dict(counts)
        
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("YOLO数据集摘要报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"子集名称: {self.subset_name}\n")
            f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集根目录: {self.dataset_root}\n")
            f.write(f"图片目录: {self.image_dir}\n")
            f.write(f"标签目录: {self.label_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            
            f.write(f"总数据量: {len(all_data)} 张\n\n")
            
            f.write("全部数据类别分布:\n")
            for cls, count in sorted(count_classes(all_data).items()):
                f.write(f"  - 类别 {cls}: {count} 个标注\n")
            
            f.write("\n" + "-" * 60 + "\n")
            
            f.write("训练集统计:\n")
            f.write(f"  - 数量: {len(train_data)} 张\n")
            for cls, count in sorted(count_classes(train_data).items()):
                f.write(f"  - 类别 {cls}: {count} 个标注\n")
            
            f.write("\n验证集统计:\n")
            f.write(f"  - 数量: {len(val_data)} 张\n")
            for cls, count in sorted(count_classes(val_data).items()):
                f.write(f"  - 类别 {cls}: {count} 个标注\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"✅ 创建摘要报告: {output_path}")
    
    def save_subset_info(self, args):
        """保存子集创建参数"""
        info_file = self.output_dir / "subset_info.txt"
        with open(info_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("子集创建参数\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"子集名称: {self.subset_name}\n")
            f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集根目录: {self.dataset_root}\n\n")
            
            for key, value in vars(args).items():
                if key not in ['dataset', 'output']:
                    f.write(f"{key}: {value}\n")
        
        print(f"✅ 保存子集信息: {info_file}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO数据集子集创建工具 - 极简版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础用法：自动创建时间戳子集
  python create_yolo_subset_simple.py --dataset /data/dataset

  # 指定子集名称，创建在 /data/dataset/subsets/cat_dog/
  python create_yolo_subset_simple.py \\
      --dataset /data/dataset \\
      --name cat_dog_subset

  # 按类别筛选
  python create_yolo_subset_simple.py \\
      --dataset /data/dataset \\
      --name cat_dog \\
      --classes 0 1 \\
      --max_count 2000

  # 随机采样并指定类别名称
  python create_yolo_subset_simple.py \\
      --dataset /data/dataset \\
      --name random_1000 \\
      --max_count 1000 \\
      --val_ratio 0.2 \\
      --random \\
      --nc 2 \\
      --names cat dog
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="数据集根目录 (包含 images/ 和 labels/ 子目录)"
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="子集名称 (默认: 自动生成时间戳)"
    )
    
    # 筛选参数
    parser.add_argument(
        "--classes", "-c",
        type=int,
        nargs="+",
        help="要筛选的类别ID (如: 0 1 2)"
    )
    parser.add_argument(
        "--include_all",
        action="store_true",
        help="必须包含所有指定类别 (默认: 包含任意)"
    )
    parser.add_argument(
        "--max_count", "-m",
        type=int,
        help="最大图片数量"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="随机采样 (默认: 顺序采样)"
    )
    
    # 分割参数
    parser.add_argument(
        "--val_ratio", "-r",
        type=float,
        default=0.2,
        help="验证集比例 (默认: 0.2)"
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="不分割数据集，只生成训练集"
    )
    parser.add_argument(
        "--no_random_split",
        action="store_true",
        help="不随机分割 (默认: 随机分割)"
    )
    
    # 类别名称
    parser.add_argument(
        "--nc",
        type=int,
        help="类别数量 (用于生成 data.yaml)"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="类别名称列表 (如: cat dog person)"
    )
    
    # 其他
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建器（输出目录自动生成在 dataset/subsets/name/）
        creator = YOLOSubsetCreator(
            dataset_root=args.dataset,
            subset_name=args.name,
            output_dir=None,  # 自动生成
            random_seed=args.seed
        )
        
        # 创建输出目录
        creator.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("开始创建YOLO数据集子集")
        print("=" * 60)
        
        # 步骤1: 加载所有数据
        all_data = creator.load_all_data()
        
        if not all_data:
            print("❌ 错误：没有找到任何有效数据")
            return 1
        
        # 显示类别分布
        class_dist = creator.get_class_distribution(all_data)
        print("\n📊 数据类别分布:")
        for cls, count in sorted(class_dist.items()):
            print(f"   类别 {cls}: {count} 个标注")
        
        # 步骤2: 类别筛选
        if args.classes:
            print(f"\n🎯 按类别筛选: {args.classes}")
            all_data = creator.filter_by_classes(
                all_data,
                target_classes=args.classes,
                include_all=args.include_all
            )
            
            if not all_data:
                print("❌ 错误：没有符合条件的数据")
                return 1
        
        # 步骤3: 数量限制
        if args.max_count:
            print(f"\n📊 数量限制: {args.max_count}")
            all_data = creator.sample_data(
                all_data,
                max_count=args.max_count,
                random_sample=args.random
            )
        
        # 步骤4: 分割数据集
        if args.no_split:
            train_data = all_data
            val_data = {}
            print("\n📊 不分割数据集，全部作为训练集")
        else:
            print(f"\n📊 分割数据集 (验证集比例: {args.val_ratio})")
            train_data, val_data = creator.split_dataset(
                all_data,
                val_ratio=args.val_ratio,
                random_split=not args.no_random_split
            )
        
        # 步骤5: 生成列表文件
        print("\n📝 生成列表文件...")
        
        train_file = f"{creator.subset_name}_train.txt"
        val_file = f"{creator.subset_name}_val.txt"
        
        creator.create_list_file(train_data, creator.output_dir / train_file)
        
        if not args.no_split and val_data:
            creator.create_list_file(val_data, creator.output_dir / val_file)
        
        # 步骤6: 生成 data.yaml
        print("\n📄 生成配置文件...")
        
        # 确定类别数量
        if args.nc:
            nc = args.nc
        else:
            # 从数据中自动推断
            all_classes = set()
            for classes in all_data.values():
                all_classes.update(classes)
            nc = len(all_classes)
            print(f"  自动检测到 {nc} 个类别")
        
        # 确定类别名称
        if args.names:
            names = args.names
        else:
            names = [f"class_{i}" for i in range(nc)]
            print(f"  使用默认类别名称: {names}")
        
        yaml_file = f"{creator.subset_name}_data.yaml"
        creator.create_data_yaml(
            train_file=train_file,
            val_file=val_file if not args.no_split else "",
            output_file=creator.output_dir / yaml_file,
            nc=nc,
            names=names
        )
        
        # 步骤7: 生成摘要报告
        print("\n📈 生成摘要报告...")
        summary_file = f"{creator.subset_name}_summary.txt"
        creator.create_summary(
            all_data=all_data,
            train_data=train_data,
            val_data=val_data,
            output_file=creator.output_dir / summary_file
        )
        
        # 步骤8: 保存子集信息
        creator.save_subset_info(args)
        
        # 完成
        print("\n" + "=" * 60)
        print("✅ 完成！")
        print("=" * 60)
        print(f"📁 输出目录: {creator.output_dir}")
        print(f"📋 训练集: {len(train_data)} 张")
        if val_data:
            print(f"📋 验证集: {len(val_data)} 张")
        print(f"📄 配置文件: {creator.output_dir / yaml_file}")
        print(f"📊 摘要报告: {creator.output_dir / summary_file}")
        print("\n💡 训练命令:")
        print(f"   from ultralytics import YOLO")
        print(f"   model = YOLO('yolo26n.pt')")
        print(f"   model.train(data='{creator.output_dir / yaml_file}', epochs=100)")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())