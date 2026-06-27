#!/bin/bash
# YOLO26 测试启动脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║           YOLO26 模型测试框架 v1.0                      ║"
echo "║           Ultralytics YOLO26 Test Framework           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 显示使用说明
show_usage() {
  cat << EOF
使用方法: $0 [选项]

测试模式:
  -i, --input FILE      输入文件（图片或视频）
  -m, --mode MODE       测试模式: image, video, benchmark (默认: video)
  --model MODEL         YOLO模型 (默认: yolo26n.pt)
  --device DEVICE       设备: cpu, cuda, mps (默认: cpu)
  --conf THRESH         置信度阈值 (默认: 0.25)
  --eval                启用评估模式
  --display             实时显示
  --save                保存结果
  --output-dir DIR      输出目录 (默认: test_results)
  -h, --help            显示帮助

示例:
  # 测试图片
  $0 -i image.jpg -m image --display
  
  # 测试视频（评估模式）
  $0 -i video.mp4 -m video --eval --device cuda
  
  # 性能基准测试
  $0 -i video.mp4 -m benchmark --iterations 500
EOF
}

# 默认参数
INPUT=""
MODE="video"
MODEL="yolo26n.pt"
DEVICE="cpu"
CONF=0.25
EVAL=""
DISPLAY=""
SAVE=""
OUTPUT_DIR="test_results"
ITERATIONS=100
WARMUP=10
MAX_FRAMES=""

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    -i | --input)
      INPUT="$2"
      shift 2
      ;;
    -m | --mode)
      MODE="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --conf)
      CONF="$2"
      shift 2
      ;;
    --eval)
      EVAL="--eval"
      shift
      ;;
    --display)
      DISPLAY="--display"
      shift
      ;;
    --save)
      SAVE="--save"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="--max-frames $2"
      shift 2
      ;;
    -h | --help)
      show_usage
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      show_usage
      exit 1
      ;;
  esac
done

# 检查输入
if [ -z "$INPUT" ]; then
  echo -e "${RED}❌ 错误: 必须指定输入文件${NC}"
  show_usage
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo -e "${RED}❌ 错误: 文件不存在: $INPUT${NC}"
  exit 1
fi

# 显示配置
echo -e "${GREEN}📋 测试配置:${NC}"
echo "  模式: $MODE"
echo "  输入: $INPUT"
echo "  模型: $MODEL"
echo "  设备: $DEVICE"
echo "  置信度: $CONF"
echo "  评估模式: ${EVAL:-否}"
echo "  实时显示: ${DISPLAY:-否}"
echo "  保存结果: ${SAVE:-否}"
echo ""

# 运行测试
echo -e "${YELLOW}🚀 开始测试...${NC}\n"

python3 ultralytics/runners/test_yolo26.py \
  --input "$INPUT" \
  --mode "$MODE" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --conf "$CONF" \
  $EVAL \
  $DISPLAY \
  $SAVE \
  --output-dir "$OUTPUT_DIR" \
  $MAX_FRAMES \
  --iterations $ITERATIONS \
  --warm
