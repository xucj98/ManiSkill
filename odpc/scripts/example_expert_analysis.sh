#!/bin/bash

# 专家策略分析示例脚本

echo "=== 专家策略局限性分析示例 ==="
echo ""

# 设置参数
# 用于分析 PegInsertionSide-v2 环境中专家策略在不同 clearance 值下的表现
# CONFIG_FILE="configs/demo/PegIns_EeDeltaPose.yaml"
# DIFFICULTY_KEY="clearance"
# DIFFICULTY_VALUES=(0.0015 0.002 0.0025 0.003 0.0035 0.004 0.005 0.007 0.010)  # 测试更多值
# NUM_TRIALS=100
# OUTPUT_DIR="analysis_outputs/peg_insertion_expert_analysis"

CONFIG_FILE="configs/demo/StackCuboid_EeDeltaPose.yaml"
DIFFICULTY_KEY="cuboid_half_size"
DIFFICULTY_VALUES=(0.001 0.002 0.003 0.004 0.005 0.010 0.015 0.020)  # 测试更多值
NUM_TRIALS=100
OUTPUT_DIR="analysis_outputs/stack_cuboid_expert_analysis"

echo "分析参数:"
echo "  配置文件: $CONFIG_FILE"
echo "  难度参数: $DIFFICULTY_KEY"
echo "  难度值: ${DIFFICULTY_VALUES[*]}"
echo "  测试次数: $NUM_TRIALS"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 运行分析
echo "开始分析..."
python scripts/analyze_expert_demo.py \
    --config "$CONFIG_FILE" \
    --difficulty_key "$DIFFICULTY_KEY" \
    --difficulty_values "${DIFFICULTY_VALUES[@]}" \
    --num_trials "$NUM_TRIALS" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "分析完成！"
echo "结果文件:"
echo "  JSON数据: $OUTPUT_DIR/expert_analysis_${DIFFICULTY_KEY}.json"
echo "  成功率曲线: $OUTPUT_DIR/success_rate_curve_${DIFFICULTY_KEY}.png" 