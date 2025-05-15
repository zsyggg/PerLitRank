#!/bin/bash

# 脚本说明: 为 MedCorpus 和 LitSearch 数据集并行运行阶段1 (认知特征提取)。
# 每个数据集的认知特征提取将在指定的GPU上运行。

# --- 配置 ---
DATASETS_TO_PROCESS="MedCorpus LitSearch"

# 为每个数据集的阶段1分配GPU ID。
# 确保这些GPU ID可用。如果GPU数量少于数据集数量，脚本会顺序执行或需要调整逻辑。
# 这里我们假设可以为每个数据集分配一个GPU并行运行，或者它们将按顺序在指定的GPU上运行。
# 为简单起见，如果GPU数量不足，此脚本会将任务顺序分配给可用的GPU。
STAGE1_GPU_IDS=(0 1) # MedCorpus on GPU_IDS[0], LitSearch on GPU_IDS[1]

BASE_CMD_ARGS="" # 例如: "--local_model_path /path/to/your/llm"

# --- 脚本逻辑 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs_parallel_test" # 与重排测试使用相同的日志目录
mkdir -p "$LOG_DIR"
MAIN_LOG_FILE="$LOG_DIR/01_extract_cog_features_parallel_${TIMESTAMP}.log"

echo "此脚本的主日志文件: $MAIN_LOG_FILE"
exec > >(tee -a "$MAIN_LOG_FILE") 2>&1

run_and_log_background() {
    local cmd_desc="$1"
    local cmd_to_run="$2"
    local specific_log_file="$3"
    echo "---------------------------------------------------------------------"
    echo "后台开始: $cmd_desc"
    echo "执行: $cmd_to_run"
    echo "此命令的特定日志文件: $specific_log_file"
    echo "---------------------------------------------------------------------"
    eval $cmd_to_run > "$specific_log_file" 2>&1 &
}

echo ""
echo "=============================================================="
echo "为数据集 ${DATASETS_TO_PROCESS} 并行提取认知特征"
echo "=============================================================="

pids=()
gpu_idx_counter=0

for DATASET_NAME in $DATASETS_TO_PROCESS; do
    ASSIGNED_GPU_ID=${STAGE1_GPU_IDS[$gpu_idx_counter % ${#STAGE1_GPU_IDS[@]}]}
    echo ""
    echo ">>> 阶段1: 为 $DATASET_NAME 在 GPU $ASSIGNED_GPU_ID 上进行认知特征提取 <<<"
    
    stage1_log="$LOG_DIR/stage1_cognitive_features_${DATASET_NAME}_gpu${ASSIGNED_GPU_ID}_${TIMESTAMP}.log"
    CMD_STAGE1="python run.py --dataset_name $DATASET_NAME \
                --mode extract_cognitive_features \
                --gpu_id $ASSIGNED_GPU_ID \
                $BASE_CMD_ARGS"

    run_and_log_background "$DATASET_NAME 阶段1 认知特征 (GPU $ASSIGNED_GPU_ID)" "$CMD_STAGE1" "$stage1_log"
    pids+=($!)
    
    gpu_idx_counter=$((gpu_idx_counter + 1))
done

echo "等待所有阶段1 (认知特征提取) 后台作业完成..."
all_stage1_successful=true
for pid in "${pids[@]}"; do
    wait $pid
    if [ $? -ne 0 ]; then
        echo "警告: 一个阶段1作业 (PID $pid) 退出时带有错误。请检查特定日志。"
        all_stage1_successful=false
    fi
done

if $all_stage1_successful; then
    echo "所有数据集的阶段1认知特征提取已成功完成。"
    for DATASET_NAME in $DATASETS_TO_PROCESS; do
        COGNITIVE_FEATURES_FILE="./results/$DATASET_NAME/cognitive_features_detailed.jsonl"
        if [ -f "$COGNITIVE_FEATURES_FILE" ]; then
            echo ">>> $DATASET_NAME 的特征位于 $COGNITIVE_FEATURES_FILE <<<"
        else
            echo "警告: 未找到 $DATASET_NAME 的阶段1输出 ($COGNITIVE_FEATURES_FILE)。"
        fi
    done
else
    echo "部分阶段1作业失败。请检查日志。"
fi
echo ""

echo "=============================================================="
echo "所有数据集的认知特征提取已尝试执行。"
echo "=============================================================="
echo ""

echo "所有处理完成。"
echo "主日志: $MAIN_LOG_FILE"
echo "各个命令的日志位于: $LOG_DIR"
