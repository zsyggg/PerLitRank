#!/bin/bash

# 脚本说明: 为 MedCorpus 和 LitSearch 数据集并行运行阶段2 (个性化叙述生成)。
# 此脚本为完整运行，生成100和300字符长度、不包含查询 (_nq) 的叙述。
# 假设 L200_nq 叙述已由之前的脚本生成。

# --- 配置 ---
DATASETS_TO_PROCESS="MedCorpus LitSearch"
TARGET_LENGTHS_TO_GENERATE="100 300" # 要生成的叙述长度

# 为每个数据集的阶段2分配GPU ID。
NARRATIVE_GPU_IDS=(0 1) # MedCorpus 在 GPU_IDS[0], LitSearch 在 GPU_IDS[1]

BASE_CMD_ARGS="--local_model_path /workspace/Qwen3-4B" # 请替换为您的LLM路径

# --- 脚本逻辑 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs_full_run_L100_L300" # 为这些长度创建新的日志目录
mkdir -p "$LOG_DIR"
MAIN_LOG_FILE="$LOG_DIR/02b_generate_narratives_full_L100_L300_nq_${TIMESTAMP}.log"

echo "此脚本的主日志文件: $MAIN_LOG_FILE"
exec > >(tee -a "$MAIN_LOG_FILE") 2>&1

run_and_log_sequential() {
    local cmd_desc="$1"
    local cmd_to_run="$2"
    local specific_log_file="$3"
    echo "---------------------------------------------------------------------"
    echo "开始 (串行): $cmd_desc"
    echo "执行: $cmd_to_run"
    echo "此命令的特定日志文件: $specific_log_file"
    echo "---------------------------------------------------------------------"
    eval $cmd_to_run > "$specific_log_file" 2>&1
    if [ $? -eq 0 ]; then
        echo "完成: $cmd_desc"
    else
        echo "错误: $cmd_desc 失败。请检查日志: $specific_log_file"
    fi
    echo "---------------------------------------------------------------------"
}

generate_narratives_for_dataset_length() {
    local DATASET_NAME="$1"
    local ASSIGNED_GPU_ID="$2"
    local TARGET_LENGTH="$3"
    local CURRENT_TIMESTAMP="$4"

    echo "--- 开始为数据集 $DATASET_NAME 在 GPU $ASSIGNED_GPU_ID 上生成 L${TARGET_LENGTH}_nq 叙述 (完整运行) ---"

    COGNITIVE_FEATURES_FILE="./results/$DATASET_NAME/cognitive_features_detailed.jsonl"
    if [ ! -f "$COGNITIVE_FEATURES_FILE" ]; then
        echo "严重错误: $DATASET_NAME - 未找到认知特征文件 ($COGNITIVE_FEATURES_FILE)。跳过。"
        return 1
    fi

    narrative_log_nq="$LOG_DIR/narrative_full_${DATASET_NAME}_L${TARGET_LENGTH}_nq_gpu${ASSIGNED_GPU_ID}_${CURRENT_TIMESTAMP}.log"
    CMD_NARRATIVE_NQ="python run.py --dataset_name $DATASET_NAME \
                        --mode generate_narratives \
                        --personalized_text_target_length $TARGET_LENGTH \
                        --no-include_query_in_narrative_prompt \
                        --gpu_id $ASSIGNED_GPU_ID \
                        --llm_gpu_id $ASSIGNED_GPU_ID \
                        $BASE_CMD_ARGS" # 不传递 --test_query_limit

    run_and_log_sequential "$DATASET_NAME 叙述生成 L${TARGET_LENGTH} (不含查询, 完整, GPU $ASSIGNED_GPU_ID)" "$CMD_NARRATIVE_NQ" "$narrative_log_nq"
    
    OUTPUT_FILE_NQ="./results/$DATASET_NAME/personalized_queries_L${TARGET_LENGTH}_nq.jsonl"
    if [ -f "$OUTPUT_FILE_NQ" ]; then
        echo ">>> $DATASET_NAME: L${TARGET_LENGTH}_nq 叙述已生成到 $OUTPUT_FILE_NQ <<<"
    else
        echo "警告: $DATASET_NAME - 未生成 L${TARGET_LENGTH}_nq 叙述文件 ($OUTPUT_FILE_NQ)。"
    fi
    echo ""
    echo "--- 数据集 $DATASET_NAME, L${TARGET_LENGTH}_nq 的叙述生成完成 ---"
    return 0
}

echo ""
echo "=============================================================="
echo "为数据集 ${DATASETS_TO_PROCESS} 并行生成 L100_nq 和 L300_nq 个性化叙述 (完整运行)"
echo "=============================================================="

pids=()
gpu_idx_counter=0

for DATASET_NAME_ITER in $DATASETS_TO_PROCESS; do
    ASSIGNED_GPU_ID_ITER=${NARRATIVE_GPU_IDS[$gpu_idx_counter % ${#NARRATIVE_GPU_IDS[@]}]}
    echo ""
    echo "--- 为数据集 $DATASET_NAME_ITER 在 GPU $ASSIGNED_GPU_ID_ITER 上启动 L100_nq 和 L300_nq 叙述生成任务 (后台) ---"
    
    # 为单个数据集启动一个后台进程，该进程内部会串行处理不同的长度
    (
      for TARGET_LEN_ITER in $TARGET_LENGTHS_TO_GENERATE; do
        generate_narratives_for_dataset_length "$DATASET_NAME_ITER" "$ASSIGNED_GPU_ID_ITER" "$TARGET_LEN_ITER" "$TIMESTAMP"
      done
    ) &
    pids+=($!)

    gpu_idx_counter=$((gpu_idx_counter + 1))
done

echo "等待所有数据集的 L100_nq 和 L300_nq 叙述生成后台作业完成..."
all_jobs_successful=true
for pid_item in "${pids[@]}"; do
    wait $pid_item
    if [ $? -ne 0 ]; then
        echo "警告: 一个叙述生成的后台作业 (PID $pid_item) 退出时带有错误。请检查特定日志。"
        all_jobs_successful=false
    fi
done

if $all_jobs_successful; then
    echo "所有数据集的 L100_nq 和 L300_nq 个性化叙述生成已成功完成。"
else
    echo "部分数据集的 L100_nq 和 L300_nq 作业失败。请检查日志。"
fi
echo ""
echo "=============================================================="
echo "所有数据集的 L100_nq 和 L300_nq 个性化叙述生成已尝试执行。"
echo "=============================================================="
echo ""
echo "所有处理完成。主日志: $MAIN_LOG_FILE。各个命令的日志位于: $LOG_DIR"
