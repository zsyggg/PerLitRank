#!/bin/bash

# 脚本说明: 为 MedCorpus 和 LitSearch 数据集并行运行重排阶段 (完整运行)。
# 每个数据集的重排任务将在指定的GPU上运行。
# 将使用100和300字符长度、不包含查询 (_nq) 的叙述。 (200字符的已单独处理)
# 对每种叙述长度，将使用 Jina 和 GTE 两种重排器。
# 输出的每个查询的文档数将由 utils.py 中的 config.final_top_k 的默认值决定（通常是10）。

# --- 配置 ---
DATASETS_TO_PROCESS="MedCorpus LitSearch"
RERANKER_TYPES_TO_RUN="jina gte" # 要运行的重排器类型
NARRATIVE_LENGTHS_TO_RUN="100 300" # 要使用的叙述长度 (移除了200)
FINAL_TOP_K_SETTING="" # 设置为空字符串以使用utils.py中的默认final_top_k (通常是10)
                       # 如果要指定，例如100，则设置为: FINAL_TOP_K_SETTING=100

# 用于重排的GPU ID。
GPU_IDS=(0 1) # MedCorpus 在 GPU_IDS[0], LitSearch 在 GPU_IDS[1]

BASE_RERANK_ARGS=""

# --- 脚本逻辑 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR_BASE="./logs_full_run_all_configs" 
mkdir -p "$LOG_DIR_BASE"

K_SUFFIX_LOG=""
if [ -n "$FINAL_TOP_K_SETTING" ]; then
    K_SUFFIX_LOG="_top${FINAL_TOP_K_SETTING}"
fi
MAIN_LOG_FILE="$LOG_DIR_BASE/03b_rerank_full_L100_L300_nq${K_SUFFIX_LOG}_${TIMESTAMP}.log"

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

rerank_dataset_config() {
    local DATASET_NAME="$1"
    local ASSIGNED_GPU_ID="$2"
    local TARGET_LENGTH="$3"
    local RERANKER_TYPE="$4"
    local CURRENT_TIMESTAMP="$5"
    local LOG_DIR_CURRENT="$6"

    echo "--- 开始为数据集 $DATASET_NAME 在 GPU $ASSIGNED_GPU_ID 上进行完整重排 ---"
    echo "    叙述: L${TARGET_LENGTH}_nq, 重排器: $RERANKER_TYPE ---"

    RETRIEVED_FILE="./results/$DATASET_NAME/retrieved.jsonl"
    if [ ! -f "$RETRIEVED_FILE" ]; then
        echo "严重错误: $DATASET_NAME - 未找到初始检索文件 ($RETRIEVED_FILE)。跳过。"
        return 1
    fi

    NARRATIVE_SUFFIX="_nq"
    NARRATIVE_FILE="./results/$DATASET_NAME/personalized_queries_L${TARGET_LENGTH}${NARRATIVE_SUFFIX}.jsonl"
    
    if [ ! -f "$NARRATIVE_FILE" ]; then
        echo "警告: $DATASET_NAME - 未找到叙述文件 $NARRATIVE_FILE。无法进行此配置的重排。"
        return 1
    fi
    
    K_SUFFIX_CMD_PARAM=""
    if [ -n "$FINAL_TOP_K_SETTING" ]; then
        K_SUFFIX_CMD_PARAM="--final_top_k $FINAL_TOP_K_SETTING"
    fi

    rerank_log_file="$LOG_DIR_CURRENT/rerank_full_${DATASET_NAME}_${RERANKER_TYPE}_L${TARGET_LENGTH}${NARRATIVE_SUFFIX}${K_SUFFIX_LOG}_${CURRENT_TIMESTAMP}.log"
    
    CMD_RERANK="python run.py --dataset_name $DATASET_NAME \
                    --mode rerank \
                    --reranker_type $RERANKER_TYPE \
                    --personalized_text_target_length $TARGET_LENGTH \
                    --narrative_query_mode_suffix_for_rerank_input \"$NARRATIVE_SUFFIX\" \
                    --use_personalized_features \
                    $K_SUFFIX_CMD_PARAM \
                    --gpu_id $ASSIGNED_GPU_ID \
                    $BASE_RERANK_ARGS" # 不传递 --test_query_limit

    run_and_log_sequential "$DATASET_NAME 完整重排 (L${TARGET_LENGTH}${NARRATIVE_SUFFIX}, ${RERANKER_TYPE}, GPU $ASSIGNED_GPU_ID)" "$CMD_RERANK" "$rerank_log_file"
    
    EXPECTED_K_SUFFIX_FILENAME_PART=""
    if [ -n "$FINAL_TOP_K_SETTING" ]; then
        EXPECTED_K_SUFFIX_FILENAME_PART="_top${FINAL_TOP_K_SETTING}"
    elif [ -z "$FINAL_TOP_K_SETTING" ] && [ "$(python -c 'from utils import get_config; print(get_config().final_top_k)')" == "10" ]; then # 检查默认是否为10
         EXPECTED_K_SUFFIX_FILENAME_PART=""
    fi

    EXPECTED_OUTPUT="./results/$DATASET_NAME/ranked_${RERANKER_TYPE}_personalized${NARRATIVE_SUFFIX}_L${TARGET_LENGTH}${EXPECTED_K_SUFFIX_FILENAME_PART}.jsonl"
    
    if [ -f "$EXPECTED_OUTPUT" ]; then
        echo "$DATASET_NAME: 使用 L${TARGET_LENGTH}${NARRATIVE_SUFFIX} 叙述和 ${RERANKER_TYPE} 重排器的结果已保存到: $EXPECTED_OUTPUT"
    else
        echo "警告: $DATASET_NAME - 未找到预期的重排输出文件: $EXPECTED_OUTPUT"
    fi
    
    echo "--- 数据集 $DATASET_NAME, L${TARGET_LENGTH}_nq, ${RERANKER_TYPE} 的重排任务完成 ---"
    return 0
}

echo ""
echo "=============================================================="
echo "为数据集 ${DATASETS_TO_PROCESS} 并行运行所有配置的完整重排"
echo "叙述长度: ${NARRATIVE_LENGTHS_TO_RUN}, 模式: _nq"
echo "重排器: ${RERANKER_TYPES_TO_RUN}"
if [ -n "$FINAL_TOP_K_SETTING" ]; then
    echo "每个查询保留 Top ${FINAL_TOP_K_SETTING} 个结果"
else
    echo "每个查询保留默认数量的结果 (通常是 Top 10)"
fi
echo "=============================================================="

pids=() 
gpu_idx_counter=0

for DATASET_NAME_ITER in $DATASETS_TO_PROCESS; do
    ASSIGNED_GPU_ID_ITER=${GPU_IDS[$gpu_idx_counter % ${#GPU_IDS[@]}]}
    echo ""
    echo "--- 为数据集 $DATASET_NAME_ITER 在 GPU $ASSIGNED_GPU_ID_ITER 上启动所有重排配置任务 (后台) ---"
    
    # 为单个数据集启动一个后台进程，该进程内部会串行处理不同的长度和重排器组合
    (
      CURRENT_LOG_SUBDIR="${LOG_DIR_BASE}/${DATASET_NAME_ITER}"
      mkdir -p "$CURRENT_LOG_SUBDIR"
      for TARGET_LEN_ITER in $NARRATIVE_LENGTHS_TO_RUN; do # 现在只包含 100 和 300
        for RERANKER_ITER in $RERANKER_TYPES_TO_RUN; do
          rerank_dataset_config "$DATASET_NAME_ITER" "$ASSIGNED_GPU_ID_ITER" "$TARGET_LEN_ITER" "$RERANKER_ITER" "$TIMESTAMP" "$CURRENT_LOG_SUBDIR"
        done
      done
    ) &
    pids+=($!) 

    gpu_idx_counter=$((gpu_idx_counter + 1))
done

echo "所有数据集的重排任务已在后台启动。等待所有任务完成..."
all_jobs_successful=true
for pid_item in "${pids[@]}"; do
    wait $pid_item
    if [ $? -ne 0 ]; then
        echo "警告: 一个后台重排作业 (PID $pid_item) 退出时带有错误。请检查特定日志。"
        all_jobs_successful=false
    fi
done

if $all_jobs_successful; then
    echo "所有后台重排作业已成功完成。"
else
    echo "部分后台重排作业失败。请检查日志。"
fi
echo ""
echo "=============================================================="
echo "所有数据集的所有重排配置已尝试执行。"
echo "=============================================================="
echo ""

echo "所有处理完成。"
echo "主日志: $MAIN_LOG_FILE"
echo "各个命令的日志位于: $LOG_DIR_BASE/<dataset_name>/"
