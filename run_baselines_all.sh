#!/bin/bash
# run_baselines_all.sh - 运行所有基线方法针对指定数据集 (修正评估命令)
# - 通过命令行参数显式传递模型路径
# - 移除连续性和Parquet逻辑

# --- Configuration ---
DATASETS_TO_RUN="LitSearch" # 示例: 仅运行 LitSearch
BASELINES_TO_RUN="rpmn htps qer conqrr"
BASE_DATA_DIR="/workspace/PerMed/data"
BASE_RESULTS_DIR="/workspace/PerMed/baselines/results"
ENCODER_MODEL_PATH="/workspace/.cache/modelscope/hub/iic/gte_sentence-embedding_multilingual-base"
LLM_MODEL_PATH="/workspace/Qwen3-4B"
COMMON_ARGS="--batch_size 8 --device cuda:1" # 使用 GPU 0

# --- Script Logic ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${BASE_RESULTS_DIR}/_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG_FILE="$LOG_DIR/baselines_run_$TIMESTAMP.log"

echo "Main log file: $MAIN_LOG_FILE"
exec > >(tee -a "$MAIN_LOG_FILE") 2>&1

echo "Starting baseline runs at $(date)"
echo "Datasets: $DATASETS_TO_RUN"; echo "Baselines: $BASELINES_TO_RUN"
echo "Encoder Model: $ENCODER_MODEL_PATH"; echo "LLM Model: $LLM_MODEL_PATH"
echo "============================================="

# 循环处理每个数据集
for DATASET_NAME in $DATASETS_TO_RUN; do
    echo ""; echo "+++++++ Processing Dataset: $DATASET_NAME +++++++"

    CMD="python run_baselines.py \
        --dataset_name \"$DATASET_NAME\" \
        --data_dir \"$BASE_DATA_DIR\" \
        --results_dir \"$BASE_RESULTS_DIR\" \
        --encoder_model_path \"$ENCODER_MODEL_PATH\" \
        --llm_model_path \"$LLM_MODEL_PATH\" \
        --baselines $BASELINES_TO_RUN \
        $COMMON_ARGS"

    echo "Executing command for $DATASET_NAME:"; echo "$CMD"
    eval $CMD; EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then echo "ERROR: run_baselines.py failed for $DATASET_NAME"; else echo "Finished running baselines for $DATASET_NAME."; fi
    echo "+++++++ Finished Dataset: $DATASET_NAME +++++++"; echo ""
done

echo "============================================="
echo "All baseline runs initiated. Starting evaluation..."
echo "============================================="

# --- 评估所有结果 ---
for DATASET_NAME in $DATASETS_TO_RUN; do
    echo ""; echo "+++++++ Evaluating Dataset: $DATASET_NAME +++++++"
    GT_FILE="$BASE_DATA_DIR/$DATASET_NAME/query_to_texts.jsonl"
    MAIN_RESULTS_BASE=$(dirname "$BASE_RESULTS_DIR")
    MAIN_RESULTS_DIR="$MAIN_RESULTS_BASE/results/$DATASET_NAME"
    UNRERANKED_FILE="$MAIN_RESULTS_DIR/retrieved.jsonl"

    if [ ! -f "$GT_FILE" ]; then echo "WARNING: GT file not found: $GT_FILE. Skipping evaluation."; continue; fi
    if [ ! -f "$UNRERANKED_FILE" ]; then echo "WARNING: Unreranked file not found: $UNRERANKED_FILE. P-imp cannot be calculated."; fi

    for baseline in $BASELINES_TO_RUN; do
        PRED_FILE="$BASE_RESULTS_DIR/$DATASET_NAME/${baseline}_results.jsonl"
        if [ -f "$PRED_FILE" ]; then
            echo "--- Evaluating $baseline for $DATASET_NAME ---"
            # **修正**: 使用 evaluate.py 正确的参数
            EVAL_CMD="python evaluate.py \
                --dataset_name \"$DATASET_NAME\" \
                --gt_file \"$GT_FILE\" \
                --rerank_pred_file \"$PRED_FILE\""
            if [ -f "$UNRERANKED_FILE" ]; then EVAL_CMD="$EVAL_CMD --unrerank_pred_file \"$UNRERANKED_FILE\""; fi
            echo "Executing: $EVAL_CMD"; eval $EVAL_CMD
            echo "--- Finished evaluating $baseline for $DATASET_NAME ---"
        else
            echo "WARNING: Prediction file not found for $baseline on $DATASET_NAME: $PRED_FILE. Skipping evaluation."
        fi; echo ""
    done
    echo "+++++++ Finished Evaluating Dataset: $DATASET_NAME +++++++"; echo ""
done

echo "============================================="
echo "All baseline runs and evaluations finished at $(date)."
echo "Main log file: $MAIN_LOG_FILE"
echo "============================================="
