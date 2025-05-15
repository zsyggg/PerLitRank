#!/bin/bash
# run_custom_ablation_L100_jina.sh - 为 MedCorpus 和 LitSearch 运行消融实验 (Jina, L100) 并评估

# --- 配置 ---
DATASETS_TO_RUN="MedCorpus LitSearch"
TARGET_RERANKER_TYPE="jina"
TARGET_NARRATIVE_LENGTH=100
TOP_K_FINAL=10 # 重排后保留的文档数，与评估时的K值对应

# 基本路径 (请根据您的环境核实)
BASE_PROJECT_DIR="/workspace/PerMed" # 项目根目录
BASE_DATA_DIR="${BASE_PROJECT_DIR}/data"
BASE_RESULTS_DIR="${BASE_PROJECT_DIR}/results" # 主结果目录，包含 cognitive_features 和 retrieved
ABLATION_SPECIFIC_OUTPUT_BASE_DIR="${BASE_PROJECT_DIR}/results_ablation" # 存放本次消融实验产出文件的特定目录

# memory_ablation.py 的通用参数
# 注意: --gpu_id 用于 memory_ablation.py 中的重排器。
# PersonalizedGenerator 会使用 config.llm_device。
# 移除了 --llm_gpu_id 因为 memory_ablation.py 当前的 argparse 不支持它。
# memory_ablation.py 内部的 PersonalizedGenerator 将依赖全局 config 来设置 LLM 设备。
COMMON_ABLATION_PY_ARGS="--reranker_type ${TARGET_RERANKER_TYPE} \
                         --personalized_text_target_length ${TARGET_NARRATIVE_LENGTH} \
                         --batch_size 8 \
                         --max_length 512 \
                         --initial_top_k 50 \
                         --top_k ${TOP_K_FINAL} \
                         --gpu_id 0"

# --- 脚本逻辑 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG_DIR="${ABLATION_SPECIFIC_OUTPUT_BASE_DIR}/_logs_L${TARGET_NARRATIVE_LENGTH}_${TARGET_RERANKER_TYPE}"
mkdir -p "$MAIN_LOG_DIR"
MAIN_ABLATION_LOG_FILE="$MAIN_LOG_DIR/run_ablation_main_${TIMESTAMP}.log"

echo "消融实验主日志文件: $MAIN_ABLATION_LOG_FILE"
exec > >(tee -a "$MAIN_ABLATION_LOG_FILE") 2>&1 # 重定向主脚本的 stdout/stderr

# 函数：运行命令并记录其特定输出
run_command_with_log() {
    local cmd_description="$1"
    local command_to_run="$2"
    local specific_log_path="$3"
    
    echo "---------------------------------------------------------------------"
    echo "开始: $cmd_description"
    echo "执行命令: $command_to_run"
    echo "此运行的特定日志: $specific_log_path"
    echo "---------------------------------------------------------------------"
    
    eval $command_to_run > "$specific_log_path" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "完成: $cmd_description"
    else
        echo "错误: $cmd_description 执行失败。请检查日志: $specific_log_path"
    fi
    echo "---------------------------------------------------------------------"
}

# 主循环：遍历数据集
for DATASET_NAME in $DATASETS_TO_RUN; do
    echo ""
    echo "======= 开始处理数据集: $DATASET_NAME (重排器: ${TARGET_RERANKER_TYPE}, 叙述长度: L${TARGET_NARRATIVE_LENGTH}) ======="

    # 为当前数据集和配置创建特定的输出和日志子目录
    CURRENT_ABLATION_OUTPUT_DIR="${ABLATION_SPECIFIC_OUTPUT_BASE_DIR}/${DATASET_NAME}/${TARGET_RERANKER_TYPE}_L${TARGET_NARRATIVE_LENGTH}"
    CURRENT_ABLATION_LOG_DIR="${MAIN_LOG_DIR}/${DATASET_NAME}"
    mkdir -p "$CURRENT_ABLATION_OUTPUT_DIR"
    mkdir -p "$CURRENT_ABLATION_LOG_DIR"

    # memory_ablation.py 所需的输入文件路径
    COGNITIVE_FEATURES_FILE="${BASE_RESULTS_DIR}/${DATASET_NAME}/cognitive_features_detailed.jsonl"
    RETRIEVED_RESULTS_FILE="${BASE_RESULTS_DIR}/${DATASET_NAME}/retrieved.jsonl"
    ORIGINAL_QUERIES_FILE="${BASE_DATA_DIR}/${DATASET_NAME}/queries.jsonl" # 用于连续性过滤

    # 检查必要的输入文件是否存在
    if [ ! -f "$COGNITIVE_FEATURES_FILE" ]; then
        echo "错误: 数据集 ${DATASET_NAME} 的认知特征文件 (${COGNITIVE_FEATURES_FILE}) 未找到。请先运行 'python run.py --mode extract_cognitive_features --dataset_name ${DATASET_NAME}'。跳过此数据集。"
        continue
    fi
    if [ ! -f "$RETRIEVED_RESULTS_FILE" ]; then
        echo "错误: 数据集 ${DATASET_NAME} 的检索结果文件 (${RETRIEVED_RESULTS_FILE}) 未找到。请先运行 'python run.py --mode retrieve --dataset_name ${DATASET_NAME}'。跳过此数据集。"
        continue
    fi
    if [ ! -f "$ORIGINAL_QUERIES_FILE" ]; then
        echo "警告: 数据集 ${DATASET_NAME} 的原始查询文件 (${ORIGINAL_QUERIES_FILE}) 未找到。memory_ablation.py 中的连续性过滤将始终评估所有查询（如果其内部默认处理是这样的话）。"
    fi

    # 定义消融选项 (仅排除单个组件)
    EXCLUDE_MEMORY_OPTIONS=("sequential" "working" "long")
    # 连续性过滤硬编码为 "all"
    fixed_continuity_filter="all"

    # 运行 memory_ablation.py
    for exclude_opt in "${EXCLUDE_MEMORY_OPTIONS[@]}"; do
        # mem_abl_suffix 现在只表示排除的组件
        mem_abl_suffix="no_${exclude_opt}"
        
        # 输出文件名格式: <reranker_type>_L<length>_<memory_ablation_suffix>_<continuity_filter_suffix>.jsonl
        # 例如: jina_L100_no_sequential_all.jsonl
        output_filename_for_ablation_run="${TARGET_RERANKER_TYPE}_L${TARGET_NARRATIVE_LENGTH}_${mem_abl_suffix}_${fixed_continuity_filter}.jsonl"
        output_path_for_this_ablation_run="${CURRENT_ABLATION_OUTPUT_DIR}/${output_filename_for_ablation_run}"
        
        specific_cmd_log_file="$CURRENT_ABLATION_LOG_DIR/ablation_${output_filename_for_ablation_run%.jsonl}_${TIMESTAMP}.log"

        # 构建 memory_ablation.py 命令
        CMD_FOR_ABLATION_PY="python ${BASE_PROJECT_DIR}/memory_ablation.py \
            --dataset_name \"$DATASET_NAME\" \
            --exclude_memory \"$exclude_opt\" \
            --cognitive_features_input_path \"$COGNITIVE_FEATURES_FILE\" \
            --retrieved_results_input_path \"$RETRIEVED_RESULTS_FILE\" \
            --original_queries_path \"$ORIGINAL_QUERIES_FILE\" \
            --output_path \"$output_path_for_this_ablation_run\" \
            --continuity_filter \"${fixed_continuity_filter}\" \
            $COMMON_ABLATION_PY_ARGS" # 包含 reranker_type, personalized_text_target_length, gpu_id 等

        run_command_with_log "$DATASET_NAME | 消融: $exclude_opt | 连续性: ${fixed_continuity_filter} | 重排器: $TARGET_RERANKER_TYPE | 长度: L$TARGET_NARRATIVE_LENGTH" \
                             "$CMD_FOR_ABLATION_PY" \
                             "$specific_cmd_log_file"
    done 

    echo "======= 数据集 $DATASET_NAME 的 memory_ablation.py 运行完成 ======="
    echo ""

done # 结束数据集循环

echo "======= 所有指定数据集的消融实验和评估流程已完成 ======="
echo "主日志文件: $MAIN_ABLATION_LOG_FILE"
echo "各个命令的特定日志位于: $MAIN_LOG_DIR/<dataset_name>/"
