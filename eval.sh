#!/bin/bash

# 脚本说明: 评估 MedCorpus 和 LitSearch 数据集上，使用 L100_nq, L200_nq, L300_nq 叙述，
# 并由 jina 和 gte 重排器生成的完整重排结果。

# --- 配置 ---
DATASETS_TO_PROCESS="MedCorpus LitSearch"
RERANKER_TYPES_TO_EVAL="jina gte" 
NARRATIVE_LENGTHS_TO_EVAL="100 200 300"
# EVAL_K 应与 03b 脚本中重排时保留的文档数 (FINAL_TOP_K_SETTING) 匹配，
# 或者设置为您希望评估的标准 K 值 (如 10)。
EVAL_K=10 # 如果 03b 脚本中 FINAL_TOP_K_SETTING 为空 (使用默认10)，则这里也用10。
          # 如果 03b 脚本中 FINAL_TOP_K_SETTING=100, 则这里应设为 100。

# --- 脚本逻辑 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR_BASE="./logs_full_run_all_configs" # 与重排脚本使用相同的日志基础目录
mkdir -p "$LOG_DIR_BASE"

K_SUFFIX_LOG=""
# 根据EVAL_K调整日志和文件名中的后缀
# 如果EVAL_K不是10 (假设10是utils.py中final_results_path不加后缀的默认值)
if [ "$EVAL_K" -ne 10 ]; then
    K_SUFFIX_LOG="_top${EVAL_K}"
fi

MAIN_LOG_FILE="$LOG_DIR_BASE/04b_evaluate_full_all_configs_nq${K_SUFFIX_LOG}_${TIMESTAMP}.log"

echo "此脚本的主日志文件: $MAIN_LOG_FILE"
exec > >(tee -a "$MAIN_LOG_FILE") 2>&1

run_evaluation() {
    local dataset_name_arg="$1"
    local reranker_type_arg="$2"
    local narrative_len_arg="$3"
    local eval_k_arg="$4"
    local eval_desc="$5"
    local eval_log_file="$6"
    
    local gt_file_arg="./data/$dataset_name_arg/query_to_texts.jsonl"
    local unranked_pred_file_arg="./results/$dataset_name_arg/retrieved.jsonl"
    local original_queries_file_arg="./data/$dataset_name_arg/queries.jsonl"

    # 构造预期的重排文件名
    local narrative_suffix_nq="_nq"
    local k_suffix_for_filename=""
    # 与03b脚本中EXPECTED_K_SUFFIX_FILENAME_PART的逻辑保持一致
    # 如果评估的K值 (eval_k_arg) 不是10 (假设10是文件名中不加_topX的默认情况)
    # 或者如果03b脚本中明确设置了FINAL_TOP_K_SETTING并与eval_k_arg匹配
    if [ "$eval_k_arg" -ne 10 ]; then
        k_suffix_for_filename="_top${eval_k_arg}"
    fi
    # 如果03b脚本中 FINAL_TOP_K_SETTING 为空 (即使用默认的10), 且 eval_k_arg 也是10, 则 k_suffix_for_filename 应该为空
    # 这个逻辑确保我们能找到由03b脚本正确生成的文件名

    local reranked_file_to_eval="./results/$dataset_name_arg/ranked_${reranker_type_arg}_personalized${narrative_suffix_nq}_L${narrative_len_arg}${k_suffix_for_filename}.jsonl"

    echo "---------------------------------------------------------------------"
    echo "开始评估: $eval_desc"
    echo "  数据集: $dataset_name_arg, 重排器: $reranker_type_arg, 叙述长度: L${narrative_len_arg}, 评估TopK: $eval_k_arg"
    echo "  期望评估的重排文件: $reranked_file_to_eval"
    echo "  Ground Truth 文件: $gt_file_arg"
    echo "  评估日志将写入: $eval_log_file"
    echo "---------------------------------------------------------------------"

    if [ ! -f "$reranked_file_to_eval" ]; then
        echo "错误: $dataset_name_arg - 找不到要评估的重排文件: $reranked_file_to_eval。"
        echo "请确认对应的 03b_run_rerank... 脚本是否已成功执行并生成了此文件。"
        echo "跳过此评估。"
        echo "---------------------------------------------------------------------"
        return 1
    fi
    if [ ! -f "$gt_file_arg" ]; then
        echo "错误: $dataset_name_arg - 找不到Ground Truth文件: $gt_file_arg。跳过评估。"
        return 1
    fi
    if [ ! -f "$unranked_pred_file_arg" ]; then
        echo "警告: $dataset_name_arg - 找不到未重排的预测文件: $unranked_pred_file_arg。"
    fi

    CMD_EVAL_ARGS="python evaluate.py --dataset_name $dataset_name_arg \
                --gt_file $gt_file_arg \
                --rerank_pred_file $reranked_file_to_eval \
                --unrerank_pred_file $unranked_pred_file_arg \
                --k $eval_k_arg \
                --reranker_type $reranker_type_arg"

    if [ -f "$original_queries_file_arg" ]; then
        CMD_EVAL_ARGS="$CMD_EVAL_ARGS --queries_file $original_queries_file_arg --continuity_filter all"
    fi

    echo "执行评估命令: $CMD_EVAL_ARGS"
    # 将评估脚本的输出追加到主日志和特定日志
    eval $CMD_EVAL_ARGS | tee -a "$eval_log_file"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then # PIPESTATUS[0] 获取管道中第一个命令的退出状态
        echo "评估完成: $eval_desc"
    else
        echo "评估错误: $eval_desc 失败。请检查日志: $eval_log_file"
    fi
    echo "---------------------------------------------------------------------"
}

echo ""
echo "=============================================================="
echo "为数据集 ${DATASETS_TO_PROCESS} 评估所有配置的完整重排结果"
echo "叙述长度: ${NARRATIVE_LENGTHS_TO_EVAL}, 模式: _nq"
echo "重排器: ${RERANKER_TYPES_TO_EVAL}, 评估TopK: $EVAL_K"
echo "=============================================================="

for DATASET_NAME_CURRENT in $DATASETS_TO_PROCESS; do
    for RERANKER_TYPE_CURRENT in $RERANKER_TYPES_TO_EVAL; do
        for NARRATIVE_LEN_CURRENT in $NARRATIVE_LENGTHS_TO_EVAL; do
            echo ""
            echo "--- 开始评估: ${DATASET_NAME_CURRENT} | ${RERANKER_TYPE_CURRENT} | L${NARRATIVE_LEN_CURRENT}_nq | Top${EVAL_K} ---"
            
            CURRENT_LOG_SUBDIR="${LOG_DIR_BASE}/${DATASET_NAME_CURRENT}"
            mkdir -p "$CURRENT_LOG_SUBDIR"
            
            EVAL_LOG_FILE_SPECIFIC="$CURRENT_LOG_SUBDIR/eval_full_${DATASET_NAME_CURRENT}_${RERANKER_TYPE_CURRENT}_L${NARRATIVE_LEN_CURRENT}_nq${K_SUFFIX_LOG}_${TIMESTAMP}.log"

            run_evaluation "$DATASET_NAME_CURRENT" \
                           "$RERANKER_TYPE_CURRENT" \
                           "$NARRATIVE_LEN_CURRENT" \
                           "$EVAL_K" \
                           "${DATASET_NAME_CURRENT} | ${RERANKER_TYPE_CURRENT} | L${NARRATIVE_LEN_CURRENT}_nq | Top${EVAL_K}" \
                           "$EVAL_LOG_FILE_SPECIFIC"
            echo ""
        done
    done
    echo "--- 数据集 $DATASET_NAME_CURRENT 的所有配置评估完成 ---"
done

echo "=============================================================="
echo "所有数据集的所有配置评估完成。"
echo "=============================================================="
echo ""

echo "所有评估完成。"
echo "主日志: $MAIN_LOG_FILE"
echo "各个评估日志位于: $LOG_DIR_BASE/<dataset_name>/"
echo "请检查日志文件获取详细的评估指标。"
