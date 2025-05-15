#!/bin/bash

# consolidated_evaluate_all.sh
# Description:
# This script consolidates the evaluation of:
# 1. Main personalized reranking method
# 2. Current query reranking (history baseline window 0)
# 3. Various baseline methods
# 4. Ablation study results.
# It uses evaluate.py for 1, 2, and 3, and evaluate_memory_ablation_vX.py for 4.
# Key change: --unrerank_pred_file is NOW CONDITIONALLY USED for evaluate.py calls to enable P-imp.
# Log change: All Python script evaluation summaries are appended to a single consolidated results file.
# Ablation fix v2: Added _unreranked argument to evaluate_memory_ablation_vX.py call.

# --- General Configuration ---
BASE_PROJECT_DIR="/workspace/PerMed" # 项目根目录
PYTHON_EXECUTABLE="python" # Or python3, if applicable

# Evaluation scripts
EVAL_SCRIPT_MAIN="${BASE_PROJECT_DIR}/evaluate.py"
# Ensure this points to the correct version of your ablation evaluation script (e.g., v3.py)
EVAL_SCRIPT_ABLATION="${BASE_PROJECT_DIR}/evaluate_memory_ablation_v2.py" 

# Data and Results Directories
BASE_DATA_DIR="${BASE_PROJECT_DIR}/data"
MAIN_METHOD_RESULTS_DIR="${BASE_PROJECT_DIR}/results" # Used for main method, win0 history baseline, and unreranked files
BASELINES_RESULTS_DIR="${BASE_PROJECT_DIR}/baselines/results"
ABLATION_RESULTS_PARENT_DIR="${BASE_PROJECT_DIR}/results_ablation"

# --- Evaluation Parameters ---
DATASETS_TO_EVALUATE="MedCorpus LitSearch"
EVAL_K=10 # Standard K for evaluation (e.g., NDCG@K, Recall@K) - used by evaluate.py
DEBUG_MODE="--debug" # Add "--debug" for verbose output, or "" to disable

# --- Main Method & Window 0 History Baseline Configuration ---
RERANKER_TYPES_MAIN_METHOD="jina gte" # Rerankers to evaluate for main method and win0
NARRATIVE_LENGTHS_MAIN_METHOD="100 200 300" # Corresponds to L100, L200, L300 for main method
NARRATIVE_SUFFIX_MAIN_METHOD="_nq" # Suffix for personalized reranking files (e.g., _nq)

# --- Baselines Configuration ---
BASELINES_TO_EVALUATE="rpmn htps qer conqrr" # List your baseline names

# --- Ablation Study Configuration ---
ABLATION_TARGET_RERANKER_TYPE="jina" # Example: evaluate jina ablations
ABLATION_TARGET_NARRATIVE_LENGTHS="100" # Example: evaluate L100 ablations

# --- Logging ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR_BASE="${BASE_PROJECT_DIR}/evaluation_logs_consolidated"
mkdir -p "$LOG_DIR_BASE"
MAIN_EVAL_LOG_FILE="${LOG_DIR_BASE}/consolidated_evaluation_SCRIPT_RUN_${TIMESTAMP}.log"
CONSOLIDATED_RESULTS_LOG_FILE="${LOG_DIR_BASE}/consolidated_evaluation_RESULTS_SUMMARY_${TIMESTAMP}.log"

# Redirect all stdout/stderr of this script to the main log file and console
exec > >(tee -a "$MAIN_EVAL_LOG_FILE") 2>&1

echo "======================================================================"
echo "Consolidated Evaluation Script Started at $(date)"
echo "Main Script Execution Log File: $MAIN_EVAL_LOG_FILE"
echo "Consolidated Results Summary Log File: $CONSOLIDATED_RESULTS_LOG_FILE"
echo "Python Executable: $PYTHON_EXECUTABLE"
echo "Datasets: $DATASETS_TO_EVALUATE"
echo "Evaluation K (for evaluate.py): $EVAL_K"
echo "Ablation Evaluation Script: $EVAL_SCRIPT_ABLATION"
echo "======================================================================"
echo ""

# Initialize the consolidated results log file (optional, creates an empty file if it doesn't exist)
touch "$CONSOLIDATED_RESULTS_LOG_FILE"

# --- Helper Function for Logging and Execution ---
run_and_log_evaluation() {
    local description="$1"
    local command_to_run="$2"
    # Consolidated results log is now a global variable: CONSOLIDATED_RESULTS_LOG_FILE

    # These echos go to the main script log and console
    echo "----------------------------------------------------------------------"
    echo "Starting Evaluation: $description"
    echo "Command: $command_to_run"
    echo "Results will be appended to: $CONSOLIDATED_RESULTS_LOG_FILE"
    echo "----------------------------------------------------------------------"

    # Execute command and capture its full output (stdout and stderr)
    local full_output
    full_output=$(eval $command_to_run 2>&1)
    local exit_code=$?

    # Append a header for this specific evaluation to the consolidated results log
    echo "######################################################################" >> "$CONSOLIDATED_RESULTS_LOG_FILE"
    echo "## EVALUATION RESULTS FOR: $description" >> "$CONSOLIDATED_RESULTS_LOG_FILE"
    echo "## COMMAND EXECUTED: $command_to_run" >> "$CONSOLIDATED_RESULTS_LOG_FILE"
    echo "## TIMESTAMP: $(date)" >> "$CONSOLIDATED_RESULTS_LOG_FILE"
    echo "######################################################################" >> "$CONSOLIDATED_RESULTS_LOG_FILE"

    if [ $exit_code -eq 0 ]; then
        local summary_output
        summary_output=$(echo "$full_output" | awk '
            BEGIN {printing=0}
            # Adjust these patterns if evaluate_memory_ablation_vX.py has different summary headers
            /^={10,}.*Evaluation Results Summary/ {printing=1}
            /^\--- Evaluation Results \(Continuity:/ {printing=1} # For evaluate_memory_ablation_2.py
            /^Key Metrics Summary/ {if(!printing) printing=1}
            /^={10,}.*Conversation Evaluation Results/ {if(!printing) printing=1}
            /^Analysis by Turn/ {if(!printing) printing=1}
            /^Ablation Study Results:/ {if(!printing) printing=1}
            printing {print}
        ')

        if [ -n "$summary_output" ]; then
            echo "$summary_output" >> "$CONSOLIDATED_RESULTS_LOG_FILE"
            echo "SUCCESS: Evaluation completed for $description. Summary results appended."
        else
            echo "Warning: Results summary not found or empty for $description using awk filter." >> "$CONSOLIDATED_RESULTS_LOG_FILE"
            echo "$full_output" >> "$CONSOLIDATED_RESULTS_LOG_FILE" # Append full output
            echo "SUCCESS (with warning): Evaluation completed for $description. Full output appended."
        fi
    else
        echo "ERROR: Evaluation FAILED for $description. Exit code: $exit_code." >> "$CONSOLIDATED_RESULTS_LOG_FILE"
        echo "$full_output" >> "$CONSOLIDATED_RESULTS_LOG_FILE" # Append full output
        echo "ERROR: Evaluation FAILED for $description. Exit code: $exit_code. Full output appended to results log." # Also to main script log
    fi
    echo "" >> "$CONSOLIDATED_RESULTS_LOG_FILE" # Add a blank line for separation
    # These echos also go to the main script log and console
    echo "----------------------------------------------------------------------"
    echo ""
    return $exit_code
}

# Determine K suffix for filenames and logs, consistent with generation scripts
K_SUFFIX_FILENAME_PART=""
K_SUFFIX_LOG_PART="" # For description in logs
if [ "$EVAL_K" -ne 10 ]; then # Assuming 10 is the default K that doesn't add a suffix for evaluate.py
    K_SUFFIX_FILENAME_PART="_top${EVAL_K}"
    K_SUFFIX_LOG_PART="_top${EVAL_K}"
fi

# --- 1. Evaluate Main Personalized Reranking Method ---
echo "======================================================================"
echo "SECTION 1: Evaluating Main Personalized Reranking Method"
echo "Using script: $EVAL_SCRIPT_MAIN"
echo "======================================================================"
for DATASET_NAME in $DATASETS_TO_EVALUATE; do
    GT_FILE="${BASE_DATA_DIR}/${DATASET_NAME}/query_to_texts.jsonl"
    ORIGINAL_QUERIES_FILE="${BASE_DATA_DIR}/${DATASET_NAME}/queries.jsonl"
    # Define path for the unreranked (retrieved) file for P-imp calculation
    UNRERANKED_FILE_FOR_PIMP="${MAIN_METHOD_RESULTS_DIR}/${DATASET_NAME}/retrieved.jsonl"

    if [ ! -f "$GT_FILE" ]; then
        echo "ERROR (Main Method): Ground Truth file not found for $DATASET_NAME: $GT_FILE. Skipping."
        continue
    fi

    for RERANKER_TYPE in $RERANKER_TYPES_MAIN_METHOD; do
        for NARRATIVE_LEN in $NARRATIVE_LENGTHS_MAIN_METHOD; do
            PRED_FILE_NAME="ranked_${RERANKER_TYPE}_personalized${NARRATIVE_SUFFIX_MAIN_METHOD}_L${NARRATIVE_LEN}${K_SUFFIX_FILENAME_PART}.jsonl"
            RERANKED_PRED_FILE="${MAIN_METHOD_RESULTS_DIR}/${DATASET_NAME}/${PRED_FILE_NAME}"

            if [ ! -f "$RERANKED_PRED_FILE" ]; then
                echo "WARNING (Main Method): Prediction file not found: $RERANKED_PRED_FILE. Skipping."
                continue
            fi

            EVAL_DESC="MainMethod_${DATASET_NAME}_${RERANKER_TYPE}_L${NARRATIVE_LEN}${NARRATIVE_SUFFIX_MAIN_METHOD}${K_SUFFIX_LOG_PART}"
            CMD_ARGS="$PYTHON_EXECUTABLE $EVAL_SCRIPT_MAIN \
                --dataset_name \"$DATASET_NAME\" \
                --gt_file \"$GT_FILE\" \
                --rerank_pred_file \"$RERANKED_PRED_FILE\" \
                --k $EVAL_K \
                --reranker_type \"$RERANKER_TYPE\" \
                $DEBUG_MODE"
            
            # Conditionally add unrerank_pred_file for P-imp
            if [ -f "$UNRERANKED_FILE_FOR_PIMP" ]; then
                CMD_ARGS="$CMD_ARGS --unrerank_pred_file \"$UNRERANKED_FILE_FOR_PIMP\""
            else
                echo "WARNING (Main Method - $EVAL_DESC): Unreranked file for P-imp not found: $UNRERANKED_FILE_FOR_PIMP. P-imp will not be calculated."
            fi

            if [ -f "$ORIGINAL_QUERIES_FILE" ]; then
                 CMD_ARGS="$CMD_ARGS --queries_file \"$ORIGINAL_QUERIES_FILE\" --continuity_filter all"
            fi
            run_and_log_evaluation "$EVAL_DESC" "$CMD_ARGS"
        done
    done
done

# --- 2. Evaluate Current Query Reranking (History Baseline Window 0) ---
echo "======================================================================"
echo "SECTION 2: Evaluating Current Query Reranking (History Baseline Win 0)"
echo "Using script: $EVAL_SCRIPT_MAIN"
echo "======================================================================"
for DATASET_NAME in $DATASETS_TO_EVALUATE; do
    GT_FILE="${BASE_DATA_DIR}/${DATASET_NAME}/query_to_texts.jsonl"
    ORIGINAL_QUERIES_FILE="${BASE_DATA_DIR}/${DATASET_NAME}/queries.jsonl"
    UNRERANKED_FILE_FOR_PIMP="${MAIN_METHOD_RESULTS_DIR}/${DATASET_NAME}/retrieved.jsonl"

    if [ ! -f "$GT_FILE" ]; then
        echo "ERROR (Win0 Baseline): Ground Truth file not found for $DATASET_NAME: $GT_FILE. Skipping."
        continue
    fi

    for RERANKER_TYPE in $RERANKER_TYPES_MAIN_METHOD; do # Assuming same rerankers as main method
        # Assumed filename for window 0 history baseline
        PRED_FILE_NAME="ranked_${RERANKER_TYPE}_history_baseline_win0${K_SUFFIX_FILENAME_PART}.jsonl"
        RERANKED_PRED_FILE="${MAIN_METHOD_RESULTS_DIR}/${DATASET_NAME}/${PRED_FILE_NAME}"

        if [ ! -f "$RERANKED_PRED_FILE" ]; then
            echo "WARNING (Win0 Baseline): Prediction file not found: $RERANKED_PRED_FILE. Skipping."
            continue
        fi

        EVAL_DESC="HistoryBaselineWin0_${DATASET_NAME}_${RERANKER_TYPE}${K_SUFFIX_LOG_PART}"
        CMD_ARGS="$PYTHON_EXECUTABLE $EVAL_SCRIPT_MAIN \
            --dataset_name \"$DATASET_NAME\" \
            --gt_file \"$GT_FILE\" \
            --rerank_pred_file \"$RERANKED_PRED_FILE\" \
            --k $EVAL_K \
            --reranker_type \"${RERANKER_TYPE}_HistWin0\" \
            $DEBUG_MODE" # Modified reranker_type for clarity in logs
        
        # Conditionally add unrerank_pred_file for P-imp
        if [ -f "$UNRERANKED_FILE_FOR_PIMP" ]; then
            CMD_ARGS="$CMD_ARGS --unrerank_pred_file \"$UNRERANKED_FILE_FOR_PIMP\""
        else
            echo "WARNING (Win0 Baseline - $EVAL_DESC): Unreranked file for P-imp not found: $UNRERANKED_FILE_FOR_PIMP. P-imp will not be calculated."
        fi

        if [ -f "$ORIGINAL_QUERIES_FILE" ]; then
             CMD_ARGS="$CMD_ARGS --queries_file \"$ORIGINAL_QUERIES_FILE\" --continuity_filter all"
        fi
        run_and_log_evaluation "$EVAL_DESC" "$CMD_ARGS"
    done
done

# --- 3. Evaluate Other Baseline Methods ---
echo "======================================================================"
echo "SECTION 3: Evaluating Other Baseline Methods"
echo "Using script: $EVAL_SCRIPT_MAIN"
echo "======================================================================"
for DATASET_NAME in $DATASETS_TO_EVALUATE; do
    GT_FILE="${BASE_DATA_DIR}/${DATASET_NAME}/query_to_texts.jsonl"
    ORIGINAL_QUERIES_FILE="${BASE_DATA_DIR}/${DATASET_NAME}/queries.jsonl"
    # For baselines, the unreranked file might be specific to the baseline or a general one.
    # Here, we assume a general retrieved.jsonl. If baselines have their own "before reranking" step,
    # this path might need adjustment or the baseline itself should output its "before" state.
    # For now, using the general one from MAIN_METHOD_RESULTS_DIR for consistency.
    UNRERANKED_FILE_FOR_PIMP="${MAIN_METHOD_RESULTS_DIR}/${DATASET_NAME}/retrieved.jsonl"


    if [ ! -f "$GT_FILE" ]; then
        echo "ERROR (Other Baselines): Ground Truth file not found for $DATASET_NAME: $GT_FILE. Skipping."
        continue
    fi

    for BASELINE_NAME in $BASELINES_TO_EVALUATE; do
        RERANKED_PRED_FILE="${BASELINES_RESULTS_DIR}/${DATASET_NAME}/${BASELINE_NAME}_results.jsonl"

        if [ ! -f "$RERANKED_PRED_FILE" ]; then
            echo "WARNING (Other Baselines): Prediction file not found for $BASELINE_NAME on $DATASET_NAME: $RERANKED_PRED_FILE. Skipping."
            continue
        fi

        EVAL_DESC="Baseline_${DATASET_NAME}_${BASELINE_NAME}${K_SUFFIX_LOG_PART}"
        CMD_ARGS="$PYTHON_EXECUTABLE $EVAL_SCRIPT_MAIN \
            --dataset_name \"$DATASET_NAME\" \
            --gt_file \"$GT_FILE\" \
            --rerank_pred_file \"$RERANKED_PRED_FILE\" \
            --k $EVAL_K \
            --reranker_type \"$BASELINE_NAME\" \
            $DEBUG_MODE"

        # Conditionally add unrerank_pred_file for P-imp
        # Note: This assumes the baseline's "reranked" output can be compared against a common "unreranked" set.
        # If a baseline *is* an initial retrieval method, P-imp might not be meaningful unless compared to another initial retrieval.
        if [ -f "$UNRERANKED_FILE_FOR_PIMP" ]; then
            CMD_ARGS="$CMD_ARGS --unrerank_pred_file \"$UNRERANKED_FILE_FOR_PIMP\""
        else
            echo "WARNING (Other Baselines - $EVAL_DESC): Unreranked file for P-imp not found: $UNRERANKED_FILE_FOR_PIMP. P-imp might not be calculated or applicable."
        fi
        
        if [ -f "$ORIGINAL_QUERIES_FILE" ]; then
             CMD_ARGS="$CMD_ARGS --queries_file \"$ORIGINAL_QUERIES_FILE\" --continuity_filter all"
        fi
        run_and_log_evaluation "$EVAL_DESC" "$CMD_ARGS"
    done
done

# --- 4. Evaluate Ablation Studies ---
echo "======================================================================"
echo "SECTION 4: Evaluating Ablation Studies"
echo "Using script: $EVAL_SCRIPT_ABLATION"
echo "Ablation Target Reranker: $ABLATION_TARGET_RERANKER_TYPE"
echo "Ablation Target Narrative Lengths: $ABLATION_TARGET_NARRATIVE_LENGTHS"
echo "======================================================================"
for DATASET_NAME in $DATASETS_TO_EVALUATE; do
    GT_FILE="${BASE_DATA_DIR}/${DATASET_NAME}/query_to_texts.jsonl"
    # This unranked file is for the ablation script, which might have its own P-imp logic or specific needs.
    UNRANKED_FILE_FOR_ABLATION_SCRIPT="${MAIN_METHOD_RESULTS_DIR}/${DATASET_NAME}/retrieved.jsonl" 

    if [ ! -f "$GT_FILE" ]; then
        echo "ERROR (Ablation): Ground Truth file not found for $DATASET_NAME: $GT_FILE. Skipping."
        continue
    fi
    if [ ! -f "$UNRANKED_FILE_FOR_ABLATION_SCRIPT" ]; then
        echo "WARNING (Ablation): Unreranked file for ablation script not found: $UNRANKED_FILE_FOR_ABLATION_SCRIPT. P-imp in ablation script might not be calculated or script might error if it's mandatory."
        # Continue, as the Python script might handle this or P-imp might not be critical for this specific ablation run.
    fi

    for NARRATIVE_LEN_ABLATION in $ABLATION_TARGET_NARRATIVE_LENGTHS; do
        CURRENT_ABLATION_CONFIG_DIR="${ABLATION_RESULTS_PARENT_DIR}/${DATASET_NAME}/${ABLATION_TARGET_RERANKER_TYPE}_L${NARRATIVE_LEN_ABLATION}"

        if [ ! -d "$CURRENT_ABLATION_CONFIG_DIR" ]; then
            echo "WARNING (Ablation): Results directory not found: $CURRENT_ABLATION_CONFIG_DIR. Skipping."
            continue
        fi
        
        DATASET_SPECIFIC_ARGS="" 
        UNRANKED_ARG_FOR_ABLATION_SCRIPT=""

        if [ -f "$UNRANKED_FILE_FOR_ABLATION_SCRIPT" ]; then # Only add unreranked arg if file exists
            # The argument name for the ablation script (e.g., --medcorpus_unreranked) is specific to that script.
            UNRANKED_ARG_FOR_ABLATION_SCRIPT="--${DATASET_NAME,,}_unreranked \"$UNRANKED_FILE_FOR_ABLATION_SCRIPT\"" 
        fi

        if [ "$DATASET_NAME" == "MedCorpus" ]; then
            DATASET_SPECIFIC_ARGS="--medcorpus_dir \"$CURRENT_ABLATION_CONFIG_DIR\" --medcorpus_gt \"$GT_FILE\" $UNRANKED_ARG_FOR_ABLATION_SCRIPT"
        elif [ "$DATASET_NAME" == "LitSearch" ]; then
            DATASET_SPECIFIC_ARGS="--litsearch_dir \"$CURRENT_ABLATION_CONFIG_DIR\" --litsearch_gt \"$GT_FILE\" $UNRANKED_ARG_FOR_ABLATION_SCRIPT"
        else
            echo "ERROR (Ablation): Dataset $DATASET_NAME not configured for ablation script's dir args. Skipping."
            continue
        fi

        EVAL_DESC="Ablation_${DATASET_NAME}_${ABLATION_TARGET_RERANKER_TYPE}_L${NARRATIVE_LEN_ABLATION}"
        
        CMD_ARGS="$PYTHON_EXECUTABLE $EVAL_SCRIPT_ABLATION \
            $DATASET_SPECIFIC_ARGS \
            $DEBUG_MODE"

        run_and_log_evaluation "$EVAL_DESC" "$CMD_ARGS"
    done
done

echo "======================================================================"
echo "Consolidated Evaluation Script Finished at $(date)"
echo "All script execution logs can be found in: $MAIN_EVAL_LOG_FILE"
echo "All evaluation result summaries are consolidated in: $CONSOLIDATED_RESULTS_LOG_FILE"
echo "======================================================================"
