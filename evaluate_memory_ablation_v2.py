#!/usr/bin/env python
# evaluate_memory_ablation_v3.py
import json
import os
import argparse
import numpy as np
from tabulate import tabulate
from typing import Dict, List, Any, Optional, Tuple, Set
import math # Import math for log2
import re # For parsing filenames

# --- Utility Functions ---
def load_ground_truth_jsonl(gt_path, debug=False):
    """Load ground truth (JSON Lines format), ensure consistent ID types"""
    ground_truth = {}
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found {gt_path}")
        return {}
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    qid = str(data["query_id"])
                    relevant_texts = {str(doc_id) for doc_id in data["relevant_texts"]} # Use set for faster lookup
                    ground_truth[qid] = {"relevant_texts": relevant_texts}
                except Exception as e:
                    if debug: print(f"Error parsing GT line {line_num}: {e} in file {gt_path}")
    except Exception as e:
        if debug: print(f"Error loading GT file {gt_path}: {e}")
        return {}
    print(f"Loaded {len(ground_truth)} ground truth entries from {gt_path}")
    return ground_truth

def load_prediction_file(file_path, results_key="ranked_results", debug=False):
    """Load prediction results from a single file, return predictions and continuity info."""
    predictions = {}
    continuity_info = {} # Continuity info per query from this specific prediction file
    if not os.path.exists(file_path):
        if debug: print(f"Warning: Prediction file not found {file_path}")
        return {}, {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    d = json.loads(line)
                    qid = str(d.get("query_id"))
                    if not qid: continue

                    is_continuous = d.get("continuity", True) # Default to True if missing
                    continuity_info[qid] = is_continuous

                    preds_list = d.get(results_key, d.get("results")) 
                    if preds_list is None:
                        if debug: print(f"Warning: No results key ('{results_key}' or 'results') found in line {line_num} of {file_path}")
                        continue

                    pred_docs = [str(p["text_id"]) for p in preds_list if "text_id" in p]
                    if not pred_docs and debug:
                        print(f"Warning: Empty prediction list for query {qid} in {file_path}")

                    predictions[qid] = pred_docs
                except Exception as e:
                    if debug: print(f"Error parsing prediction line {line_num} in {file_path}: {e}")
    except Exception as e:
        if debug: print(f"Error loading prediction file {file_path}: {e}")
        return {}, {}
    if debug: print(f"Loaded {len(predictions)} predictions from {file_path}")
    return predictions, continuity_info

# --- Metric Calculation Functions (Unchanged) ---
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

def average_precision(r, k):
    r = np.asarray(r)[:k]
    out = [precision_at_k(r, i + 1) for i in range(r.size) if r[i]]
    if not out:
        return 0.0
    return np.mean(out)

def precision_at_k(r, k):
    r = np.asarray(r)[:k]
    return np.mean(r) if r.size else 0.0

def recall_at_k(r, total_relevant, k):
    r = np.asarray(r)[:k]
    return np.sum(r) / total_relevant if total_relevant > 0 else 0.0

def reciprocal_rank(r):
    r = np.asarray(r)
    for i, rel in enumerate(r):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0

def calculate_metrics_for_query_set(predictions, ground_truth, k=10):
    ndcgs, maps, recalls, rranks, p1s = [], [], [], [], []
    query_count = 0

    for qid, gt_data in ground_truth.items():
        if qid not in predictions:
            continue 

        relevant_docs = gt_data["relevant_texts"]
        if not relevant_docs:
            continue 

        pred_docs = predictions[qid]
        relevance_list = [1 if doc_id in relevant_docs else 0 for doc_id in pred_docs]

        ndcgs.append(ndcg_at_k(relevance_list, k))
        maps.append(average_precision(relevance_list, k))
        recalls.append(recall_at_k(relevance_list, len(relevant_docs), k))
        rranks.append(reciprocal_rank(relevance_list))
        p1s.append(float(relevance_list[0]) if relevance_list else 0.0)
        query_count += 1

    metrics = {
        "query_count": query_count,
        f"NDCG@{k}": np.mean(ndcgs) if ndcgs else 0.0,
        f"MAP@{k}": np.mean(maps) if maps else 0.0,
        f"Recall@{k}": np.mean(recalls) if recalls else 0.0,
        "MRR": np.mean(rranks) if rranks else 0.0,
        "P@1": np.mean(p1s) if p1s else 0.0
    }
    return metrics

def calculate_p_imp(unrerank_preds, rerank_preds, ground_truth, k=10):
    improved_pairs, total_comparable_pairs = 0, 0
    for qid, gt_data in ground_truth.items():
        if qid not in unrerank_preds or qid not in rerank_preds: continue
        relevant_set = gt_data["relevant_texts"]
        unrerank_list = unrerank_preds[qid][:k]
        rerank_list = rerank_preds[qid][:k]

        unrerank_pos = {doc_id: i for i, doc_id in enumerate(unrerank_list)}
        
        for i, doc_i in enumerate(rerank_list):
            for j in range(i + 1, len(rerank_list)):
                doc_j = rerank_list[j]
                rel_i = 1 if doc_i in relevant_set else 0
                rel_j = 1 if doc_j in relevant_set else 0

                if rel_i != rel_j:
                    if doc_i in unrerank_pos and doc_j in unrerank_pos:
                        total_comparable_pairs += 1
                        correct_new_order = (rel_i > rel_j)
                        correct_old_order = (rel_i > rel_j and unrerank_pos[doc_i] < unrerank_pos[doc_j]) or \
                                            (rel_j > rel_i and unrerank_pos[doc_j] < unrerank_pos[doc_i])
                        if correct_new_order and not correct_old_order:
                            improved_pairs += 1
    return improved_pairs / total_comparable_pairs if total_comparable_pairs > 0 else 0.0

# --- Evaluation Logic ---

def parse_config_name_from_filename(filename: str) -> Optional[str]:
    """Parses a friendly configuration name from the ablation result filename."""
    if "no_sequential" in filename:
        return "WM+LM (no Sequential)"
    elif "no_working" in filename:
        return "SM+LM (no Working)"
    elif "no_long" in filename:
        return "SM+WM (no Long)"
    # Add other patterns if needed, e.g., for "all_memory" if you reintroduce it
    # elif "all_memory" in filename:
    #     return "SM+WM+LM (All Memory)"
    return None

def load_all_results_dynamically(dataset_name: str, ablation_results_dir: str, gt_file: str, unreranked_file: str, debug=False):
    """
    Loads GT, unranked, and all ablation config results by scanning the ablation_results_dir.
    """
    ground_truth = load_ground_truth_jsonl(gt_file, debug)
    if not ground_truth: 
        return None, None, None, None, None # Return 5 Nones if GT fails

    unrerank_predictions, _ = load_prediction_file(unreranked_file, "results", debug)
    if not unrerank_predictions: 
        return ground_truth, None, None, None, None # Return 5 values if unreranked fails

    all_predictions = {}
    all_continuity_info = {} # Stores continuity info from each loaded ablation file

    if not os.path.isdir(ablation_results_dir):
        print(f"Error: Ablation results directory not found: {ablation_results_dir}")
        return ground_truth, unrerank_predictions, {}, {}, set(ground_truth.keys()) & set(unrerank_predictions.keys())

    print(f"Scanning directory for ablation results: {ablation_results_dir}")
    for filename in os.listdir(ablation_results_dir):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(ablation_results_dir, filename)
            config_name_friendly = parse_config_name_from_filename(filename)
            
            if config_name_friendly:
                if debug: print(f"Found potential ablation file: {filename}, parsed as config: {config_name_friendly}")
                preds, continuity = load_prediction_file(file_path, "ranked_results", debug)
                if preds:
                    all_predictions[config_name_friendly] = preds
                    all_continuity_info[config_name_friendly] = continuity # Store continuity for this config
                elif debug:
                     print(f"Warning: No predictions loaded for config '{config_name_friendly}' from {file_path}")
            elif debug:
                print(f"Skipping file (could not parse config name): {filename}")


    if not all_predictions:
        print(f"Warning: No results loaded for any ablation configuration for {dataset_name} from directory {ablation_results_dir}")
        # Return 5 values, with all_predictions and all_continuity_info as empty dicts
        return ground_truth, unrerank_predictions, {}, {}, set(ground_truth.keys()) & set(unrerank_predictions.keys())

    # Find common query IDs across GT, unranked, and *at least one* config result
    common_qids = set(ground_truth.keys()) & set(unrerank_predictions.keys())
    if all_predictions: # Only intersect with config_qids if any were loaded
        config_qids = set().union(*[set(preds.keys()) for preds in all_predictions.values()])
        common_qids &= config_qids
    
    if not common_qids:
        print(f"Warning: No common query IDs found across GT, unranked, and loaded configuration results for {dataset_name}")
        # Return empty for all_predictions and all_continuity_info but keep others
        return ground_truth, unrerank_predictions, {}, {}, set()


    print(f"Found {len(common_qids)} common queries for evaluation in {dataset_name} after dynamic loading.")
    return ground_truth, unrerank_predictions, all_predictions, all_continuity_info, common_qids


def evaluate_ablation_configs(all_predictions, all_continuity_info, common_query_ids, ground_truth, unrerank_predictions, k=10, debug=False):
    """Evaluate metrics for different memory configurations. Continuity is assumed 'all'."""
    results = {"configs": {}, "unreranked": {}}
    
    # Since continuity_filter is fixed to "all", all common_query_ids are used.
    query_ids_to_evaluate = common_query_ids

    if not query_ids_to_evaluate:
        print(f"Warning: No common queries to evaluate. Skipping.")
        return None

    print(f"Evaluating {len(query_ids_to_evaluate)} queries (continuity_type: all)")

    filtered_gt = {qid: ground_truth[qid] for qid in query_ids_to_evaluate if qid in ground_truth}
    filtered_unrerank = {qid: unrerank_predictions[qid] for qid in query_ids_to_evaluate if qid in unrerank_predictions}

    results["unreranked"] = calculate_metrics_for_query_set(filtered_unrerank, filtered_gt, k)
    results["unreranked"]["config"] = "Unranked"
    results["unreranked"]["continuity_type"] = "all" # Fixed

    for config_name, preds in all_predictions.items():
        filtered_preds = {qid: preds[qid] for qid in query_ids_to_evaluate if qid in preds}
        if not filtered_preds:
            if debug: print(f"Warning: No predictions for config '{config_name}' after filtering for common queries.")
            continue

        config_metrics = calculate_metrics_for_query_set(filtered_preds, filtered_gt, k)
        config_metrics["P-imp"] = calculate_p_imp(filtered_unrerank, filtered_preds, filtered_gt, k)
        config_metrics["config"] = config_name
        config_metrics["continuity_type"] = "all" # Fixed
        results["configs"][config_name] = config_metrics
    return results

# --- Display Functions ---
def display_results_table(metrics_data, k=10):
    """Displays the evaluation results in a formatted table for 'all' continuity."""
    if not metrics_data or not metrics_data.get("unreranked"):
        print("No metrics data to display.")
        return

    headers = ["Memory Config", f"NDCG@{k}", f"MAP@{k}", f"Recall@{k}", "MRR", "P@1", "P-imp"]
    metric_keys = [f"NDCG@{k}", f"MAP@{k}", f"Recall@{k}", "MRR", "P@1", "P-imp"]
    
    continuity_type_display = "all" # Fixed
    query_count_display = metrics_data["unreranked"].get('query_count', 0)

    print(f"\n--- Evaluation Results (Continuity: {continuity_type_display}, Queries: {query_count_display}) ---")
    table_data = []

    unranked_row = ["Unranked"] + [f"{metrics_data['unreranked'].get(key, 0.0):.4f}" if key != "P-imp" else "N/A" for key in metric_keys]
    table_data.append(unranked_row)

    # Desired order for ablation configurations
    config_order = [
        "WM+LM (no Sequential)", 
        "SM+LM (no Working)", 
        "SM+WM (no Long)"
    ]
    # If you add "SM+WM+LM (All Memory)" back to generation, add it here:
    # config_order.insert(0, "SM+WM+LM (All Memory)")


    for config_name in config_order:
        if config_name in metrics_data["configs"]:
            config_metrics = metrics_data["configs"][config_name]
            row = [config_name] + [f"{config_metrics.get(key, 0.0):.4f}" for key in metric_keys]
            table_data.append(row)
        # else:
        #     print(f"Debug: Config '{config_name}' not found in metrics_data['configs']")


    if table_data:
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    else:
        print("No configuration data to display in table.")

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Memory Ablation Experiment Evaluation (V3 - Dynamic File Loading)")
    parser.add_argument("--coral_dir", help="Directory containing CORAL ablation results for a specific config (e.g., jina_L100).")
    parser.add_argument("--coral_gt", help="Ground truth file for CORAL.")
    parser.add_argument("--coral_unreranked", help="Unreranked (retrieved) results for CORAL.")
    
    parser.add_argument("--litsearch_dir", type=str, help="Directory containing LitSearch ablation results for a specific config.")
    parser.add_argument("--litsearch_gt", type=str, help="LitSearch ground truth file.")
    parser.add_argument("--litsearch_unreranked", type=str, help="LitSearch unranked results file.")

    parser.add_argument("--medcorpus_dir", type=str, help="Directory containing MedCorpus ablation results for a specific config.")
    parser.add_argument("--medcorpus_gt", type=str, help="MedCorpus ground truth file.")
    parser.add_argument("--medcorpus_unreranked", type=str, help="MedCorpus unranked results file.")
    
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    args = parser.parse_args()

    datasets_to_evaluate = []
    if args.coral_dir and args.coral_gt and args.coral_unreranked:
        datasets_to_evaluate.append({
            "name": "CORAL",
            "ablation_results_dir": args.coral_dir, # This dir contains the jina_L100_no_xxx_all.jsonl files
            "gt_file": args.coral_gt,
            "unreranked_file": args.coral_unreranked
        })
    if args.litsearch_dir and args.litsearch_gt and args.litsearch_unreranked:
         datasets_to_evaluate.append({
             "name": "LitSearch",
             "ablation_results_dir": args.litsearch_dir,
             "gt_file": args.litsearch_gt,
             "unreranked_file": args.litsearch_unreranked
         })
    if args.medcorpus_dir and args.medcorpus_gt and args.medcorpus_unreranked:
         datasets_to_evaluate.append({
             "name": "MedCorpus",
             "ablation_results_dir": args.medcorpus_dir,
             "gt_file": args.medcorpus_gt,
             "unreranked_file": args.medcorpus_unreranked
         })

    if not datasets_to_evaluate:
        print("No datasets specified or required files missing. Provide paths using arguments like --coral_dir, --coral_gt, etc.")
        return

    for dataset_info in datasets_to_evaluate:
        dataset_name = dataset_info["name"]
        ablation_results_dir = dataset_info["ablation_results_dir"] # Directory with actual ablation files
        gt_file = dataset_info["gt_file"]
        unreranked_file = dataset_info["unreranked_file"]

        print(f"\n{'='*60}")
        print(f"Evaluating Memory Ablation for: {dataset_name}")
        print(f"Ablation results directory: {ablation_results_dir}")
        print(f"{'='*60}")
        
        # Load results dynamically for the "all" continuity type
        load_result = load_all_results_dynamically(
            dataset_name, ablation_results_dir, gt_file, unreranked_file, args.debug
        )

        if load_result is None or load_result[2] is None : # Check if all_predictions is None or empty
            print(f"Failed to load any valid results for {dataset_name}. Skipping evaluation for this dataset.")
            if load_result is not None and not load_result[2] and args.debug: # all_predictions is empty dict
                 print(f"Debug: load_all_results_dynamically returned empty 'all_predictions' for {dataset_name}.")
            continue
        
        # Unpack results
        ground_truth, unrerank_predictions, all_predictions, all_continuity_info, common_qids = load_result

        if not all_predictions: # Double check if all_predictions is empty after unpacking
            print(f"No ablation configurations loaded for {dataset_name} from {ablation_results_dir}. Skipping evaluation display.")
            continue
            
        # Evaluate metrics (continuity is implicitly "all" now)
        eval_metrics_for_all_continuity = evaluate_ablation_configs(
            all_predictions, all_continuity_info, common_qids,
            ground_truth, unrerank_predictions, k=10, debug=args.debug
        )
        
        if eval_metrics_for_all_continuity:
            display_results_table(eval_metrics_for_all_continuity, k=10)
        else:
            print(f"No evaluation metrics generated for {dataset_name} (continuity: all).")

if __name__ == "__main__":
    main()
