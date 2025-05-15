#!/usr/bin/env python
# evaluate.py
import json
import math
import os
import argparse
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

def load_continuity_info(queries_path):
    """
    Load continuity information from original queries file
    
    Args:
        queries_path: Path to queries file
        
    Returns:
        Dictionary mapping query ID to continuity value
    """
    continuity_map = {}
    try:
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                query_id = str(data.get("query_id", ""))
                continuity = data.get("continuity", True)  # Default to True if not specified
                if query_id:
                    continuity_map[query_id] = continuity
        print(f"Loaded continuity information for {len(continuity_map)} queries")
        return continuity_map
    except Exception as e:
        print(f"Error loading continuity information: {e}")
        return {}

def load_ground_truth_jsonl(gt_path: str, debug=False):
    """Load ground truth (JSON Lines format), ensure consistent ID types"""
    ground_truth = {}
    
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth file not found {gt_path}")
        return {}
        
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                line_count += 1
                try:
                    data = json.loads(line)
                    qid = str(data["query_id"])  # Ensure string ID type
                    relevant_texts = [str(doc_id) for doc_id in data["relevant_texts"]]  # Ensure document IDs are strings
                    
                    ground_truth[qid] = {
                        "relevant_texts": relevant_texts
                    }
                except Exception as e:
                    print(f"Error parsing line {line_count}: {e}")
                    if debug:
                        print(f"Problematic line content: {line[:100]}...")
            
            if debug:
                print(f"Successfully loaded {len(ground_truth)} ground truth items")
                if len(ground_truth) > 0:
                    sample_key = next(iter(ground_truth))
                    print(f"Sample data - Query ID: {sample_key}")
                    print(f"Relevant docs: {ground_truth[sample_key]['relevant_texts'][:5]}...")
    
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return {}
        
    return ground_truth

def load_predictions(pred_path: str, results_key: str = "ranked_results", debug=False):
    """Load prediction results, ensure consistent ID types"""
    predictions = {}
    raw_data = {}
    
    if not os.path.exists(pred_path):
        print(f"Error: Prediction file not found {pred_path}")
        return {}, {}
    
    try:    
        with open(pred_path, 'r', encoding='utf-8') as f:
            line_count = 0
            valid_count = 0
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                line_count += 1
                try:
                    d = json.loads(line)
                    qid = str(d["query_id"])  # Ensure string ID type
                    
                    # Try to get results using specified key, fallback to alternatives
                    if results_key in d:
                        preds = d[results_key]
                    elif "results" in d:
                        preds = d["results"]
                    else:
                        # Try to intelligently infer the right key
                        potential_keys = [k for k in d.keys() if "result" in k.lower()]
                        if potential_keys:
                            preds = d[potential_keys[0]]
                        else:
                            print(f"Warning: No results found in line {line_count}, skipping")
                            continue
                    
                    # Ensure document IDs are strings
                    pred_docs = [str(p["text_id"]) for p in preds if "text_id" in p]
                    
                    if not pred_docs:
                        # Allow empty predictions for a query, it will be handled in evaluation
                        # print(f"Warning: Line {line_count} for qid {qid} has empty or invalid results list")
                        pass # Continue to store empty list if that's the case
                        
                    predictions[qid] = pred_docs
                    raw_data[qid] = d
                    valid_count += 1
                    
                except Exception as e:
                    print(f"Error parsing line {line_count} (qid: {d.get('query_id', 'unknown')}): {e}")
                    if debug:
                        print(f"Problematic line content: {line[:100]}...")
            
            if debug:
                print(f"Successfully loaded {valid_count}/{line_count} prediction results")
                if valid_count > 0 and predictions: # Check if predictions is not empty
                    sample_key = next(iter(predictions))
                    print(f"Sample data - Query ID: {sample_key}")
                    print(f"Predicted docs: {predictions[sample_key][:5]}...")
    
    except Exception as e:
        print(f"Error loading prediction file: {e}")
        return {}, {}
        
    return predictions, raw_data

def dcg_at_k(r: List[int], k: int) -> float:
    """Calculate DCG@k"""
    r = r[:k]
    return sum([(2**rel - 1)/math.log2(i+2) for i, rel in enumerate(r)])

def ndcg_at_k(r: List[int], k: int) -> float:
    """Calculate NDCG@k"""
    dcg = dcg_at_k(r, k)
    sorted_r = sorted(r, reverse=True) # Ideal ranking
    idcg = dcg_at_k(sorted_r, k)
    return dcg/idcg if idcg > 0 else 0.0

def average_precision(r: List[int], k: int) -> float:
    """Calculate average precision"""
    r = r[:k] # Consider only top k results
    hits = 0
    sum_precisions = 0.0
    for i, rel in enumerate(r):
        if rel == 1:
            hits += 1
            sum_precisions += hits / (i+1)
    return sum_precisions / hits if hits > 0 else 0.0

def mean_reciprocal_rank(rs: List[List[int]]) -> float:
    """Calculate mean reciprocal rank.
    Args:
        rs: A list of lists, where each inner list is the binary relevance judgements for a query.
    """
    mrr_val = 0.0
    query_count = 0
    for r_item in rs:
        for i, rel in enumerate(r_item):
            if rel == 1:
                mrr_val += 1/(i+1)
                break  # Found first relevant, move to next query's r_item
        query_count += 1 # Increment count for every query list processed (even if no relevant docs)
    return mrr_val / query_count if query_count > 0 else 0.0

def recall_at_k(r: List[int], total_pos: int, k: int) -> float:
    """Calculate Recall@k"""
    hits = sum(r[:k]) # Sum of relevant items in top k
    return hits / total_pos if total_pos > 0 else 0.0

def calculate_p_imp(unrerank_preds: Dict[str, List[str]], rerank_preds: Dict[str, List[str]], 
                    ground_truth: Dict[str, Dict[str, List[str]]], k: int) -> float:
    """
    Calculate P-imp (Pairwise Improvement Rate) - document pair improvement rate
    Measures actual ranking improvement from personalized reranking
    """
    improved_pairs = 0
    total_comparable_pairs = 0 # Pairs where one doc is relevant and other is not
    
    for qid in ground_truth.keys():
        if qid not in unrerank_preds or qid not in rerank_preds:
            continue
            
        relevant_docs_set = set(ground_truth[qid]["relevant_texts"])
        # Consider only top-k for fair comparison if lists are of different lengths
        unrerank_list_k = unrerank_preds[qid][:k]
        rerank_list_k = rerank_preds[qid][:k]
        
        # Iterate through all unique pairs of documents in the reranked list (up to k)
        for i in range(len(rerank_list_k)):
            for j in range(i + 1, len(rerank_list_k)):
                doc_i = rerank_list_k[i]
                doc_j = rerank_list_k[j]

                doc_i_is_relevant = doc_i in relevant_docs_set
                doc_j_is_relevant = doc_j in relevant_docs_set

                # Only consider pairs where one is relevant and the other is not
                if doc_i_is_relevant != doc_j_is_relevant:
                    total_comparable_pairs += 1
                    
                    # Current order in reranked list: doc_i is before doc_j
                    # Correct order: relevant doc is before irrelevant doc
                    current_order_is_correct = (doc_i_is_relevant and not doc_j_is_relevant)

                    # Find positions in unreranked list
                    try:
                        pos_i_unreranked = unrerank_list_k.index(doc_i)
                    except ValueError:
                        pos_i_unreranked = float('inf') # Treat as if at the end if not found

                    try:
                        pos_j_unreranked = unrerank_list_k.index(doc_j)
                    except ValueError:
                        pos_j_unreranked = float('inf')

                    # Original order in unreranked list
                    # If original order was incorrect (irrelevant before relevant, or same relevance but wrong order if we considered that)
                    original_order_was_incorrect = (not doc_i_is_relevant and doc_j_is_relevant and pos_i_unreranked < pos_j_unreranked) or \
                                                   (doc_i_is_relevant and not doc_j_is_relevant and pos_j_unreranked < pos_i_unreranked)


                    if current_order_is_correct and original_order_was_incorrect:
                        improved_pairs += 1
    
    print(f"P-imp: Improved pairs: {improved_pairs}, Total comparable pairs (Rel vs Irrel): {total_comparable_pairs}")
    return improved_pairs / total_comparable_pairs if total_comparable_pairs > 0 else 0.0

def precision_at_1(predictions: Dict[str, List[str]], ground_truth: Dict[str, Dict[str, List[str]]]) -> float:
    """
    Calculate P@1 (Precision at 1) - measures if system can place most relevant document first
    """
    p1_values = []
    for qid in ground_truth: # Iterate over ground truth to ensure all relevant queries are considered
        if qid not in predictions:
            p1_values.append(0.0) # Query not in predictions, P@1 is 0
            continue
            
        relevant_docs_set = set(ground_truth[qid]["relevant_texts"])
        pred_docs = predictions[qid]
        
        if not pred_docs: # No predictions for this query
            p1_values.append(0.0)
            continue
            
        # Check if first predicted document is relevant
        if pred_docs[0] in relevant_docs_set:
            p1_values.append(1.0)
        else:
            p1_values.append(0.0)
    
    return sum(p1_values) / len(p1_values) if p1_values else 0.0

def evaluate(predictions: Dict[str, List[str]], ground_truth: Dict[str, Dict[str, List[str]]], k: int, debug=False):
    """Evaluate retrieval performance"""
    ndcgs = []
    aps = []
    rr_list_for_mrr = [] # Stores binary relevance lists for all queries for MRR calculation
    recalls = []
    
    sorted_qids = sorted(ground_truth.keys()) # Ensure consistent order
    
    for qid in sorted_qids:
        if qid not in predictions:
            if debug:
                print(f"Warning: Query ID {qid} not found in predictions. Metrics will be 0 for this query.")
            # For queries in GT but not preds, treat as all non-relevant predictions for fairness
            # This means NDCG, MAP, Recall, P@1 will be 0. For MRR, it's as if no relevant doc found.
            r = [] 
            # Metrics for this query if not in predictions
            ndcgs.append(0.0)
            aps.append(0.0)
            recalls.append(0.0) # Recall needs total_pos, which might be > 0
            rr_list_for_mrr.append(r) # Append empty list for MRR consistency
            continue
            
        relevant_docs_set = set(ground_truth[qid]["relevant_texts"])
        
        # total_pos should be based on ground truth, not predictions
        total_pos = len(relevant_docs_set)

        if not relevant_docs_set and debug: # No relevant documents in GT for this query
            print(f"Warning: Query ID {qid} has no relevant documents in ground truth. Metrics like Recall, NDCG might be 0 or undefined.")
            # If no relevant docs, NDCG, MAP, Recall are typically 0. MRR is also 0.
            # r will be all 0s if pred_docs exist.
        
        pred_docs = predictions[qid]
        
        # Build binary relevance list based on pred_docs
        r = [1 if doc_id in relevant_docs_set else 0 for doc_id in pred_docs]
        
        ndcgs.append(ndcg_at_k(r, k))
        aps.append(average_precision(r, k)) # AP already considers up to k internally
        rr_list_for_mrr.append(r) # Add the full relevance list for this query for MRR
        
        recalls.append(recall_at_k(r, total_pos, k))
        
        if debug and (len(ndcgs) <= 5 or ndcgs[-1] > 0 or recalls[-1] > 0): # Show some examples
            print(f"Query ID {qid}: NDCG@{k}={ndcgs[-1]:.4f}, MAP@{k}={aps[-1]:.4f}, Recall@{k}={recalls[-1]:.4f}")
            print(f"  Predicted order (top {k}): {pred_docs[:k]}")
            print(f"  Relevant docs (GT): {list(relevant_docs_set)}")
            print(f"  Binary relevance (pred): {r[:k]}")

    mean_ndcg = sum(ndcgs)/len(ndcgs) if ndcgs else 0.0
    mean_map = sum(aps)/len(aps) if aps else 0.0
    mrr = mean_reciprocal_rank(rr_list_for_mrr) # Calculate MRR over all queries
    mean_recall = sum(recalls)/len(recalls) if recalls else 0.0

    return mean_ndcg, mean_map, mrr, mean_recall

def evaluate_by_conversation(predictions: Dict[str, List[str]], 
                             ground_truth: Dict[str, Dict[str, List[str]]], 
                             k=10, debug=False):
    """Evaluate by conversation groups"""
    conversation_metrics_collection = defaultdict(lambda: {
        "queries": [], "ndcgs": [], "maps": [], "recalls": [], "p1s": [], "r_lists": []
    })
    
    for qid in ground_truth.keys(): # Iterate GT to ensure all queries are considered
        topic_id = qid.split("_")[0] if "_" in qid else qid # Basic topic_id extraction
        
        current_preds = predictions.get(qid, []) # Get predictions or empty list if qid not found
        relevant_docs_set = set(ground_truth[qid]["relevant_texts"])
        total_pos = len(relevant_docs_set)

        r = [1 if doc_id in relevant_docs_set else 0 for doc_id in current_preds]
        
        conversation_metrics_collection[topic_id]["queries"].append(qid)
        conversation_metrics_collection[topic_id]["ndcgs"].append(ndcg_at_k(r, k))
        conversation_metrics_collection[topic_id]["maps"].append(average_precision(r, k))
        conversation_metrics_collection[topic_id]["recalls"].append(recall_at_k(r, total_pos, k))
        conversation_metrics_collection[topic_id]["p1s"].append(1.0 if r and r[0] == 1 else 0.0)
        conversation_metrics_collection[topic_id]["r_lists"].append(r)
    
    aggregated_by_conv = {}
    all_q_ndcgs, all_q_maps, all_q_recalls, all_q_p1s, all_q_r_lists = [], [], [], [], []

    for topic_id, metrics_data in conversation_metrics_collection.items():
        if not metrics_data["queries"]: continue
            
        aggregated_by_conv[topic_id] = {
            "query_count": len(metrics_data["queries"]),
            "mean_ndcg": sum(metrics_data["ndcgs"]) / len(metrics_data["ndcgs"]) if metrics_data["ndcgs"] else 0.0,
            "mean_map": sum(metrics_data["maps"]) / len(metrics_data["maps"]) if metrics_data["maps"] else 0.0,
            "mean_recall": sum(metrics_data["recalls"]) / len(metrics_data["recalls"]) if metrics_data["recalls"] else 0.0,
            "mean_p1": sum(metrics_data["p1s"]) / len(metrics_data["p1s"]) if metrics_data["p1s"] else 0.0,
            "mrr": mean_reciprocal_rank(metrics_data["r_lists"])
        }
        all_q_ndcgs.extend(metrics_data["ndcgs"])
        all_q_maps.extend(metrics_data["maps"])
        all_q_recalls.extend(metrics_data["recalls"])
        all_q_p1s.extend(metrics_data["p1s"])
        all_q_r_lists.extend(metrics_data["r_lists"])
            
    overall_metrics = {
        "topic_count": len(aggregated_by_conv),
        "query_count": len(all_q_ndcgs), # Total number of queries processed
        "mean_ndcg": sum(all_q_ndcgs) / len(all_q_ndcgs) if all_q_ndcgs else 0.0,
        "mean_map": sum(all_q_maps) / len(all_q_maps) if all_q_maps else 0.0,
        "mean_recall": sum(all_q_recalls) / len(all_q_recalls) if all_q_recalls else 0.0,
        "mean_p1": sum(all_q_p1s) / len(all_q_p1s) if all_q_p1s else 0.0,
        "mrr": mean_reciprocal_rank(all_q_r_lists)
    }
    
    turn_analysis_results = analyze_by_turn(predictions, ground_truth, k, debug) # Pass original data
    
    return {
        "by_conversation": aggregated_by_conv,
        "by_turn": turn_analysis_results,
        "overall": overall_metrics
    }

def analyze_by_turn(predictions: Dict[str, List[str]], 
                    ground_truth: Dict[str, Dict[str, List[str]]], 
                    k=10, debug=False):
    """Analyze performance by turn, using predictions and ground_truth directly."""
    turn_metrics_collection = defaultdict(lambda: {
        "queries": [], "ndcgs": [], "maps": [], "recalls": [], "p1s": [], "r_lists": []
    })
    
    for qid in ground_truth.keys(): # Iterate GT
        turn_id_val = 0 # Default turn_id
        if "_" in qid:
            parts = qid.split("_")
            if len(parts) > 1 and parts[-1].isdigit(): # Assumes turn ID is the last part if numeric
                turn_id_val = int(parts[-1])
            # Add more sophisticated turn ID parsing if needed, e.g., for "topic_1_0" vs "topic_0"
        
        current_preds = predictions.get(qid, [])
        relevant_docs_set = set(ground_truth[qid]["relevant_texts"])
        total_pos = len(relevant_docs_set)

        r = [1 if doc_id in relevant_docs_set else 0 for doc_id in current_preds]
        
        turn_metrics_collection[turn_id_val]["queries"].append(qid)
        turn_metrics_collection[turn_id_val]["ndcgs"].append(ndcg_at_k(r, k))
        turn_metrics_collection[turn_id_val]["maps"].append(average_precision(r, k))
        turn_metrics_collection[turn_id_val]["recalls"].append(recall_at_k(r, total_pos, k))
        turn_metrics_collection[turn_id_val]["p1s"].append(1.0 if r and r[0] == 1 else 0.0)
        turn_metrics_collection[turn_id_val]["r_lists"].append(r)
        
    aggregated_by_turn = {}
    for turn_id_val, metrics_data in sorted(turn_metrics_collection.items()):
        if not metrics_data["queries"]: continue
        aggregated_by_turn[turn_id_val] = {
            "query_count": len(metrics_data["queries"]),
            "mean_ndcg": sum(metrics_data["ndcgs"]) / len(metrics_data["ndcgs"]) if metrics_data["ndcgs"] else 0.0,
            "mean_map": sum(metrics_data["maps"]) / len(metrics_data["maps"]) if metrics_data["maps"] else 0.0,
            "mean_recall": sum(metrics_data["recalls"]) / len(metrics_data["recalls"]) if metrics_data["recalls"] else 0.0,
            "mean_p1": sum(metrics_data["p1s"]) / len(metrics_data["p1s"]) if metrics_data["p1s"] else 0.0,
            "mrr": mean_reciprocal_rank(metrics_data["r_lists"])
        }
    return aggregated_by_turn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="MedCorpus",
                        help="Name of the dataset (e.g. MedCorpus or LitSearch)")
    parser.add_argument("--gt_file", type=str, default="./data/MedCorpus/query_to_texts.jsonl",
                        help="Ground truth file in JSON Lines format")
    parser.add_argument("--rerank_pred_file", type=str, default="./results/MedCorpus/ranked.jsonl",
                        help="Predictions file after reranking (JSON Lines)")
    parser.add_argument("--unrerank_pred_file", type=str, default="./results/MedCorpus/retrieved.jsonl",
                        help="Predictions file before reranking (JSON Lines)")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for metrics (k in NDCG@k, MAP@k, etc.)")
    parser.add_argument("--personalized_queries_file", type=str, default="",
                        help="Optional: Personalized queries file (not used in this script for metrics)")
    parser.add_argument("--queries_file", type=str, default="",
                        help="Original queries file with continuity information")
    parser.add_argument("--continuity_filter", type=str, 
                        choices=["all", "continuous", "non_continuous"], 
                        default="all",  
                        help="Filter queries by continuity")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed output")
    parser.add_argument("--reranker_type", default="all", help="Reranker type (e.g. minicpm, for logging/naming)")

    args = parser.parse_args()
    
    debug = args.debug # Use the parsed debug flag
    
    print(f"Evaluating dataset: {args.dataset_name}")
    print(f"Ground truth file: {args.gt_file}")
    print(f"Reranked results file: {args.rerank_pred_file}")
    print(f"Unranked results file: {args.unrerank_pred_file}")
    print(f"Metric cutoff k={args.k}")
    print(f"Continuity filter: {args.continuity_filter}")
    
    if debug:
        print("Debug mode enabled, will show detailed information")

    # Load ground truth
    print("\nLoading ground truth...")
    ground_truth_all = load_ground_truth_jsonl(args.gt_file, debug)
    
    if not ground_truth_all:
        print("Error: Could not load ground truth, cannot continue evaluation")
        return
        
    print(f"Successfully loaded ground truth for {len(ground_truth_all)} queries")

    # Apply continuity filter if specified
    continuity_map = {}
    if args.continuity_filter != "all":
        queries_path_for_continuity = args.queries_file
        if not queries_path_for_continuity: # Try to infer if not provided
            data_dir_of_gt = os.path.dirname(args.gt_file) if args.gt_file else "."
            queries_path_for_continuity = os.path.join(data_dir_of_gt, "queries.jsonl")
        
        if os.path.exists(queries_path_for_continuity):
            print(f"\nLoading continuity information from: {queries_path_for_continuity}")
            continuity_map = load_continuity_info(queries_path_for_continuity)
            
            if continuity_map:
                filtered_gt_continuity = {}
                for qid, gt_item in ground_truth_all.items():
                    is_continuous = continuity_map.get(qid, True) # Default to True if qid not in map
                    if (args.continuity_filter == "continuous" and is_continuous) or \
                       (args.continuity_filter == "non_continuous" and not is_continuous):
                        filtered_gt_continuity[qid] = gt_item
                
                ground_truth_all = filtered_gt_continuity # Update ground_truth_all
                print(f"After continuity filtering, {len(ground_truth_all)} ground truth items remain")
        else:
            print(f"Warning: Original queries file for continuity not found: {queries_path_for_continuity}. Will use all queries from GT.")

    # Load unranked predictions
    print("\nLoading unranked prediction results...")
    # Common key for retrieved results is "results"
    unrerank_predictions_all, _ = load_predictions(args.unrerank_pred_file, "results", debug)
    if not unrerank_predictions_all: print("Warning: Could not load unranked prediction results or file is empty.")
    else: print(f"Successfully loaded unranked predictions for {len(unrerank_predictions_all)} queries")
    
    # Load reranked predictions
    print("\nLoading reranked prediction results...")
    # Common key for reranked results is "ranked_results"
    rerank_predictions_all, rerank_raw_data = load_predictions(args.rerank_pred_file, "ranked_results", debug)
    if not rerank_predictions_all: print("Warning: Could not load reranked prediction results or file is empty.")
    else: print(f"Successfully loaded reranked predictions for {len(rerank_predictions_all)} queries")
    
    # Filter predictions by qids present in (possibly continuity-filtered) ground_truth
    # This ensures we only evaluate queries for which we have ground truth.
    
    qids_for_eval = set(ground_truth_all.keys())
    
    # Filter unreranked predictions
    unrerank_predictions_eval = {qid: unrerank_predictions_all[qid] for qid in qids_for_eval if qid in unrerank_predictions_all}
    missing_in_unreranked = qids_for_eval - set(unrerank_predictions_eval.keys())
    if missing_in_unreranked:
        print(f"Warning: {len(missing_in_unreranked)} qids from ground truth are missing in unreranked predictions.")
        if debug: print(f"Missing qids (sample): {list(missing_in_unreranked)[:5]}")

    # Filter reranked predictions
    rerank_predictions_eval = {qid: rerank_predictions_all[qid] for qid in qids_for_eval if qid in rerank_predictions_all}
    missing_in_reranked = qids_for_eval - set(rerank_predictions_eval.keys())
    if missing_in_reranked:
        print(f"Warning: {len(missing_in_reranked)} qids from ground truth are missing in reranked predictions.")
        if debug: print(f"Missing qids (sample): {list(missing_in_reranked)[:5]}")

    # The ground truth for evaluation is ground_truth_all (already filtered by continuity if applicable)
    ground_truth_eval = ground_truth_all
    
    num_common_queries = len(qids_for_eval) # This is the number of GT queries we are attempting to evaluate
    print(f"\nAttempting to evaluate {num_common_queries} queries from ground truth.")
    print(f"Found {len(unrerank_predictions_eval)} matching queries in unreranked file for evaluation.")
    print(f"Found {len(rerank_predictions_eval)} matching queries in reranked file for evaluation.")

    # Evaluate unranked results (using filtered_gt and filtered_unrerank)
    print("\nEvaluating unranked results...")
    if unrerank_predictions_eval: # Only evaluate if there are predictions
        ndcg_before, map_before, mrr_before, recall_before = evaluate(unrerank_predictions_eval, ground_truth_eval, args.k, debug)
        p1_before = precision_at_1(unrerank_predictions_eval, ground_truth_eval)
    else:
        ndcg_before, map_before, mrr_before, recall_before, p1_before = 0,0,0,0,0
        print("Skipping unranked evaluation as no common queries with predictions were found.")

    # Evaluate reranked results (using filtered_gt and filtered_rerank)
    print("\nEvaluating reranked results...")
    if rerank_predictions_eval: # Only evaluate if there are predictions
        ndcg_after, map_after, mrr_after, recall_after = evaluate(rerank_predictions_eval, ground_truth_eval, args.k, debug)
        p1_after = precision_at_1(rerank_predictions_eval, ground_truth_eval)
    else:
        ndcg_after, map_after, mrr_after, recall_after, p1_after = 0,0,0,0,0
        print("Skipping reranked evaluation as no common queries with predictions were found.")

    # Calculate personalization metrics
    print("\nCalculating personalization metrics...")
    p_imp = 0.0
    p_imp_calculated = False # Flag to track if P-imp calculation was attempted and successful
    if unrerank_predictions_eval and rerank_predictions_eval:
        p_imp = calculate_p_imp(unrerank_predictions_eval, rerank_predictions_eval, ground_truth_eval, args.k)
        p_imp_calculated = True # P-imp was calculated
    else:
        print("Skipping P-imp calculation as one or both prediction sets are missing for common queries.")
    
    is_litsearch = args.dataset_name.lower() == "litsearch"

    print("\n" + "="*80)
    print(f"            Evaluation Results Summary ({args.dataset_name})            ")
    print("="*80)
    print(f"Number of queries from GT considered for evaluation: {num_common_queries}")
    print(f"  Queries evaluated (in unreranked): {len(unrerank_predictions_eval)}")
    print(f"  Queries evaluated (in reranked): {len(rerank_predictions_eval)}")

    if args.continuity_filter != "all":
        print(f"Continuity filter: {args.continuity_filter}")
    
    print("\n" + "-"*80)
    header = f"{'Metric':^15}|{'Before Reranking':^18}|{'After Reranking':^18}|{'Improvement':^15}|{'Improvement %':^15}"
    print(header)
    print("-"*80)
    
    metrics_to_display = [
        ("NDCG@"+str(args.k), ndcg_before, ndcg_after),
        ("MAP@"+str(args.k), map_before, map_after),
        ("MRR", mrr_before, mrr_after), 
        ("Recall@"+str(args.k), recall_before, recall_after),
        ("P@1", p1_before, p1_after)
    ]
    # The 'is_litsearch' check here was a placeholder, actual P-imp display is handled below
    # if is_litsearch: 
    #     pass 

    for name, before_val, after_val in metrics_to_display:
        improvement = after_val - before_val
        # Handle division by zero for percentage improvement
        if abs(before_val) < 1e-9: # Effectively zero
            percent_improvement_str = "N/A (Before=0)" if abs(improvement) > 1e-9 else "0.00%"
        else:
            percent_improvement_str = f"{((improvement)/before_val*100):^+15.2f}%"
        print(f"{name:^15}|{before_val:^18.4f}|{after_val:^18.4f}|{improvement:^+15.4f}|{percent_improvement_str:^15}")

    # Display P-imp metric
    if p_imp_calculated:
        print(f"{'P-imp':^15}|{'N/A':^18}|{p_imp:^18.4f}|{'N/A':^15}|{'N/A':^15}")
    elif is_litsearch: # If P-imp not calculated, but it's LitSearch, show N/A row
        print(f"{'P-imp':^15}|{'N/A':^18}|{'N/A (not calc.)':^18}|{'N/A':^15}|{'N/A':^15}")
    print("-"*80)
    
    print("\nKey Metrics Summary (After Reranking):")
    print(f"- NDCG@{args.k}: {ndcg_after:.4f} (vs Before: {ndcg_before:.4f}, Change: {ndcg_after-ndcg_before:+.4f})")
    print(f"- MAP@{args.k}: {map_after:.4f} (vs Before: {map_before:.4f}, Change: {map_after-map_before:+.4f})")
    print(f"- MRR: {mrr_after:.4f} (vs Before: {mrr_before:.4f}, Change: {mrr_after-mrr_before:+.4f})")
    print(f"- Recall@{args.k}: {recall_after:.4f} (vs Before: {recall_before:.4f}, Change: {recall_after-recall_before:+.4f})")
    print(f"- P@1: {p1_after:.4f} (vs Before: {p1_before:.4f}, Change: {p1_after-p1_before:+.4f})")
    
    # Display P-imp in key metrics summary
    if p_imp_calculated:
        print(f"- P-imp (Pairwise Improvement Rate): {p_imp:.4f}")
    elif is_litsearch: # If P-imp not calculated, but it's LitSearch, show N/A
        print(f"- P-imp (Pairwise Improvement Rate): N/A (not calculated)")
    
    if args.personalized_queries_file and os.path.exists(args.personalized_queries_file):
        # Note: This file isn't directly used for metric calculation in this script,
        # but its presence might be relevant for context if personalized features were used by the reranker.
        print(f"\nPersonalized queries file detected: {args.personalized_queries_file}")
        if rerank_raw_data: # Check if rerank_raw_data was loaded
             print(f"Reranked data loaded for {len(rerank_raw_data)} queries, potentially using personalization.")
    
    print("\n" + "="*80)

    is_conversation_dataset = False
    # Check if any query ID in the (filtered) ground truth contains an underscore, common for conversational qids
    if ground_truth_eval and any("_" in qid for qid in ground_truth_eval.keys()):
        is_conversation_dataset = True
        print("Detected conversational search dataset (based on query ID format), proceeding with conversation evaluation.")
    
    if is_conversation_dataset and rerank_predictions_eval: # Only if reranked predictions exist
        print("\nEvaluating reranked results by conversation grouping...")
        # Use rerank_predictions_eval and ground_truth_eval for conversational analysis
        conv_eval_results = evaluate_by_conversation(rerank_predictions_eval, ground_truth_eval, args.k, debug)
        
        print("\n" + "="*40)
        print("     Conversation Evaluation Results (Reranked)     ")
        print("="*40)
        
        overall_conv = conv_eval_results["overall"]
        print(f"Total conversations: {overall_conv['topic_count']}")
        print(f"Total queries: {overall_conv['query_count']}")
        print(f"Average NDCG@{args.k}: {overall_conv['mean_ndcg']:.4f}")
        print(f"Average MAP@{args.k}: {overall_conv['mean_map']:.4f}")
        print(f"Average Recall@{args.k}: {overall_conv['mean_recall']:.4f}")
        print(f"Average P@1: {overall_conv['mean_p1']:.4f}")
        print(f"Overall MRR: {overall_conv['mrr']:.4f}") # Added MRR

        print("\nAnalysis by Turn (Reranked):")
        for turn_id, metrics_data in sorted(conv_eval_results["by_turn"].items()):
            print(f"Turn {turn_id}: NDCG@{args.k}={metrics_data['mean_ndcg']:.4f}, " +
                  f"MAP@{args.k}={metrics_data['mean_map']:.4f}, " +
                  f"Recall@{args.k}={metrics_data['mean_recall']:.4f}, " +
                  f"P@1={metrics_data['mean_p1']:.4f}, " +
                  f"MRR={metrics_data['mrr']:.4f} " + 
                  f"(Queries: {metrics_data['query_count']})")
    elif is_conversation_dataset:
        print("\nSkipping conversational evaluation as no reranked predictions were available for common queries.")

    print("\n" + "="*80)
    print("Evaluation script finished.")

if __name__ == "__main__":
    main()
