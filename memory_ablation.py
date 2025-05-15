#!/usr/bin/env python
# memory_ablation.py
"""
Memory ablation experiment.
Reads detailed cognitive features, filters them based on ablation settings,
generates an "abated" personalized narrative using PersonalizedGenerator,
and then reranks documents using this ablated narrative.
"""
import os
import json
import argparse
import logging
import torch
import re # Not strictly used in this version's reranker prompt, but kept for potential future use
import gc
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Assuming utils.py and personalized_generator.py are in the python path
try:
    from utils import get_config, logger as ablation_logger # Use a distinct logger name if needed
    from personalized_generator import PersonalizedGenerator
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ablation_logger = logging.getLogger("MemoryAblation_Fallback")
    ablation_logger.error(f"Failed to import necessary modules: {e}. Using fallback.")
    # Define dummy get_config and PersonalizedGenerator if imports fail
    class DummyConfig:
        device="cpu"; reranker_path=None; personalized_text_target_length=200;
        reranker_type="gte"; local_model_path="dummy_llm_path"; llm_device="cpu"; # Added llm_device
        # Add other necessary attributes that PersonalizedGenerator might expect from config
        local_model_tokenizer=None; local_model_dtype="float16"; local_model_max_tokens=512;
        local_model_temperature=0.7; local_model_top_p=0.8; local_model_top_k=20;
        local_model_presence_penalty=None; local_model_repetition_penalty=1.0; enable_thinking=False;
        def _update_text_length_constraints(self): pass # Dummy method
        def __getattr__(self, name): return None
    def get_config(): return DummyConfig()
    class PersonalizedGenerator:
        def __init__(self, config=None): 
            self.config = config or DummyConfig()
            self.model = None # Indicate model is not loaded for dummy
            self.tokenizer = None
            ablation_logger.info("Using Dummy PersonalizedGenerator")
        def generate_personalized_text(self, query, memory_results, excluded_memory=None): 
            return f"Ablated narrative for {query[:30]}... (target length {self.config.personalized_text_target_length})"

# Use the logger from utils if available, otherwise the fallback
logger = ablation_logger


# --- Reranker Factory and Formatting (Simplified for ablated narrative) ---
DEFAULT_RERANKER_PATHS = {
    "gte": "/workspace/.cache/modelscope/hub/iic/gte_passage-ranking_multilingual-base",
    "jina": "/workspace/.cache/modelscope/hub/jinaai/jina-reranker-v2-base-multilingual",
    "minicpm": "/workspace/.cache/modelscope/hub/OpenBMB/MiniCPM-Reranker-Light"
}

def get_reranker_model_and_tokenizer(reranker_type: str, model_path: Optional[str], device: str, use_flash_attention: bool = False):
    actual_path = model_path or DEFAULT_RERANKER_PATHS.get(reranker_type)
    if not actual_path:
        raise ValueError(f"Path for reranker '{reranker_type}' not found.")
    logger.info(f"Loading {reranker_type} model: {actual_path} to {device}. FA2: {use_flash_attention and reranker_type == 'minicpm'}")
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    try:
        trust_remote = reranker_type in ["jina", "minicpm"] # GTE might not need it
        tokenizer = AutoTokenizer.from_pretrained(actual_path, trust_remote_code=trust_remote, padding_side="right")
        model_kwargs = {"trust_remote_code": trust_remote, "torch_dtype": torch.float16 if torch.cuda.is_available() and "cuda" in device else torch.float32}
        if use_flash_attention and reranker_type == 'minicpm' and torch.cuda.is_available() and "cuda" in device : 
            model_kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForSequenceClassification.from_pretrained(actual_path, **model_kwargs).to(device).eval()
        return model, tokenizer
    except Exception as e: logger.error(f"Failed to load model {actual_path}: {e}", exc_info=True); raise


class AblatedRerankerPromptFormatter:
    def format_input(self, query: str,
                     ablated_narrative: Optional[str], 
                     document_text_dict: Dict[str, Any],
                     reranker_type: str):
        doc_content = " ".join(filter(None, [document_text_dict.get("title",""), document_text_dict.get("text",""), document_text_dict.get("full_paper","")])).strip().replace("\n"," ")
        
        is_personalized_mode = bool(ablated_narrative) # True if narrative is present

        if reranker_type == "jina":
            query_part = f"Query: {query}"
            if is_personalized_mode:
                query_part += f" User Background: {ablated_narrative}"
            return (query_part.strip(), doc_content)
        else: # GTE / MiniCPM
            if is_personalized_mode:
                template = self._get_personalized_template()
                formatted_text = template.format(query=query, personalized_features=ablated_narrative, document_text=doc_content)
            else: 
                template = self._get_baseline_template()
                formatted_text = template.format(query=query, document_text=doc_content)

            if reranker_type == "minicpm":
                instruction = "Evaluate document relevance for the query."
                if is_personalized_mode:
                    instruction = "Considering user background, evaluate document relevance for the query."
                    # For MiniCPM, ensure User Background is clearly part of the input if present
                    formatted_text = f"<s>Instruction: {instruction}\nQuery: {query}\nUser Background: {ablated_narrative}\nDocument: {doc_content}</s>"
                else:
                    formatted_text = f"<s>Instruction: {instruction}\nQuery: {query}\nDocument: {doc_content}</s>"
            return formatted_text

    def _get_personalized_template(self):
        return """Task: Evaluate document relevance for the query, considering the user's background and interests.
Aspects: 1. Technical relevance. 2. Alignment with user's profile. 3. Usefulness.

Query: {query}
User Background and Interests: 
{personalized_features}
Document: {document_text}"""

    def _get_baseline_template(self):
        return "Task: Evaluate document relevance for the query based on content.\n\nQuery: {query}\n\nDocument: {document_text}"


def batch_rerank_documents_ablation(
    reranker_type: str, model, tokenizer,
    query: str, ablated_narrative: Optional[str], 
    docs: List[Dict], device: str, batch_size=4, max_length=512
) -> List[Dict]:
    results = []
    formatter = AblatedRerankerPromptFormatter()
    for i in range(0, len(docs), batch_size):
        batch_docs_data = docs[i:i+batch_size]
        batch_formatted_inputs = []
        original_docs_for_batch = []
        for doc_data in batch_docs_data:
            # Pass the whole doc_data dictionary to the formatter
            formatted_input = formatter.format_input(query, ablated_narrative, doc_data, reranker_type)
            if formatted_input:
                batch_formatted_inputs.append(formatted_input)
                original_docs_for_batch.append(doc_data)
        
        if not batch_formatted_inputs: continue
        try:
            with torch.no_grad():
                if reranker_type == "jina":
                    scores = model.compute_score(batch_formatted_inputs, max_length=max_length)
                else:
                    inputs = tokenizer(batch_formatted_inputs, padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(device)
                    outputs = model(**inputs, return_dict=True)
                    scores = outputs.logits.view(-1).float().cpu().numpy()
            for j, doc_d in enumerate(original_docs_for_batch):
                score_val = float(scores[j]) if j < len(scores) else 0.0
                res = {"text_id":doc_d.get("text_id",""), "title":doc_d.get("title",""), "text":doc_d.get("text",""), "score":score_val}
                if "full_paper" in doc_d and doc_d.get("full_paper"): res["full_paper"] = doc_d["full_paper"]
                results.append(res)
        except Exception as e:
            logger.error(f"Batch rerank error (ablation, type {reranker_type}): {e}", exc_info=True)
            for doc_d in original_docs_for_batch: # Fallback
                 res = {"text_id":doc_d.get("text_id",""), "title":doc_d.get("title",""), "text":doc_d.get("text",""), "score":doc_d.get("score",0.0)}
                 if "full_paper" in doc_d and doc_d.get("full_paper"): res["full_paper"] = doc_d["full_paper"]
                 results.append(res)
    return sorted(results, key=lambda x: x['score'], reverse=True)

# --- Helper Functions (Filtering, Loading) ---
def filter_memory_features(all_features: List[str], exclude_memory_type: Optional[str]) -> List[str]:
    if exclude_memory_type is None or exclude_memory_type == "none" or not all_features:
        return all_features # Return all if no exclusion or no features
    
    filtered = []
    # Define tags consistently with memory_system.py or how they appear in cognitive_features_detailed.jsonl
    TAG_MAP = {
        "sequential": "[SEQUENTIAL_MEMORY]",
        "working": "[WORKING_MEMORY]",
        "long_explicit": "[LONG_EXPLICIT]", # Assuming this is how it's tagged
        "long_implicit": "[LONG_IMPLICIT]"  # Assuming this is how it's tagged
    }
    
    tags_to_exclude_prefixes = []
    if exclude_memory_type == "long":
        tags_to_exclude_prefixes.extend([TAG_MAP["long_explicit"], TAG_MAP["long_implicit"]])
    elif exclude_memory_type in TAG_MAP:
        tags_to_exclude_prefixes.append(TAG_MAP[exclude_memory_type])
    else: # Unknown exclusion type, or "none" was already handled
        return all_features

    if not tags_to_exclude_prefixes: # Should not happen if exclude_memory_type is valid and not "none"
        return all_features

    for feature_str in all_features:
        is_excluded = any(feature_str.startswith(prefix) for prefix in tags_to_exclude_prefixes)
        if not is_excluded:
            filtered.append(feature_str)
            
    logger.debug(f"Ablation: Excluded '{exclude_memory_type}'. Kept {len(filtered)}/{len(all_features)} tagged features.")
    return filtered

def load_cognitive_and_retrieved_data(
    cognitive_features_path: str, 
    retrieved_results_path: str, 
    continuity_map: Optional[Dict[str, bool]] = None, 
    continuity_filter: str = "all"
) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
    cognitive_data_loaded = {}
    retrieved_data_loaded = {}
    query_ids_to_keep = set()

    if not os.path.exists(cognitive_features_path):
        logger.error(f"Cognitive features file not found: {cognitive_features_path}"); return {}, {}
    try:
        with open(cognitive_features_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    if not query_id: continue
                    
                    keep = True
                    if continuity_filter != "all" and continuity_map is not None:
                        # Use continuity from cognitive_data if available, else from continuity_map
                        query_continuity = data.get("continuity", continuity_map.get(query_id, True))
                        if (continuity_filter == "continuous" and not query_continuity) or \
                           (continuity_filter == "non_continuous" and query_continuity):
                            keep = False
                    
                    if keep:
                        cognitive_data_loaded[query_id] = data 
                        query_ids_to_keep.add(query_id)
                except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON in {cognitive_features_path}: {line.strip()}")
    except Exception as e: logger.error(f"Error loading {cognitive_features_path}: {e}"); return {}, {}
    logger.info(f"Loaded {len(cognitive_data_loaded)} cognitive entries (filter: {continuity_filter}).")

    if not os.path.exists(retrieved_results_path):
        logger.error(f"Retrieved results file not found: {retrieved_results_path}"); return {}, {}
    try:
        with open(retrieved_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    if query_id in query_ids_to_keep: # Only load if query is in the filtered set
                         retrieved_data_loaded[query_id] = data.get("results", [])
                except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON in {retrieved_results_path}: {line.strip()}")
    except Exception as e: logger.error(f"Error loading {retrieved_results_path}: {e}"); return {}, {}
    
    # Ensure final cognitive_data only contains qids present in retrieved_data_loaded
    final_cognitive_data = {qid: data for qid, data in cognitive_data_loaded.items() if qid in retrieved_data_loaded}
    if len(final_cognitive_data) < len(cognitive_data_loaded):
        logger.info(f"Final common queries for ablation after matching with retrieved: {len(final_cognitive_data)}")
    
    return final_cognitive_data, retrieved_data_loaded

def load_original_queries_continuity(file_path: str) -> Dict[str, bool]:
    continuity_map = {}
    if not os.path.exists(file_path):
        logger.warning(f"Original queries file for continuity info not found: {file_path}"); return continuity_map
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    # Expect 'continuity' boolean field in queries.jsonl
                    is_continuous = data.get("continuity", False) # Default to False if not specified
                    if query_id: continuity_map[query_id] = is_continuous
                except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON in original queries file: {line.strip()}")
        logger.info(f"Loaded continuity info for {len(continuity_map)} queries from {file_path}")
    except Exception as e: logger.error(f"Error loading original queries {file_path}: {e}")
    return continuity_map

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Memory Ablation Experiment with Narrative Generation")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["CORAL", "LitSearch", "MedCorpus"])
    parser.add_argument("--reranker_type", type=str, default="minicpm", choices=["gte", "jina", "minicpm"])
    parser.add_argument("--reranker_path", type=str, help="Optional: Explicit path to the reranker model")
    parser.add_argument("--exclude_memory", type=str, default="none", choices=["sequential", "working", "long", "none"],
                        help="Memory component to exclude for narrative generation. 'none' uses all available features.")
    
    # Input file paths
    parser.add_argument("--cognitive_features_input_path", type=str, required=True, help="Path to cognitive_features_detailed.jsonl")
    parser.add_argument("--retrieved_results_input_path", type=str, required=True, help="Path to retrieved.jsonl")
    parser.add_argument("--original_queries_path", type=str, required=True, help="Path to original queries.jsonl (for continuity info)")
    
    # Output path for this ablation run's reranked results
    parser.add_argument("--output_path", type=str, required=True, help="Output path for reranked results of this specific ablation run")
    
    # LLM and Narrative Generation params (these will usually be set globally by run.py when calling this script)
    # However, providing them here allows standalone testing or overriding.
    parser.add_argument("--personalized_text_target_length", type=int, help="Target length for ablated narrative (uses global config if not set here)")
    parser.add_argument("--local_model_path", type=str, help="Path to LLM for PersonalizedGenerator (uses global config if not set)")


    # Common operational params
    parser.add_argument("--use_flash_attention", action="store_true", help="Enable Flash Attention 2 for MiniCPM")
    parser.add_argument("--batch_size", type=int, default=8, help="Reranking batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for reranker")
    parser.add_argument("--top_k", type=int, default=10, help="Number of final documents to save after reranking")
    parser.add_argument("--initial_top_k", type=int, default=50, help="Number of candidates from retrieval to rerank")
    parser.add_argument("--continuity_filter", type=str, default="all", choices=["all", "continuous", "non_continuous"],
                        help="Filter queries by continuity before processing")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID for reranker. LLM uses config.llm_device.")
    args = parser.parse_args()

    # --- Setup Config and Device ---
    config = get_config() 
    # Update global config with relevant args from this script, so PersonalizedGenerator picks them up
    # This is important if run_memory_ablation.sh passes different values than the main run.py
    if args.local_model_path: config.local_model_path = args.local_model_path
    if args.personalized_text_target_length:
        config.personalized_text_target_length = args.personalized_text_target_length
        config.length_suffix = f"_L{config.personalized_text_target_length}"
        config._update_text_length_constraints() # Update min/max based on new target
    # Note: LLM device (config.llm_device) is typically set by run.py's --gpu_id.
    # If memory_ablation.py needs to run LLM on a different GPU than reranker,
    # config.llm_device would need to be settable here too. For now, assume it's inherited.

    # Reranker device setup
    reranker_device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id < torch.cuda.device_count() else "cpu"
    if "cuda" in reranker_device and args.gpu_id >= torch.cuda.device_count():
        logger.warning(f"Reranker GPU ID {args.gpu_id} invalid, using CPU for reranker.")
        reranker_device = "cpu"
    logger.info(f"Reranker will use device: {reranker_device}. LLM for narrative generation will use device: {config.llm_device}")

    logger.info(f"--- Memory Ablation Run ---")
    logger.info(f"Dataset: {args.dataset_name}, Reranker: {args.reranker_type}, Exclude: {args.exclude_memory}, Continuity: {args.continuity_filter}")
    logger.info(f"Output to: {args.output_path}, Ablated Narrative Target Length: {config.personalized_text_target_length}")

    continuity_map = load_original_queries_continuity(args.original_queries_path)
    cognitive_data, retrieved_results = load_cognitive_and_retrieved_data(
        args.cognitive_features_input_path, args.retrieved_results_input_path,
        continuity_map, args.continuity_filter
    )

    if not cognitive_data or not retrieved_results:
        logger.error("Failed to load input data. Exiting ablation run."); return

    try: 
        reranker_model, reranker_tokenizer = get_reranker_model_and_tokenizer(
            args.reranker_type, args.reranker_path, reranker_device, args.use_flash_attention
        )
    except Exception as e: logger.error(f"Failed to init reranker: {e}. Exiting."); return

    try: 
        # PersonalizedGenerator will use LLM settings from the global config object
        narrative_generator = PersonalizedGenerator(config=config) 
        if narrative_generator.model is None: 
            raise RuntimeError("PersonalizedGenerator LLM failed to load. Check LLM path and config.")
    except Exception as e: logger.error(f"Failed to init PersonalizedGenerator: {e}. Exiting."); return

    final_output_for_file = []
    query_ids_to_process = list(cognitive_data.keys()) # Already filtered by continuity

    for query_id in tqdm(query_ids_to_process, desc=f"Ablation ({args.reranker_type}, {args.exclude_memory}, {args.continuity_filter})"):
        if query_id not in retrieved_results: 
            logger.warning(f"Query {query_id} missing from retrieved results. Skipping.")
            continue

        current_cognitive_data = cognitive_data[query_id]
        original_query_text = current_cognitive_data["query"]
        all_tagged_features = current_cognitive_data.get("tagged_memory_features", [])
        
        filtered_tagged_features = filter_memory_features(all_tagged_features, args.exclude_memory)
        
        memory_input_for_generator = {"tagged_memory_features": filtered_tagged_features}
        # Generate the (potentially ablated) narrative
        ablated_narrative = narrative_generator.generate_personalized_text(
            query=original_query_text,
            memory_results=memory_input_for_generator
        )
        if "Error" in ablated_narrative: # Handle generation errors
            logger.warning(f"Narrative generation error for {query_id}. Using empty narrative for reranking.")
            ablated_narrative = "" 

        candidate_docs_to_rerank = retrieved_results[query_id][:args.initial_top_k]

        reranked_docs_list = batch_rerank_documents_ablation(
            args.reranker_type, reranker_model, reranker_tokenizer,
            original_query_text, ablated_narrative, 
            candidate_docs_to_rerank, 
            reranker_device, args.batch_size, args.max_length
        )
        final_top_docs = reranked_docs_list[:args.top_k]

        result_entry = {
            "query_id": query_id, "query": original_query_text,
            "ablated_personalized_narrative": ablated_narrative, 
            "used_filtered_tagged_features_for_narrative": filtered_tagged_features, 
            "continuity": current_cognitive_data.get("continuity", True), # From cognitive_features_detailed
            "ranked_results": final_top_docs
        }
        final_output_for_file.append(result_entry)

    logger.info(f"Saving {len(final_output_for_file)} results to: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    try:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            for res_item in final_output_for_file:
                f.write(json.dumps(res_item, ensure_ascii=False) + '\n')
        logger.info("Ablation run and saving completed successfully.")
    except IOError as e: logger.error(f"Failed to write results to {args.output_path}: {e}")

    del reranker_model, reranker_tokenizer, narrative_generator, cognitive_data, retrieved_results
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()
