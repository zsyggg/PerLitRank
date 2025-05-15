# cognitive_retrieval.py (Stage 1: Cognitive Feature Extraction)
import json
import os
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
import torch
import gc
from collections import defaultdict
import time

try:
    from memory_system import CognitiveMemorySystem
except ImportError:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_fallback = logging.getLogger('CognitiveRetrieval_Fallback_MS')
    logger_fallback.error("Failed to import CognitiveMemorySystem.")
    class CognitiveMemorySystem: # Dummy
        def __init__(self, config=None): pass
        def process_query(self, query, user_id, clicks, topic_id=None): # Added topic_id for dummy
            return {
                "sequential_results_raw": {"sequential_continuity": {}, "sequential_terminology": {}},
                "working_memory_state_raw": {},
                "long_term_memory_results_raw": {}
            }
        def get_tagged_features(self, results, components, continuity_score_override=None): return []

try:
    from utils import Query, get_config, logger, load_queries
except ImportError: # Fallback
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_fallback = logging.getLogger('CognitiveRetrieval_Fallback_Utils')
    logger = logger_fallback; logger.error("Failed to import from utils.py.")
    from dataclasses import dataclass, field
    @dataclass
    class Query: query_id: str; query: str; topic_id: str = ""; turn_id: int = 0; continuity: bool = False;
    class DummyConfig: # Simplified
        device="cpu"; llm_device="cpu"; results_dir="."; dataset_name="dummy";
        cognitive_features_detailed_path = "results/dummy/cognitive_features_detailed.jsonl";
        memory_components=["sequential", "working", "long"]; continuity_threshold=0.3;
        feature_extractor='keybert'; memory_type='vector'; queries_path='queries.jsonl';
        dataset_type='unknown'; # Added dataset_type
        def __getattr__(self, name): return None
    def get_config(): return DummyConfig()
    def load_queries(config): return []


class CognitiveFeatureExtractor:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.dataset_type = getattr(self.config, 'dataset_type', 'unknown')
        logger.info(f"CognitiveFeatureExtractor initializing for Stage 1.")
        try:
             self.memory_system = CognitiveMemorySystem(config=self.config)
             logger.info("CognitiveMemorySystem initialized for feature extraction.")
        except Exception as e: logger.error(f"Failed to init CognitiveMemorySystem: {e}", exc_info=True); raise
        logger.info(f"CognitiveFeatureExtractor ready. Extractor: {getattr(self.config, 'feature_extractor', 'N/A')}, Memory: {getattr(self.config, 'memory_type', 'N/A')}")

    def extract_features_for_query(self, query_obj: Query, user_id: str) -> Dict:
        """
        Processes a single query object to extract detailed cognitive features.
        """
        query_id_str = getattr(query_obj, 'query_id', 'UNKNOWN_QUERY_ID')
        query_text = getattr(query_obj, 'query', '')
        topic_id = getattr(query_obj, 'topic_id', None) # Get topic_id from Query object
        turn_id = getattr(query_obj, 'turn_id', 0)

        logger.debug(f"Extracting cognitive features for query: {query_id_str} (Memory User ID: {user_id}, Topic: {topic_id}, Turn: {turn_id})")
        try:
            # Pass topic_id to process_query for potential WM reset
            raw_memory_results = self.memory_system.process_query(
                query_text,
                user_id,
                [], # clicked_docs, assuming none for this stage or pass if available
                topic_id=topic_id
            )
            if not isinstance(raw_memory_results, dict):
                logger.warning(f"Memory system returned non-dict for {query_id_str}: {type(raw_memory_results)}. Using empty.")
                raw_memory_results = {
                    "sequential_results_raw": {}, "working_memory_state_raw": {}, "long_term_memory_results_raw": {}
                }

            seq_res_raw = raw_memory_results.get("sequential_results_raw", {})
            cont_info = seq_res_raw.get("sequential_continuity", {})
            cont_score = cont_info.get("confidence", 0.0)
            # Use continuity from query_obj if pre-calculated, otherwise from SM
            is_continuous = getattr(query_obj, 'continuity', cont_info.get("detected", False))


            active_mem_comp = getattr(self.config, 'memory_components', ["sequential", "working", "long"])
            # MODIFICATION: get_tagged_features now extracts simplified keywords
            tagged_keywords_list = self.memory_system.get_tagged_features(
                raw_memory_results, active_mem_comp, cont_score # Pass cont_score for SM to use
            )

            detailed_features = {
                "query_id": query_id_str,
                "query": query_text,
                "topic_id": topic_id if topic_id else getattr(query_obj, 'topic_id', ''), # Ensure topic_id is stored
                "turn_id": turn_id,
                "continuity": is_continuous, # Store the determined continuity
                "continuity_score": cont_score,
                "high_continuity_terms": cont_info.get("high_continuity_terms", []), # From SM
                "tagged_memory_features": tagged_keywords_list, # MODIFIED: This now contains simplified keywords
                "sequential_results_raw": seq_res_raw,
                "working_memory_state_raw": raw_memory_results.get("working_memory_state_raw", {}),
                "long_term_memory_results_raw": raw_memory_results.get("long_term_memory_results_raw", {})
            }
            return detailed_features
        except Exception as e:
            logger.error(f"Error extracting cognitive features for query {query_id_str}: {e}", exc_info=True)
            return {
                "query_id": query_id_str, "query": query_text,
                "tagged_memory_features": [], "error": str(e),
                "topic_id": topic_id if topic_id else getattr(query_obj, 'topic_id', ''),
                "turn_id": turn_id,
                "continuity": getattr(query_obj, 'continuity', False) # Fallback continuity
            }

    def batch_extract_cognitive_features(self, queries: List[Query]) -> List[Dict]:
        """
        Processes a batch of queries to extract and save detailed cognitive features.
        Handles MedCorpus specific logic: process all turns for memory, save features for last turn.
        """
        all_extracted_features_to_save = []
        output_file_path = self.config.cognitive_features_detailed_path
        
        logger.info(f"Starting batch cognitive feature extraction for {len(queries)} queries.")
        logger.info(f"Output detailed cognitive features to: {output_file_path}")

        if output_file_path:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f: pass # Clear file
                logger.info(f"Cleared cognitive features output file: {output_file_path}")
            except IOError as e:
                logger.error(f"Could not clear/create output file {output_file_path}: {e}")
                return [] # Cannot proceed if output file is not writable
        
        save_every_n = 50
        processed_for_saving_count = 0
        current_batch_to_save = []
        total_queries_memory_updated = 0

        # Group queries by topic_id for session-aware processing (especially for WM reset)
        # The user_id for memory system will be based on topic_id to maintain session context.
        topic_groups = defaultdict(list)
        for q_obj in queries:
            # Query.__post_init__ should have populated topic_id and turn_id
            topic_id_for_group = getattr(q_obj, 'topic_id', None)
            if not topic_id_for_group: # Fallback if topic_id is somehow not set
                 topic_id_for_group = str(getattr(q_obj, 'query_id', 'unknown_topic')).split("_")[0]
            topic_groups[topic_id_for_group].append(q_obj)

        logger.info(f"Grouped into {len(topic_groups)} sessions/topics for feature extraction.")

        for topic_id_key, topic_queries_list in tqdm(topic_groups.items(), desc="Extracting Cognitive Features per Topic"):
            # Sort queries within a topic by turn_id to process them in order
            sorted_topic_queries = sorted(topic_queries_list, key=lambda q_sort: getattr(q_sort, 'turn_id', 0))
            if not sorted_topic_queries: continue

            user_memory_id_for_session = f"user_session_{topic_id_key}" # Consistent memory ID for the whole session/topic

            # For MedCorpus, only save features of the last turn (e.g., turn_id _3 if 0-indexed)
            # For other datasets, save features for all queries.
            save_only_last_turn = (self.dataset_type == "medcorpus")
            
            last_query_obj_in_session = sorted_topic_queries[-1] if save_only_last_turn else None

            for query_obj_to_process in sorted_topic_queries:
                total_queries_memory_updated += 1
                q_id_log = getattr(query_obj_to_process, 'query_id', 'N/A_ID')
                
                # Extract features. This call also updates the memory state for user_memory_id_for_session.
                # The topic_id from query_obj_to_process is passed to handle WM reset if it's a new topic.
                extracted_data = self.extract_features_for_query(query_obj_to_process, user_memory_id_for_session)

                # Determine if we should save features for this specific query
                should_save_features = False
                if save_only_last_turn:
                    if query_obj_to_process.query_id == last_query_obj_in_session.query_id:
                        should_save_features = True
                        logger.info(f"MedCorpus Session {topic_id_key}, LAST Turn {getattr(query_obj_to_process, 'turn_id', 'N/A')}: Preparing to save features for '{getattr(query_obj_to_process, 'query', '')[:30]}...'")
                    else:
                        logger.debug(f"MedCorpus Session {topic_id_key}, Turn {getattr(query_obj_to_process, 'turn_id', 'N/A')}: Memory updated for '{getattr(query_obj_to_process, 'query', '')[:30]}...' (features not saved for this turn).")
                else: # Not MedCorpus, or MedCorpus but saving all turns (if logic changes)
                    should_save_features = True
                    logger.debug(f"{self.dataset_type} Query {q_id_log}: Preparing to save features.")


                if should_save_features:
                    if extracted_data and not extracted_data.get("error"):
                        all_extracted_features_to_save.append(extracted_data)
                        current_batch_to_save.append(extracted_data)
                        processed_for_saving_count += 1
                        if processed_for_saving_count % save_every_n == 0 and current_batch_to_save:
                            self._save_features_batch(current_batch_to_save, output_file_path)
                            current_batch_to_save = []
                    else:
                         logger.warning(f"Skipping save for query {q_id_log} (marked for saving) due to feature extraction error or no data.")
                
                self._cleanup_memory_iteration(q_id_log)

        if current_batch_to_save:
            self._save_features_batch(current_batch_to_save, output_file_path)

        logger.info(f"Cognitive feature extraction complete. Total queries processed for memory updates: {total_queries_memory_updated}. Features saved for: {processed_for_saving_count} queries to {output_file_path}.")
        return all_extracted_features_to_save


    def _save_features_batch(self, features_batch: List[Dict], output_file_path: str):
        if not features_batch or not output_file_path: return
        try:
            with open(output_file_path, 'a', encoding='utf-8') as f_out:
                for feature_set in features_batch:
                    f_out.write(json.dumps(feature_set, ensure_ascii=False) + "\n")
            logger.debug(f"Saved batch of {len(features_batch)} cognitive feature sets to {output_file_path}")
        except Exception as e:
            logger.error(f"Error writing cognitive features batch to {output_file_path}: {e}", exc_info=True)

    def _cleanup_memory_iteration(self, query_id_for_log: str):
        logger.debug(f"Post-query cleanup for {query_id_for_log}")
        gc.collect()
        if torch.cuda.is_available():
            try: torch.cuda.empty_cache()
            except: pass

def main():
    config = get_config()
    # Ensure logger is configured, especially if run standalone
    if not logger.handlers: # Check if handlers are already attached
        log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(), logging.FileHandler('perslitrank_stage1.log')])
        # Update the global logger instance if it was a fallback
        if logger.name == 'CognitiveRetrieval_Fallback_Utils' or logger.name == 'PersLitRank':
             globals()['logger'] = logging.getLogger('PersLitRank_Stage1')


    logger.info("--- Running Cognitive Feature Extraction (Stage 1) Standalone ---")
    logger.info(f"Dataset: {config.dataset_name}, Type: {config.dataset_type}")
    logger.info(f"Output will be saved to: {config.cognitive_features_detailed_path}")

    queries = load_queries(config)
    if not queries:
        logger.error(f"No queries loaded from {config.queries_path}. Exiting Stage 1.")
        return
    logger.info(f"Loaded {len(queries)} original queries for Stage 1 processing.")

    try:
        feature_extractor_instance = CognitiveFeatureExtractor(config)
    except Exception as e:
         logger.error(f"Failed to initialize CognitiveFeatureExtractor: {e}", exc_info=True)
         return

    start_time = time.time()
    feature_extractor_instance.batch_extract_cognitive_features(queries)
    end_time = time.time()

    logger.info(f"Stage 1 (Cognitive Feature Extraction) complete. Total time: {end_time - start_time:.2f}s.")
    logger.info(f"Detailed cognitive features saved to: {config.cognitive_features_detailed_path}")

if __name__ == "__main__":
    main()
