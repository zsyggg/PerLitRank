# memory_system.py
import json
from typing import Dict, List, Any, Optional, Tuple, Set
import re
from datetime import datetime
import numpy as np
from collections import defaultdict
import os

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from utils import FeatureExtractorRegistry, MemorySystemRegistry, logger, get_config
from sentence_transformers import SentenceTransformer
import torch
import gc

_shared_embedding_model = None
_embedding_model_device = "cpu"

SEQUENTIAL_MEMORY_TAG = "[SEQUENTIAL_MEMORY]"
WORKING_MEMORY_TAG = "[WORKING_MEMORY]"
LONG_EXPLICIT_TAG = "[LONG_EXPLICIT]"
# LONG_IMPLICIT_TAG = "[LONG_IMPLICIT]" # Not actively used for keyword output in this version

@FeatureExtractorRegistry.register('keybert')
class KeyBERTExtractor:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KeyBERTExtractor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name=None, config=None):
        if self._initialized: return
        self.config = config or get_config()
        try:
            from keybert import KeyBERT
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            keybert_embedder_device = getattr(self.config, 'keybert_embedder_device', 'cpu')
            effective_model_name = model_name or getattr(self.config, 'keybert_model', 'all-MiniLM-L6-v2')
            if effective_model_name and (os.path.isdir(effective_model_name) or "gte" in effective_model_name.lower() or "sentence-transformer" in effective_model_name.lower()):
                trust_remote = "gte" in effective_model_name.lower() or "modelscope" in effective_model_name.lower()
                keybert_sentence_model = SentenceTransformer(effective_model_name, device=keybert_embedder_device, trust_remote_code=trust_remote)
                self.model = KeyBERT(model=keybert_sentence_model)
            else:
                self.model = KeyBERT(model=effective_model_name)
            self._initialized = True
            logger.info(f"KeyBERT initialized with '{effective_model_name}' on {keybert_embedder_device if 'keybert_sentence_model' in locals() else 'default KeyBERT device'}.")
        except ImportError: logger.error("KeyBERT not found.", exc_info=True); self._initialized = False; raise
        except Exception as e: logger.error(f"Error initializing KeyBERT: {e}", exc_info=True); self._initialized = False; raise

    def extract_terms(self, text: str, top_n=10) -> List[Tuple[str, float]]:
        if not self._initialized: return []
        if not text or len(text.strip()) < 5: return []
        try:
            return self.model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', use_mmr=True, diversity=0.7, top_n=top_n) or []
        except Exception as e: logger.error(f"KeyBERT keyword extraction error: {e}", exc_info=True); return []

    def extract_concepts(self, text: str, top_n=10) -> List[Tuple[str, float]]:
        return self.extract_terms(text, top_n)

class SequentialMemory:
    def __init__(self, capacity: int = 10, feature_extractor=None, config=None, embedding_model=None):
        self.recent_queries = []
        self.capacity = capacity
        self.term_usage = defaultdict(int)
        self.feature_extractor = feature_extractor
        self.config = config if config else get_config()
        self.embedding_model = embedding_model

    def process_query(self, query: str, user_id: str, clicked_docs: List[Dict] = None) -> Dict[str, Any]:
        continuity_result = self._identify_research_continuity(query)
        self._update_memory(query, clicked_docs or [])
        terminology_result = self._detect_terminology_consistency(query)
        return {
            "query": query,
            "sequential_continuity": continuity_result,
            "sequential_terminology": terminology_result,
        }

    def _update_memory(self, query: str, clicked_docs: List[Dict]) -> None:
        self.recent_queries.append({"query": query, "timestamp": datetime.now().isoformat(), "clicked_docs": clicked_docs})
        if len(self.recent_queries) > self.capacity: self.recent_queries.pop(0)
        if self.feature_extractor:
            for term, _ in self.feature_extractor.extract_terms(query): self.term_usage[term] += 1

    def _identify_research_continuity(self, current_query: str) -> Dict[str, Any]:
        if not self.recent_queries: return {"detected": False, "confidence": 0.0, "previous_query": None, "high_continuity_terms": []}
        previous_query_text = self.recent_queries[-1]["query"]
        if not previous_query_text: return {"detected": False, "confidence": 0.0, "previous_query": None, "high_continuity_terms": []}
        similarity = self._calculate_query_similarity(current_query, previous_query_text)
        is_continuous = similarity > getattr(self.config, 'continuity_threshold', 0.25)
        high_continuity_terms = []
        if is_continuous and self.feature_extractor:
            prev_terms_data = self.feature_extractor.extract_terms(previous_query_text, top_n=5)
            high_continuity_terms = [term for term, score in prev_terms_data if score > 0.1]
        return {"detected": is_continuous, "confidence": similarity, "previous_query": previous_query_text if is_continuous else None, "high_continuity_terms": high_continuity_terms}

    def _detect_terminology_consistency(self, current_query: str) -> Dict[str, Any]:
        if not self.feature_extractor: return {"detected": False, "consistent_terms": [], "confidence": 0.0}
        frequent_historical_terms = [{"term": term, "frequency": freq} for term, freq in sorted(self.term_usage.items(), key=lambda item: item[1], reverse=True) if freq > 0]
        detected = bool(frequent_historical_terms)
        confidence = len([t for t in frequent_historical_terms if t["frequency"] > 1]) / len(frequent_historical_terms) if frequent_historical_terms else 0.0
        return {"detected": detected, "consistent_terms": frequent_historical_terms, "confidence": confidence}

    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        if not query1 or not query2: return 0.0
        if self.embedding_model:
            try:
                with torch.no_grad(): embeddings = self.embedding_model.encode([query1.strip() or "[empty]", query2.strip() or "[empty]"], normalize_embeddings=True)
                if embeddings is None or not isinstance(embeddings, np.ndarray) or embeddings.shape[0] != 2: return 0.0
                similarity = max(0.0, min(1.0, float(np.dot(embeddings[0], embeddings[1]))))
                del embeddings; gc.collect(); return similarity
            except Exception as e: logger.error(f"SM semantic similarity error: {e}", exc_info=True); return self._calculate_jaccard_similarity(query1, query2)
        return self._calculate_jaccard_similarity(query1, query2)

    def _calculate_jaccard_similarity(self, query1: str, query2: str) -> float:
        words1, words2 = set(re.findall(r'\b\w+\b', query1.lower())), set(re.findall(r'\b\w+\b', query2.lower()))
        return len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0.0

class WorkingMemory:
    def __init__(self, concept_limit: int = 20, feature_extractor=None, config=None):
        self.current_session_queries = []
        self.current_session_concepts = defaultdict(lambda: {"count": 0, "query_indices": []})
        self.concept_limit = concept_limit
        self.feature_extractor = feature_extractor
        self.config = config or get_config()

    def process_query(self, query: str, sequential_memory_results: Dict[str, Any], clicked_docs: List[Dict] = None) -> Dict[str, Any]:
        if not query or not self.feature_extractor: return {"session_focus": None, "current_query_core_concepts": []}
        self._update_session_state(query, clicked_docs or [])
        current_query_concepts_data = self.feature_extractor.extract_concepts(query, top_n=5)
        current_query_core_concepts = [concept for concept, score in current_query_concepts_data if score > 0.1]
        session_focus = self._determine_session_focus()
        return {"session_focus": session_focus, "current_query_core_concepts": current_query_core_concepts}

    def _update_session_state(self, query: str, clicked_docs: List[Dict]) -> None:
        self.current_session_queries.append(query)
        query_idx = len(self.current_session_queries) - 1
        if self.feature_extractor:
            for concept, _ in self.feature_extractor.extract_concepts(query, top_n=10):
                self.current_session_concepts[concept]["count"] += 1
                if query_idx not in self.current_session_concepts[concept]["query_indices"]: self.current_session_concepts[concept]["query_indices"].append(query_idx)

    def _determine_session_focus(self) -> Optional[str]:
        if not self.current_session_concepts: return None
        sorted_concepts = sorted(self.current_session_concepts.items(), key=lambda item: (item[1]["count"], max(item[1]["query_indices"] if item[1]["query_indices"] else [-1])), reverse=True)
        return sorted_concepts[0][0] if sorted_concepts else None

    def new_session(self):
        self.current_session_queries = []
        self.current_session_concepts.clear()
        logger.info("WorkingMemory: New session started.")

class LongTermMemory:
    def __init__(self, feature_extractor=None, vector_file=None, embedding_model_name=None, config=None):
        self.explicit_memory = {"research_topics": defaultdict(float), "methodologies": defaultdict(float)}
        self.implicit_memory = {"academic_background": {}, "technical_familiarity": defaultdict(float), "academic_level": "unknown", "level_confidence": 0.0}
        self.feature_extractor = feature_extractor
        self.config = config if config else get_config()
        self.vectors = {"topics": {}, "methods": {}}; self.indices = {} # FAISS related, not primary for keyword output

        global _shared_embedding_model, _embedding_model_device
        try:
            if _shared_embedding_model is None:
                _embedding_model_device = self.config.device if torch.cuda.is_available() and self.config.device and self.config.device != "cpu" else "cpu"
                model_path = embedding_model_name or getattr(self.config, 'sentence_transformer_model', 'all-MiniLM-L6-v2')
                use_trust = "gte" in model_path.lower() or "modelscope" in model_path.lower()
                _shared_embedding_model = SentenceTransformer(model_path, device=_embedding_model_device, trust_remote_code=use_trust)
            self.embedding_model = _shared_embedding_model
            # self.vector_dim = self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else 0
            # self._initialize_indices() # FAISS index not critical for current keyword extraction
        except Exception as e: logger.error(f"LTM SBERT model error: {e}", exc_info=True); self.embedding_model = None

    def update(self, query: str, working_memory_state: Dict[str, Any], clicked_docs: List[Dict] = None) -> None:
        if not self.feature_extractor: return
        try:
            for concept, score in self.feature_extractor.extract_concepts(query, top_n=5):
                if score > 0.2: self.explicit_memory["research_topics"][concept] += score
            if clicked_docs:
                for doc in clicked_docs or []:
                    content = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                    if not content: continue
                    for concept, score in self.feature_extractor.extract_concepts(content, top_n=3):
                        if score > 0.3: self.explicit_memory["research_topics"][concept] += score * 0.5
            # Simplified method update - can be expanded
            method_keywords = ["method", "approach", "algorithm", "technique", "analysis", "model"]
            extracted_terms = self.feature_extractor.extract_terms(query.lower(), top_n=5)
            for term, score in extracted_terms:
                if any(mk in term for mk in method_keywords):
                    self.explicit_memory["methodologies"][term] += score
        except Exception as e: logger.error(f"LTM update error: {e}", exc_info=True)

    def retrieve(self, query: str, working_memory_state: Dict[str, Any]) -> Dict[str, Any]:
        persistent_research_topics = [topic for topic, strength in sorted(self.explicit_memory["research_topics"].items(), key=lambda x: x[1], reverse=True)]
        persistent_methodologies = [method for method, strength in sorted(self.explicit_memory["methodologies"].items(), key=lambda x: x[1], reverse=True)]
        query_relevant_ltm_topics = []
        if self.feature_extractor:
            current_query_concepts = [c for c, _ in self.feature_extractor.extract_concepts(query, top_n=3)]
            for topic in persistent_research_topics: # Check against all persistent topics
                if any(qc.lower() in topic.lower() or topic.lower() in qc.lower() for qc in current_query_concepts):
                    query_relevant_ltm_topics.append(topic)

        return {"explicit_memory_keywords": {"persistent_research_topics": persistent_research_topics,
                                           "persistent_methodologies": persistent_methodologies,
                                           "query_relevant_ltm_topics": query_relevant_ltm_topics},
                "implicit_memory_snapshot": self._retrieve_implicit_memory_snapshot()} # Keep snapshot for other uses

    def _retrieve_implicit_memory_snapshot(self) -> Dict[str, Any]:
        max_items = getattr(self.config, 'max_phrases_per_tag', 3) # From utils.Config
        acad_bg = [{"discipline": d.replace("_", " "), "confidence": c} for d,c in sorted(self.implicit_memory["academic_background"].items(),key=lambda x:x[1],reverse=True)[:max_items]]
        top_techs = [{"technology": t, "familiarity": f} for t,f in sorted(self.implicit_memory["technical_familiarity"].items(),key=lambda x:x[1],reverse=True)[:max_items]]
        return {"academic_background_profile": acad_bg,
                "academic_level_profile": {"level": self.implicit_memory["academic_level"], "confidence": self.implicit_memory["level_confidence"]},
                "top_technologies_profile": top_techs}
    def new_session(self): pass

class CognitiveMemorySystem:
    def __init__(self, config=None):
        self.user_profiles = {}
        self.config = config if config else get_config()
        self.dataset_type = getattr(self.config, 'dataset_type', 'unknown')
        self.feature_extractor_type = getattr(self.config, 'feature_extractor', 'keybert')
        global _shared_embedding_model # Ensure we use the global instance
        self.embedding_model_instance = _shared_embedding_model # Set from global after LTM potentially initializes it
        try:
            keybert_model_path = getattr(self.config, 'keybert_model', None)
            self.feature_extractor = FeatureExtractorRegistry.get_extractor(self.feature_extractor_type, model_name=keybert_model_path, config=self.config)
        except Exception as e: logger.warning(f"CMS: Failed to init {self.feature_extractor_type} extractor: {e}. Using simple.", exc_info=True); self.feature_extractor = self._create_simple_extractor()

    def _create_simple_extractor(self):
        class SimpleExtractor:
            def extract_terms(self, text, top_n=10): words=re.findall(r'\b[a-zA-Z]{3,}\b',text.lower());c=defaultdict(int);[c[w].__iadd__(1) for w in words];s={'the','and','is','in','to','of','that','for','on','with','an','are'};l=len(words);return [(w,v/(l or 1)) for w,v in sorted(c.items(),key=lambda x:x[1],reverse=True) if w not in s][:top_n]
            def extract_concepts(self, text, top_n=10): return self.extract_terms(text, top_n)
        return SimpleExtractor()

    def _get_or_initialize_user_memory(self, user_id: str):
        if user_id not in self.user_profiles:
            logger.info(f"CMS: Initializing new memory profile for user_id: {user_id}")
            self.user_profiles[user_id] = {
                "sequential_memory": SequentialMemory(feature_extractor=self.feature_extractor, config=self.config, embedding_model=self.embedding_model_instance),
                "working_memory": WorkingMemory(feature_extractor=self.feature_extractor, config=self.config),
                "long_term_memory": LongTermMemory(feature_extractor=self.feature_extractor, config=self.config), # LTM now also gets embedding_model via global
                "current_topic_id": None
            }
        return self.user_profiles[user_id]

    def process_query(self, query_text: str, user_id: str, clicked_docs: List[Dict] = None, topic_id: Optional[str] = None) -> Dict[str, Any]:
        user_memory = self._get_or_initialize_user_memory(user_id)
        if topic_id and user_memory.get("current_topic_id") != topic_id:
            user_memory["working_memory"].new_session(); user_memory["current_topic_id"] = topic_id
        seq_mem, work_mem, lt_mem = user_memory["sequential_memory"], user_memory["working_memory"], user_memory["long_term_memory"]
        seq_res_raw = seq_mem.process_query(query_text, user_id, clicked_docs)
        work_state_raw = work_mem.process_query(query_text, seq_res_raw, clicked_docs)
        lt_mem.update(query_text, work_state_raw, clicked_docs)
        ltm_res_raw = lt_mem.retrieve(query_text, work_state_raw)
        return {"sequential_results_raw": seq_res_raw, "working_memory_state_raw": work_state_raw, "long_term_memory_results_raw": ltm_res_raw}

    def get_tagged_features(self, memory_results_raw: Dict[str, Any],
                            active_components: List[str] = None,
                            continuity_score_override: float = None) -> List[str]:
        active_components = [c.lower() for c in (active_components or getattr(self.config, 'memory_components', []))]
        config = self.config
        max_overall_features = getattr(config, 'max_tagged_features_for_llm', 7)
        max_from_module = getattr(config, 'max_features_per_memory_module', 2)
        max_kws_in_str = getattr(config, 'max_phrases_per_tag', 3)

        collected_features = []

        # Sequential Memory
        if 'sequential' in active_components:
            seq_module_features = []
            seq_raw = memory_results_raw.get("sequential_results_raw", {})
            cont_info = seq_raw.get("sequential_continuity", {})
            term_info = seq_raw.get("sequential_terminology", {})
            cont_score = continuity_score_override if continuity_score_override is not None else cont_info.get("confidence", 0.0)

            if cont_score > getattr(config, 'continuity_threshold', 0.25) and len(seq_module_features) < max_from_module:
                seq_module_features.append(f"{SEQUENTIAL_MEMORY_TAG} Continuity with previous query: {cont_score:.2f}")
            prev_themes = cont_info.get("high_continuity_terms", [])
            if prev_themes and len(seq_module_features) < max_from_module:
                seq_module_features.append(f"{SEQUENTIAL_MEMORY_TAG} Continuing previous themes: {', '.join(prev_themes[:max_kws_in_str])}")
            # consistent_terms is list of {"term": t, "frequency": f}
            recent_freq = [item['term'] for item in term_info.get("consistent_terms", [])[:max_kws_in_str]] # Take top N by pre-sorted freq
            if recent_freq and len(seq_module_features) < max_from_module:
                seq_module_features.append(f"{SEQUENTIAL_MEMORY_TAG} Recent frequent themes: {', '.join(recent_freq)}")
            collected_features.extend(seq_module_features)

        # Working Memory
        if 'working' in active_components and len(collected_features) < max_overall_features:
            wm_module_features = []
            wm_raw = memory_results_raw.get("working_memory_state_raw", {})
            focus = wm_raw.get("session_focus")
            if focus and len(wm_module_features) < max_from_module:
                wm_module_features.append(f"{WORKING_MEMORY_TAG} Current session focus: {focus}")
            core_concepts = wm_raw.get("current_query_core_concepts", [])
            if core_concepts and len(wm_module_features) < max_from_module:
                wm_module_features.append(f"{WORKING_MEMORY_TAG} Current query core concepts: {', '.join(core_concepts[:max_kws_in_str])}")
            collected_features.extend(wm_module_features)

        # Long-Term Memory
        if 'long' in active_components and len(collected_features) < max_overall_features:
            ltm_module_features = []
            ltm_raw = memory_results_raw.get("long_term_memory_results_raw", {})
            explicit_kws = ltm_raw.get("explicit_memory_keywords", {})
            persistent_topics = explicit_kws.get("persistent_research_topics", [])
            if persistent_topics and len(ltm_module_features) < max_from_module:
                ltm_module_features.append(f"{LONG_EXPLICIT_TAG} Long-term research directions: {', '.join(persistent_topics[:max_kws_in_str])}")
            
            # Optionally add more LTM features if space and configured
            persistent_methods = explicit_kws.get("persistent_methodologies", [])
            if persistent_methods and len(ltm_module_features) < max_from_module:
                 ltm_module_features.append(f"{LONG_EXPLICIT_TAG} Long-term methodologies: {', '.join(persistent_methods[:max_kws_in_str])}")
            
            # query_relevant_ltm = explicit_kws.get("query_relevant_ltm_topics", [])
            # if query_relevant_ltm and len(ltm_module_features) < max_from_module:
            #      ltm_module_features.append(f"{LONG_EXPLICIT_TAG} Query-relevant LTM topics: {', '.join(query_relevant_ltm[:max_kws_in_str])}")
            
            collected_features.extend(ltm_module_features)

        final_features = collected_features[:max_overall_features]
        logger.info(f"CMS: Generated {len(final_features)} tagged keyword strings for LLM (selected {max_from_module} per module, total {max_overall_features}): {final_features}")
        return final_features

    def new_user_session(self, user_id: str, topic_id: str): # Renamed from _initialize_user for clarity
        self._get_or_initialize_user_memory(user_id) # Ensures profile exists
        user_memory = self.user_profiles[user_id]
        if user_memory.get("current_topic_id") != topic_id:
            logger.info(f"CMS: New session for user '{user_id}', topic_id '{topic_id}'. Resetting Working Memory.")
            if "working_memory" in user_memory and hasattr(user_memory["working_memory"], "new_session"):
                user_memory["working_memory"].new_session()
            user_memory["current_topic_id"] = topic_id
