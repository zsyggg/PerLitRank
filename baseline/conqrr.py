# conqrr.py
"""
CONQRR (Conversational Query Rewriting with Retrieval-Augmented Refinement) (修正参数、排序和导入)
会话式查询重写框架，结合检索增强进行查询精炼
"""
import torch
import json
import logging
import os
import sys
import numpy as np
import faiss
from tqdm import tqdm
from collections import defaultdict # Import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import argparse # Import argparse

# --- Add parent directory to sys.path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# --- End Path Addition ---

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CONQRR')

# --- Import Dependencies after path setup ---
try:
    from personalized_generator import PersonalizedGenerator
except ImportError:
    logger.error("Failed to import PersonalizedGenerator. Make sure 'personalized_generator.py' exists in the parent directory or PYTHONPATH is set correctly.", exc_info=True)
    sys.exit(1)

# --- PyTorch and FAISS Check ---
try:
    import torch
    import faiss
except ImportError as e:
    logger.error(f"Missing required library: {e}. Please install torch and faiss-cpu or faiss-gpu.")
    sys.exit(1)


class CONQRRBaseline:
    """CONQRR基线方法实现"""

    def __init__(self, model_path=None, device=None, batch_size=8, encoder_model=None):
        self.logger = logger
        self.model_path = model_path # LLM Path
        if not self.model_path: raise ValueError("LLM model_path is required for CONQRR.")
        self.encoder_model_path = encoder_model
        if not self.encoder_model_path: raise ValueError("Encoder model path is required for CONQRR retrieval.")

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size # Used for encoder
        self.logger.info(f"CONQRR Initializing with LLM: {self.model_path}, Encoder: {self.encoder_model_path}, Device: {self.device}, Batch Size (Encoder): {self.batch_size}")

        self.retrieval_k = 10 # For initial retrieval
        self.final_retrieval_k = 10 # For retrieval after rewrite

        self.generator = None
        self.encoder_tokenizer = None
        self.encoder_model = None
        self.corpus = None
        self.doc_ids = None
        self.corpus_embeddings = None
        self.index = None

        self._setup_llm()
        # Retrieval setup called in process_dataset if needed

    def _setup_llm(self):
        """设置LLM模型 - 使用PersonalizedGenerator"""
        try:
            self.logger.info(f"Initializing PersonalizedGenerator with LLM: {self.model_path} on device {self.device}")
            # Pass necessary config details to PersonalizedGenerator if needed
            self.generator = PersonalizedGenerator(local_model_path=self.model_path, devices=self.device)
            if self.generator.model is None or self.generator.tokenizer is None:
                 raise RuntimeError("PersonalizedGenerator failed to load model/tokenizer.")
            self.logger.info(f"PersonalizedGenerator initialized successfully.")
        except Exception as e:
            self.logger.error(f"初始化PersonalizedGenerator失败: {e}", exc_info=True)
            raise

    def rewrite_query(self, query, history_queries=None, retrieved_docs=None, max_tokens=50):
        """使用LLM重写查询，结合历史和检索到的文档"""
        if not self.generator: self.logger.error("Generator not initialized."); return query
        try:
            prompt_parts = ["Rewrite the 'Current Query' to be a self-contained query suitable for academic literature search.",
                            "Consider the 'Query History' and potentially relevant 'Retrieved Document Snippets'."]
            if history_queries:
                prompt_parts.append("\nQuery History (most recent first):")
                for i, hq in enumerate(reversed(history_queries[-3:])): prompt_parts.append(f"- Turn {-i-1}: {hq}")
            prompt_parts.append(f"\nCurrent Query: {query}")
            if retrieved_docs:
                prompt_parts.append("\nRetrieved Document Snippets:")
                for i, doc in enumerate(retrieved_docs[:3]):
                    title = doc.get("title", ""); text = doc.get("text", ""); snippet = f"{title} - {text}"[:150]
                    prompt_parts.append(f"- Doc {i+1}: {snippet}...")
            prompt_parts.append("\nRewritten Query:")
            rewrite_prompt = "\n".join(prompt_parts)

            if hasattr(self.generator, 'model') and hasattr(self.generator, 'tokenizer'):
                inputs = self.generator.tokenizer(rewrite_prompt, return_tensors="pt").to(self.generator.device)
                with torch.no_grad():
                    outputs = self.generator.model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.5, pad_token_id=self.generator.tokenizer.eos_token_id)
                rewritten_query = self.generator.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
                rewritten_query = rewritten_query.split("Rewritten Query:")[-1].strip()
            else:
                 logger.warning("Using generate_personalized_text for query rewriting (suboptimal).")
                 rewritten_query_raw = self.generator.generate_personalized_text(rewrite_prompt, {}, None)
                 rewritten_query = rewritten_query_raw.split(':')[-1].strip()

            if not rewritten_query or len(rewritten_query) < 5:
                 self.logger.warning(f"Rewritten query empty/short. Falling back to original: '{query}'.")
                 rewritten_query = query
            self.logger.debug(f"Original: '{query}', Rewritten: '{rewritten_query}'")
            return rewritten_query
        except Exception as e: self.logger.error(f"重写查询失败 for query '{query}': {e}", exc_info=True); return query

    def _setup_retrieval(self, corpus_path, corpus_embeddings_path):
        """设置检索所需的组件"""
        # (Same implementation as qer_py_v2)
        try:
            self.logger.info(f"Loading corpus (JSONL) from {corpus_path}")
            self.corpus = {}; self.doc_ids = []
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line); doc_id = str(data.get('text_id', ''))
                        if doc_id: self.corpus[doc_id] = data; self.doc_ids.append(doc_id)
                    except json.JSONDecodeError: self.logger.warning(f"Skipping invalid JSON line in corpus: {line[:100]}...")

            self.logger.info(f"Loading corpus embeddings from {corpus_embeddings_path}")
            self.corpus_embeddings = np.load(corpus_embeddings_path).astype(np.float32)
            if self.corpus_embeddings.shape[0] != len(self.doc_ids):
                 raise ValueError(f"Embeddings shape mismatch: Expected ({len(self.doc_ids)}, dim), Got {self.corpus_embeddings.shape}")

            self.logger.info(f"Loading encoder model from {self.encoder_model_path}")
            use_trust = "gte" in self.encoder_model_path.lower() or "modelscope" in self.encoder_model_path
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_model_path, trust_remote_code=use_trust)
            self.encoder_model = AutoModel.from_pretrained(self.encoder_model_path, trust_remote_code=use_trust).to(self.device)
            self.encoder_model.eval()

            dimension = self.corpus_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.corpus_embeddings)
            self.logger.info(f"Retrieval setup completed with {self.index.ntotal} vectors.")
            self.retrieval_ready = True
        except Exception as e:
            self.logger.error(f"Error setting up retrieval: {e}", exc_info=True)
            self.retrieval_ready = False

    def retrieve(self, query, top_k=10):
        """检索相关文档"""
        # (Same implementation as qer_py_v2)
        if not self.retrieval_ready: self.logger.error("Retrieval components not ready."); return []
        try:
            inputs = self.encoder_tokenizer([query], padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.encoder_model(**inputs)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None: query_embedding = outputs.pooler_output.cpu().numpy()
                elif hasattr(outputs, 'last_hidden_state'): query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else: raise ValueError("Encoder output missing pooler_output/last_hidden_state")
            norm = np.linalg.norm(query_embedding);
            if norm > 1e-12: query_embedding = query_embedding / norm
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            results = []
            if scores.size > 0 and indices.size > 0:
                 for score, idx in zip(scores[0], indices[0]):
                     if idx != -1 and idx < len(self.doc_ids):
                         doc_id = self.doc_ids[idx]
                         if doc_id in self.corpus:
                            doc_data = self.corpus[doc_id]
                            results.append({"text_id": doc_id, "title": doc_data.get("title", ""), "text": doc_data.get("text", ""), "score": float(score)})
                         else: self.logger.warning(f"Doc ID {doc_id} from search not in corpus.")
            return results
        except Exception as e: self.logger.error(f"Error during retrieval for query '{query}': {e}", exc_info=True); return []


    def process_dataset(self, input_file, output_file, retrieved_file=None, corpus_path=None, corpus_embeddings_path=None, do_retrieval=True):
        """批量处理数据集中的查询 (JSONL)"""
        if not os.path.exists(input_file): self.logger.error(f"Input file not found: {input_file}"); return
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.logger.info(f"Processing queries from {input_file}")

        retrieved_docs_map = {}
        if retrieved_file and os.path.exists(retrieved_file):
            self.logger.info(f"Loading pre-retrieved results from {retrieved_file}")
            try:
                with open(retrieved_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip()); query_id = str(data.get("query_id", ""))
                        if query_id: retrieved_docs_map[query_id] = data.get("results", [])
            except Exception as e: self.logger.error(f"Error loading retrieved file: {e}")

        self.retrieval_ready = False
        if do_retrieval:
            if not corpus_path or not corpus_embeddings_path: self.logger.error("Corpus/Embeddings path required for retrieval."); return
            if not os.path.exists(corpus_path) or not os.path.exists(corpus_embeddings_path): self.logger.error("Corpus or embeddings file not found."); return
            self._setup_retrieval(corpus_path, corpus_embeddings_path)

        try:
            queries_data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try: queries_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError: self.logger.warning(f"Skipping invalid JSON line in input: {line[:100]}...")

            # Removed continuity filtering

            results = []
            grouped_queries = defaultdict(list)
            for item in queries_data: # Group queries by conversation/topic ID
                 query_id = str(item.get("query_id", ""))
                 topic_id = query_id.split('_')[0] if '_' in query_id else query_id
                 grouped_queries[topic_id].append(item)

            for topic_id, items in tqdm(grouped_queries.items(), desc="Processing conversations/queries"):
                 # Define a safe sorting key function
                 def sort_key(x):
                     qid = x.get("query_id", "0")
                     # Check if qid is a string before splitting
                     if isinstance(qid, str) and '_' in qid:
                         parts = qid.split('_')
                         if len(parts) > 1 and parts[-1].isdigit():
                             return int(parts[-1])
                     # Handle cases where qid is not string or has no '_' or last part is not digit
                     # Maybe assign a default order or use original position? For now, 0.
                     return 0

                 items.sort(key=sort_key) # Use the safe sort key

                 history_for_topic = []
                 for i, item in enumerate(items):
                     query_id = str(item.get("query_id", ""))
                     current_query = item.get("query", "")
                     if not query_id or not current_query: continue

                     initial_retrieved = retrieved_docs_map.get(query_id, [])
                     if not initial_retrieved and do_retrieval and self.retrieval_ready:
                         initial_retrieved = self.retrieve(current_query, self.retrieval_k)

                     rewritten_query = self.rewrite_query(current_query, history_for_topic, initial_retrieved)

                     final_retrieved_docs = []
                     if do_retrieval and self.retrieval_ready:
                         final_retrieved_docs = self.retrieve(rewritten_query, self.final_retrieval_k)
                     elif not do_retrieval: final_retrieved_docs = []
                     else: final_retrieved_docs = []

                     results.append({
                         "query_id": query_id, "original_query": current_query,
                         "rewritten_query": rewritten_query, "history_queries": list(history_for_topic),
                         "results": final_retrieved_docs
                     })
                     history_for_topic.append(current_query); history_for_topic = history_for_topic[-3:]

            with open(output_file, 'w', encoding='utf-8') as f:
                for r in results: f.write(json.dumps(r, ensure_ascii=False) + '\n')
            self.logger.info(f"Processed {len(results)} queries. Results saved to {output_file}")

        except Exception as e: self.logger.error(f"Error processing dataset: {e}", exc_info=True); raise

def main():
    parser = argparse.ArgumentParser(description='CONQRR Baseline (No Continuity/Parquet)')
    parser.add_argument('--input_file', type=str, required=True, help='Input queries file path (JSONL)')
    parser.add_argument('--output_file', type=str, required=True, help='Output results file path (JSONL)')
    parser.add_argument('--retrieved_file', type=str, help='Optional: Pre-retrieved results file path (JSONL)')
    parser.add_argument('--corpus_path', type=str, help='Corpus file path (JSONL)') # Required if not --no_retrieval
    parser.add_argument('--corpus_embeddings_path', type=str, help='Corpus embeddings file path (.npy)') # Required if not --no_retrieval
    parser.add_argument('--model_path', type=str, required=True, help='Path to the LLM model')
    parser.add_argument('--encoder_model', type=str, required=True, help='Path to the encoder model for retrieval')
    parser.add_argument('--no_retrieval', action='store_true', help='Disable retrieval step')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for encoder') # Added batch_size
    parser.add_argument('--device', type=str, default=None, help='Device for inference (e.g., cuda:0, cpu)') # Added device
    # Removed continuity and corpus_format args

    args = parser.parse_args()

    if not args.no_retrieval and (not args.corpus_path or not args.corpus_embeddings_path):
         parser.error("--corpus_path and --corpus_embeddings_path are required unless --no_retrieval is set.")

    if args.device is None:
         args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
         logger.info(f"Device not specified, using: {args.device}")
    elif not (args.device == "cpu" or (args.device.startswith("cuda:") and args.device.split(':')[1].isdigit())):
         logger.error(f"Invalid device format: '{args.device}'. Use 'cpu' or 'cuda:N'. Exiting.")
         sys.exit(1)

    try:
        conqrr = CONQRRBaseline(
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            encoder_model=args.encoder_model
        )
        conqrr.process_dataset(
            args.input_file,
            args.output_file,
            retrieved_file=args.retrieved_file,
            corpus_path=args.corpus_path,
            corpus_embeddings_path=args.corpus_embeddings_path,
            do_retrieval=not args.no_retrieval
        )
    except Exception as e:
         logger.error(f"CONQRR execution failed: {e}", exc_info=True)
         sys.exit(1)

if __name__ == '__main__':
    main()
