# htps.py
"""
HTPS (Hierarchical Transformer for Personalized Search) (修正参数)
层次化Transformer个性化搜索框架
"""
import json
import logging
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import argparse # Import argparse

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HTPS')

# --- PyTorch and FAISS Check ---
try:
    import torch
    import faiss
except ImportError as e:
    logger.error(f"Missing required library: {e}. Please install torch and faiss-cpu or faiss-gpu.")
    sys.exit(1)


class HTPSBaseline:
    """HTPS基线方法实现"""

    def __init__(self, model_path=None, device=None, batch_size=16):
        self.logger = logger
        self.model_path = model_path
        if not self.model_path:
             raise ValueError("Encoder model_path is required for HTPS.")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.logger.info(f"HTPS Initializing with device: {self.device}, batch_size: {self.batch_size}")

        self.user_history = {} # Store history embeddings per user/topic_id

        self.tokenizer = None
        self.model = None
        self._setup_model()

    def _setup_model(self):
        """Loads the encoder model."""
        try:
            self.logger.info(f"加载编码器模型: {self.model_path}")
            use_trust_remote_code = "gte" in self.model_path.lower() or "modelscope" in self.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=use_trust_remote_code
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=use_trust_remote_code
            ).to(self.device)
            self.model.eval()
            self.logger.info(f"编码器模型加载成功到设备: {self.device}")
        except Exception as e:
            self.logger.error(f"加载编码器模型失败: {e}", exc_info=True)
            raise

    def _encode_text(self, texts):
        """编码文本列表为向量"""
        if not texts: return np.array([])
        embeddings = []
        self.model.eval()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
                    outputs = self.model(**inputs)
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                         batch_embeddings = outputs.pooler_output.cpu().numpy()
                    elif hasattr(outputs, 'last_hidden_state'):
                         batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    else:
                         raise ValueError("Model output does not contain pooler_output or last_hidden_state")
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1e-12
                    batch_embeddings = batch_embeddings / norms
                    embeddings.append(batch_embeddings)
            except Exception as e:
                self.logger.error(f"Error encoding batch (start index {i}): {e}", exc_info=True)
                batch_embeddings = np.zeros((len(batch), self.model.config.hidden_size))
                embeddings.append(batch_embeddings)
        return np.vstack(embeddings) if embeddings else np.array([])


    def _build_hierarchical_representation(self, query, user_id=None):
        """构建层次化表示 (Simplified version using stored history)"""
        query_embedding = self._encode_text([query])
        if query_embedding.size == 0:
            self.logger.warning(f"Failed to encode query for {user_id}, returning zero vector.")
            return np.zeros(self.model.config.hidden_size)
        query_embedding = query_embedding[0]

        # Retrieve history for the user/topic
        current_history_embeddings = self.user_history.get(user_id, [])

        history_vector = np.zeros_like(query_embedding)
        if current_history_embeddings:
             history_vector = np.mean(current_history_embeddings, axis=0)

        # Add current query embedding to history for next time
        if user_id not in self.user_history: self.user_history[user_id] = []
        self.user_history[user_id].append(query_embedding)
        self.user_history[user_id] = self.user_history[user_id][-10:] # Keep last 10

        # Combine current query and history
        personalized_vector = 0.7 * query_embedding + 0.3 * history_vector
        norm = np.linalg.norm(personalized_vector)
        if norm > 1e-12: personalized_vector = personalized_vector / norm
        else: return query_embedding # Fallback

        return personalized_vector

    def load_corpus_and_embeddings(self, corpus_path, corpus_embeddings_path):
        """加载语料库 (JSONL) 和嵌入向量"""
        self.logger.info(f"Loading corpus (JSONL) from {corpus_path}")
        corpus = {}; doc_ids = []
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line); doc_id = str(data.get('text_id', ''))
                        if doc_id: corpus[doc_id] = data; doc_ids.append(doc_id)
                    except json.JSONDecodeError: self.logger.warning(f"Skipping invalid JSON line in corpus: {line[:100]}...")
        except Exception as e: self.logger.error(f"Error loading corpus: {e}", exc_info=True); raise

        self.logger.info(f"Loading corpus embeddings from {corpus_embeddings_path}")
        try:
            corpus_embeddings = np.load(corpus_embeddings_path).astype(np.float32)
            if corpus_embeddings.ndim != 2 or corpus_embeddings.shape[0] != len(doc_ids):
                 raise ValueError(f"Embeddings shape mismatch: Expected ({len(doc_ids)}, dim), Got {corpus_embeddings.shape}")
        except Exception as e: self.logger.error(f"Error loading/validating embeddings: {e}", exc_info=True); raise
        self.logger.info(f"Loaded {len(corpus)} documents and embeddings shape {corpus_embeddings.shape}")
        return corpus, doc_ids, corpus_embeddings

    def search(self, query, user_id=None, corpus_embeddings=None, doc_ids=None, index=None, top_k=10):
        """使用 FAISS 索引搜索相关文档"""
        if index is None: self.logger.error("FAISS index not provided"); return []
        try:
            # Pass user_id to get history-aware vector
            query_vector = self._build_hierarchical_representation(query, user_id=user_id)
            query_vector_np = query_vector.reshape(1, -1).astype(np.float32)
            scores, indices = index.search(query_vector_np, top_k)
            results = []
            if scores.size > 0 and indices.size > 0:
                 for score, idx in zip(scores[0], indices[0]):
                     if idx != -1 and idx < len(doc_ids): results.append({"text_id": doc_ids[idx], "score": float(score)})
            return results
        except Exception as e: self.logger.error(f"Error during FAISS search for user {user_id}: {e}", exc_info=True); return []


    def process_dataset(self, input_file, output_file, corpus_path, corpus_embeddings_path):
        """处理数据集 (JSONL 格式)"""
        if not all(os.path.exists(p) for p in [input_file, corpus_path, corpus_embeddings_path]):
            self.logger.error("One or more input files not found. Exiting.")
            return
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.logger.info(f"Processing queries from {input_file}")

        try:
            corpus, doc_ids, corpus_embeddings = self.load_corpus_and_embeddings(corpus_path, corpus_embeddings_path)
            dimension = corpus_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension); index.add(corpus_embeddings)
            self.logger.info(f"FAISS index created with {index.ntotal} vectors.")

            queries_data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try: queries_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError: self.logger.warning(f"Skipping invalid JSON line in input: {line[:100]}...")

            final_results = []; processed_count = 0
            grouped_queries = defaultdict(list)
            for item in queries_data: # Group queries by conversation/topic ID
                 query_id = str(item.get("query_id", ""))
                 topic_id = query_id.split('_')[0] if '_' in query_id else query_id
                 grouped_queries[topic_id].append(item)

            for topic_id, items in tqdm(grouped_queries.items(), desc="Processing conversations/queries"):
                 # Define a safe sorting key function
                 def sort_key(x):
                     qid = x.get("query_id", "0")
                     if isinstance(qid, str) and '_' in qid:
                         try:
                             return int(qid.split('_')[-1])
                         except ValueError:
                             return 0 # Fallback for non-integer turn IDs
                     return 0 # Default for non-conversational IDs

                 items.sort(key=sort_key) # Use the safe sort key

                 # Reset history for each new topic/user
                 self.user_history[topic_id] = []

                 for i, item in enumerate(items):
                     query_id = str(item.get("query_id", ""))
                     current_query = item.get("query", "")
                     if not query_id or not current_query: continue

                     # Search uses the history stored in self.user_history[topic_id]
                     search_results = self.search(
                         current_query,
                         user_id=topic_id, # Pass topic_id for history lookup
                         corpus_embeddings=corpus_embeddings,
                         doc_ids=doc_ids,
                         index=index,
                         top_k=10
                     )
                     # Note: _build_hierarchical_representation updates history internally now

                     full_results = []
                     for res in search_results:
                         doc_id_res = res["text_id"]
                         if doc_id_res in corpus:
                             doc_data = corpus[doc_id_res]
                             full_results.append({"text_id": doc_id_res, "title": doc_data.get("title", ""), "text": doc_data.get("text", ""), "score": res["score"]})
                         else: self.logger.warning(f"Doc ID {doc_id_res} not found in corpus.")
                     final_results.append({"query_id": query_id, "query": current_query, "results": full_results})
                     processed_count += 1

            with open(output_file, 'w', encoding='utf-8') as f_out:
                for result in final_results: f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            self.logger.info(f"Processed {processed_count} queries. Results saved to {output_file}")

        except Exception as e: self.logger.error(f"Error processing dataset: {e}", exc_info=True); raise


def main():
    parser = argparse.ArgumentParser(description='HTPS Baseline (No Continuity/Parquet)')
    parser.add_argument('--input_file', type=str, required=True, help='Input queries file path (JSONL format)')
    parser.add_argument('--output_file', type=str, required=True, help='Output results file path (JSONL)')
    parser.add_argument('--corpus_path', type=str, required=True, help='Corpus file path (JSONL format)')
    parser.add_argument('--corpus_embeddings_path', type=str, required=True, help='Corpus embeddings file path (.npy)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the encoder model (e.g., GTE)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for encoding')
    parser.add_argument('--device', type=str, default=None, help='Device (e.g., cuda:0, cpu)') # Added device arg
    # Removed continuity and corpus_format args

    args = parser.parse_args()

    if args.device is None:
         args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
         logger.info(f"Device not specified, using: {args.device}")
    elif not (args.device == "cpu" or (args.device.startswith("cuda:") and args.device.split(':')[1].isdigit())):
         logger.error(f"Invalid device format: '{args.device}'. Use 'cpu' or 'cuda:N'. Exiting.")
         sys.exit(1)

    try:
        htps = HTPSBaseline(model_path=args.model_path, device=args.device, batch_size=args.batch_size)
        htps.process_dataset(args.input_file, args.output_file, args.corpus_path, args.corpus_embeddings_path)
    except Exception as e:
         logger.error(f"HTPS execution failed: {e}", exc_info=True)
         sys.exit(1)

if __name__ == '__main__':
    main()
