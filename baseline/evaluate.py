# rpmn.py
"""
RPMN (Re-finding Personalized Memory Network) (移除连续性和Parquet逻辑)
基于记忆网络的个性化搜索模型
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
# Removed continuity_utils import

class RPMNBaseline:
    """
    RPMN基线方法实现
    使用三种外部记忆进行个性化搜索
    """

    def __init__(self, model_path=None, device=None, batch_size=16):
        """初始化RPMN模型"""
        self.logger = logging.getLogger('RPMN')
        self.model_path = model_path # Required, passed from run_baselines.py
        if not self.model_path:
             raise ValueError("Encoder model_path is required for RPMN.")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # 记忆网络
        self.short_term_memory = {}
        self.medium_term_memory = {}
        self.long_term_memory = {}

        # 编码器模型
        self.tokenizer = None
        self.model = None
        self._setup_model()

    def _setup_model(self):
        """Loads the encoder model."""
        try:
            self.logger.info(f"加载编码器模型: {self.model_path}")
            # Assuming GTE or similar model requiring trust_remote_code
            use_trust_remote_code = "gte" in self.model_path.lower() or "modelscope" in self.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=use_trust_remote_code
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=use_trust_remote_code
            ).to(self.device)
            self.model.eval() # Set to eval mode
            self.logger.info(f"编码器模型加载成功到设备: {self.device}")
        except Exception as e:
            self.logger.error(f"加载编码器模型失败: {e}", exc_info=True)
            raise

    def _encode_text(self, texts):
        """编码文本列表为向量"""
        if not texts:
            return np.array([]) # Return empty numpy array

        embeddings = []
        self.model.eval() # Ensure model is in eval mode
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
                    outputs = self.model(**inputs)
                    # Pooler output or CLS token hidden state
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                         batch_embeddings = outputs.pooler_output.cpu().numpy()
                    elif hasattr(outputs, 'last_hidden_state'):
                         batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() # CLS token
                    else:
                         raise ValueError("Model output does not contain pooler_output or last_hidden_state")

                    # Normalize embeddings
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    # Avoid division by zero for zero vectors
                    norms[norms == 0] = 1e-12
                    batch_embeddings = batch_embeddings / norms
                    embeddings.append(batch_embeddings)

            except Exception as e:
                self.logger.error(f"Error encoding batch (start index {i}): {e}", exc_info=True)
                # Return zero vectors as fallback for the batch
                batch_embeddings = np.zeros((len(batch), self.model.config.hidden_size))
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings) if embeddings else np.array([])


    def _update_memories(self, query, history_queries=None, user_id=None):
        """更新记忆网络"""
        if user_id is None: user_id = "default_user"

        try:
            query_embedding = self._encode_text([query])
            if query_embedding.size == 0: return # Skip if encoding failed

            query_embedding = query_embedding[0] # Get the single embedding

            # Update short-term memory (last 5 queries)
            if user_id not in self.short_term_memory: self.short_term_memory[user_id] = []
            self.short_term_memory[user_id].append({"embedding": query_embedding})
            self.short_term_memory[user_id] = self.short_term_memory[user_id][-5:]

            # Update medium-term memory (current session - simplified)
            if user_id not in self.medium_term_memory: self.medium_term_memory[user_id] = []
            if history_queries:
                 history_embeddings = self._encode_text(history_queries)
                 if history_embeddings.size > 0:
                     for emb in history_embeddings:
                         self.medium_term_memory[user_id].append({"embedding": emb})
            self.medium_term_memory[user_id].append({"embedding": query_embedding})
            # Simple session memory: keep last 10 interactions for medium term average
            self.medium_term_memory[user_id] = self.medium_term_memory[user_id][-10:]


            # Update long-term memory (user interest profile - average embedding)
            if user_id not in self.long_term_memory:
                self.long_term_memory[user_id] = {"count": 0, "avg_embedding": np.zeros_like(query_embedding)}
            memory = self.long_term_memory[user_id]
            memory["count"] += 1
            # Incremental update of average embedding
            memory["avg_embedding"] = memory["avg_embedding"] + (query_embedding - memory["avg_embedding"]) / memory["count"]

        except Exception as e:
            self.logger.error(f"Error updating memories for user {user_id}: {e}", exc_info=True)

    def _get_personalized_vector(self, query, history_queries=None, user_id=None):
        """获取个性化查询向量"""
        if user_id is None: user_id = "default_user"

        try:
            self._update_memories(query, history_queries, user_id)

            query_embedding = self._encode_text([query])
            if query_embedding.size == 0:
                 self.logger.warning(f"Failed to encode query for {user_id}, returning zero vector.")
                 return np.zeros(self.model.config.hidden_size) # Return zero vector on failure
            query_embedding = query_embedding[0]


            # Short-term contribution (most recent query embedding)
            short_term_vec = np.zeros_like(query_embedding)
            if user_id in self.short_term_memory and self.short_term_memory[user_id]:
                short_term_vec = self.short_term_memory[user_id][-1]["embedding"]

            # Medium-term contribution (average of recent session embeddings)
            medium_term_vec = np.zeros_like(query_embedding)
            if user_id in self.medium_term_memory and self.medium_term_memory[user_id]:
                 session_embeddings = [m["embedding"] for m in self.medium_term_memory[user_id]]
                 if session_embeddings:
                     medium_term_vec = np.mean(session_embeddings, axis=0)


            # Long-term contribution (average embedding)
            long_term_vec = np.zeros_like(query_embedding)
            if user_id in self.long_term_memory:
                 long_term_vec = self.long_term_memory[user_id]["avg_embedding"]

            # Combine memories (adjust weights as needed)
            # Weights: Current Query (0.6), Short-Term (0.2), Medium-Term (0.15), Long-Term (0.05)
            personalized_vector = (0.6 * query_embedding +
                                   0.2 * short_term_vec +
                                   0.15 * medium_term_vec +
                                   0.05 * long_term_vec)

            # Normalize the final vector
            norm = np.linalg.norm(personalized_vector)
            if norm > 1e-12: # Avoid division by zero
                personalized_vector = personalized_vector / norm
            else:
                 self.logger.warning(f"Personalized vector norm is zero for user {user_id}. Returning original query embedding.")
                 return query_embedding # Fallback to original if personalized is zero

            return personalized_vector

        except Exception as e:
            self.logger.error(f"Error creating personalized vector for user {user_id}: {e}", exc_info=True)
            # Fallback to original query embedding on error
            query_embedding = self._encode_text([query])
            return query_embedding[0] if query_embedding.size > 0 else np.zeros(self.model.config.hidden_size)


    def load_corpus_and_embeddings(self, corpus_path, corpus_embeddings_path):
        """加载语料库 (JSONL) 和嵌入向量"""
        self.logger.info(f"Loading corpus (JSONL) from {corpus_path}")
        corpus = {}
        doc_ids = []
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        doc_id = str(data.get('text_id', '')) # Ensure string ID
                        if doc_id:
                            corpus[doc_id] = data
                            doc_ids.append(doc_id)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Skipping invalid JSON line in corpus: {line[:100]}...")
                        continue
        except Exception as e:
            self.logger.error(f"Error loading corpus: {e}", exc_info=True)
            raise

        self.logger.info(f"Loading corpus embeddings from {corpus_embeddings_path}")
        try:
            corpus_embeddings = np.load(corpus_embeddings_path)
            # Basic validation
            if corpus_embeddings.ndim != 2 or corpus_embeddings.shape[0] != len(doc_ids):
                 raise ValueError(f"Embeddings shape mismatch: Expected ({len(doc_ids)}, dim), Got {corpus_embeddings.shape}")
            # Ensure float32 for FAISS
            if corpus_embeddings.dtype != np.float32:
                 self.logger.warning(f"Corpus embeddings dtype is {corpus_embeddings.dtype}, converting to float32 for FAISS.")
                 corpus_embeddings = corpus_embeddings.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Error loading or validating corpus embeddings: {e}", exc_info=True)
            raise

        self.logger.info(f"Loaded {len(corpus)} documents and embeddings with shape {corpus_embeddings.shape}")
        return corpus, doc_ids, corpus_embeddings

    def search(self, query, history_queries=None, user_id=None, corpus_embeddings=None, doc_ids=None, index=None, top_k=10):
        """使用 FAISS 索引搜索相关文档"""
        if corpus_embeddings is None or doc_ids is None or index is None:
            self.logger.error("Corpus embeddings, document IDs, or FAISS index not provided")
            return []

        try:
            query_vector = self._get_personalized_vector(query, history_queries, user_id)
            query_vector_np = query_vector.reshape(1, -1).astype(np.float32) # FAISS expects float32

            # Search the FAISS index
            scores, indices = index.search(query_vector_np, top_k)

            results = []
            if scores.size > 0 and indices.size > 0:
                 for score, idx in zip(scores[0], indices[0]):
                     if idx != -1 and idx < len(doc_ids): # Check for valid index
                         doc_id = doc_ids[idx]
                         results.append({"text_id": doc_id, "score": float(score)})
                     # else: self.logger.debug(f"Invalid index {idx} returned by FAISS.")

            return results
        except Exception as e:
            self.logger.error(f"Error during FAISS search for user {user_id}: {e}", exc_info=True)
            return []

    def process_dataset(self, input_file, output_file, corpus_path, corpus_embeddings_path):
        """处理数据集 (JSONL 格式)"""
        if not os.path.exists(input_file):
            self.logger.error(f"Input file not found: {input_file}")
            return
        if not os.path.exists(corpus_path):
            self.logger.error(f"Corpus file not found: {corpus_path}")
            return
        if not os.path.exists(corpus_embeddings_path):
            self.logger.error(f"Corpus embeddings file not found: {corpus_embeddings_path}")
            return

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.logger.info(f"Processing queries from {input_file}")

        try:
            # 加载语料库和嵌入
            corpus, doc_ids, corpus_embeddings = self.load_corpus_and_embeddings(
                corpus_path, corpus_embeddings_path
            )

            # 创建 FAISS 索引
            dimension = corpus_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension) # Using Inner Product (cosine on normalized vectors)
            index.add(corpus_embeddings)
            self.logger.info(f"FAISS index created with {index.ntotal} vectors.")


            # 读取输入查询
            queries_data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        queries_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                         self.logger.warning(f"Skipping invalid JSON line in input: {line[:100]}...")
                         continue

            # Removed continuity filtering logic

            # 按用户ID（或查询ID）分组处理
            # Assuming query_id represents a unique user/session for baseline purposes
            final_results = []
            processed_count = 0
            for item in tqdm(queries_data, desc="Processing queries"):
                query_id = str(item.get("query_id", ""))
                current_query = item.get("query", "")
                if not query_id or not current_query:
                    self.logger.warning(f"Skipping item due to missing query_id or query: {item}")
                    continue

                # 简单处理：将每个查询视为独立会话的开始，没有历史
                # 对于需要历史的基线，需要调整数据格式或这里的逻辑
                history_queries = [] # RPMN/HTPS might need actual history if available in data

                # 执行搜索
                search_results = self.search(
                    current_query,
                    history_queries=history_queries, # Pass empty history for simplicity here
                    user_id=query_id, # Use query_id as user_id
                    corpus_embeddings=corpus_embeddings, # Pass embeddings directly
                    doc_ids=doc_ids,
                    index=index, # Pass the FAISS index
                    top_k=10 # Or get from args if needed
                )

                # 获取完整文档信息
                full_results = []
                for res in search_results:
                    doc_id_res = res["text_id"]
                    if doc_id_res in corpus:
                        doc_data = corpus[doc_id_res]
                        full_results.append({
                            "text_id": doc_id_res,
                            "title": doc_data.get("title", ""),
                            "text": doc_data.get("text", ""),
                            "score": res["score"]
                            # Add full_paper if needed and available
                            # "full_paper": doc_data.get("full_paper", None)
                        })
                    else:
                         self.logger.warning(f"Document ID {doc_id_res} from search results not found in loaded corpus.")


                # 创建结果对象
                result = {
                    "query_id": query_id,
                    "query": current_query,
                    "results": full_results # Use 'results' key for consistency
                }
                final_results.append(result)
                processed_count += 1

                # 定期保存结果 (可选)
                # if processed_count % 100 == 0:
                #     with open(output_file, 'w', encoding='utf-8') as f_out:
                #         for r in final_results:
                #             f_out.write(json.dumps(r, ensure_ascii=False) + '\n')
                #     self.logger.info(f"Saved progress: {processed_count} queries processed.")

            # 最终保存
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for result in final_results:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

            self.logger.info(f"Processed {processed_count} queries. Results saved to {output_file}")

        except ImportError:
             self.logger.error("FAISS library not found. Please install it (`pip install faiss-cpu` or `pip install faiss-gpu`). Exiting.")
        except Exception as e:
            self.logger.error(f"Error processing dataset: {e}", exc_info=True)
            raise

def main():
    """RPMN基线方法主函数"""
    import argparse
    # Setup logging basic config in case not already set
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='RPMN Baseline (No Continuity/Parquet)')
    parser.add_argument('--input_file', type=str, required=True, help='Input queries file path (JSONL format)')
    parser.add_argument('--output_file', type=str, required=True, help='Output results file path (JSONL)')
    parser.add_argument('--corpus_path', type=str, required=True, help='Corpus file path (JSONL format)')
    parser.add_argument('--corpus_embeddings_path', type=str, required=True, help='Corpus embeddings file path (.npy)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the encoder model (e.g., GTE)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for encoding')
    parser.add_argument('--device', type=str, default=None, help='Device (e.g., cuda:0, cpu)')
    # Removed continuity and corpus_format args

    args = parser.parse_args()

    # Determine device if not specified
    if args.device is None:
         try:
             import torch
             args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
         except ImportError:
             args.device = "cpu"
         logging.info(f"Device not specified, using: {args.device}")


    # 创建RPMN实例
    try:
        # Check FAISS installation before proceeding
        try:
            import faiss
        except ImportError:
             logging.error("FAISS library not found. Please install it (`pip install faiss-cpu` or `pip install faiss-gpu`).")
             sys.exit(1)

        rpmn = RPMNBaseline(
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size
        )
        # 处理数据集
        rpmn.process_dataset(
            args.input_file,
            args.output_file,
            args.corpus_path,
            args.corpus_embeddings_path,
        )
    except Exception as e:
         logging.error(f"RPMN execution failed: {e}", exc_info=True)
         sys.exit(1)


if __name__ == '__main__':
    # Ensure necessary libraries are installed before running main
    try:
        import faiss
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        print(f"Error: Missing required library - {e}. Please install necessary packages (torch, transformers, faiss-cpu/faiss-gpu).")
        sys.exit(1)
    main()
