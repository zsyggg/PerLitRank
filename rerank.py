# rerank.py
import gc
import json
import logging
import os
import torch
import argparse
import re
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from utils import get_config, logger
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('Rerank_Fallback')
    logger.error(f"Failed to import from utils: {e}.")
    class DummyConfig: 
        device="cpu"; reranker_path=None; dataset_name="unknown"; use_personalized_features=True;
        reranker_type="gte"; length_suffix="_L200";
        personalized_queries_path="results/unknown/personalized_queries_L200_wq.jsonl"; # 默认带查询
        final_results_path="results/unknown/ranked_gte_personalized_wq_L200.jsonl";
        retrieved_results_path="results/unknown/retrieved.jsonl";
        initial_top_k=50; final_top_k=10; batch_size=8; reranker_max_length=512;
        dataset_type="unknown"; local_model_max_tokens=512;
        test_query_limit: Optional[int] = None # 添加 test_query_limit
        narrative_query_mode_suffix_for_rerank_input = "_wq" # 添加默认值
        def __getattr__(self, name): return None
    def get_config(): return DummyConfig()

from tqdm import tqdm

def load_personalized_features(personalized_queries_path_with_suffix: str) -> Dict[str, Dict[str, Any]]:
    """
    加载个性化特征 (叙述文本)。
    如果文件中存在 'tagged_memory_features'，它们仍会被加载，
    但重排器提示将只使用叙述 'personalized_features'。
    """
    features_data = {}
    if not os.path.exists(personalized_queries_path_with_suffix):
        logger.warning(f"个性化特征文件未找到: {personalized_queries_path_with_suffix}。")
        return features_data
    try:
        with open(personalized_queries_path_with_suffix, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    p_text = data.get("personalized_features", "")
                    tagged_list = data.get("tagged_memory_features", []) # 为了数据完整性而加载
                    if query_id:
                        features_data[query_id] = {
                            "personalized_features": p_text if "Error" not in p_text else "",
                            "tagged_memory_features": tagged_list 
                        }
                except json.JSONDecodeError: 
                    logger.warning(f"跳过 {personalized_queries_path_with_suffix} 中的无效JSON")
        logger.info(f"从 {personalized_queries_path_with_suffix} 加载了 {len(features_data)} 个查询的特征")
    except Exception as e: 
        logger.error(f"加载 {personalized_queries_path_with_suffix} 时出错: {e}")
    return features_data

def load_retrieved_results(retrieved_results_path: str) -> Dict[str, Dict]:
    """加载初始检索结果"""
    retrieved_data = {}
    if not os.path.exists(retrieved_results_path):
        logger.error(f"检索结果文件未找到: {retrieved_results_path}"); return retrieved_data
    try:
        with open(retrieved_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    if query_id: 
                        retrieved_data[query_id] = {
                            "query": data.get("query", ""), 
                            "results": data.get("results", [])
                        }
                except json.JSONDecodeError: 
                    logger.warning(f"跳过 {retrieved_results_path} 中的无效JSON")
    except Exception as e: 
        logger.error(f"加载 {retrieved_results_path} 时出错: {e}")
    return retrieved_data

class RerankerPromptFormatter:
    def format_input(self, query: str,
                     personalized_features_text: Optional[str], 
                     document_text_dict: Dict[str, Any],
                     reranker_type: str,
                     rerank_mode: str): # rerank_mode 决定是否使用 personalized_features_text
        doc_content = " ".join(filter(None, [document_text_dict.get("title",""), document_text_dict.get("text",""), document_text_dict.get("full_paper","")])).strip().replace("\n"," ")
        
        if reranker_type == "jina":
            query_part = f"Query: {query}"
            if rerank_mode == "personalized" and personalized_features_text:
                query_part += f" User Background: {personalized_features_text}"
            return (query_part.strip(), doc_content)
        else: # GTE / MiniCPM
            if rerank_mode == "personalized" and personalized_features_text:
                template = self._get_personalized_template()
                formatted_text = template.format(
                    query=query,
                    personalized_features=personalized_features_text,
                    document_text=doc_content
                )
            else: 
                template = self._get_baseline_template()
                formatted_text = template.format(query=query, document_text=doc_content)

            if reranker_type == "minicpm":
                # MiniCPM 的特定格式化逻辑
                instruction_match = re.search(r"Task:(.*?)\n\n", formatted_text, re.DOTALL)
                query_doc_part = re.sub(r"Task:.*?\n\n", "", formatted_text, count=1, flags=re.DOTALL).strip()
                if instruction_match:
                    instruction = instruction_match.group(1).strip()
                    final_instruction = f"Instruction: {instruction}"
                    if rerank_mode == "personalized" and personalized_features_text:
                         final_instruction += " Consider user background for relevance."
                    formatted_text = f"<s>{final_instruction}\n{query_doc_part}</s>"
                else: 
                    base_instruction = "Evaluate document relevance for query."
                    if rerank_mode == "personalized" and personalized_features_text:
                        base_instruction = "Considering user background, evaluate document relevance for the query."
                    if rerank_mode == "personalized" and personalized_features_text:
                        formatted_text = f"<s>Instruction: {base_instruction}\nQuery: {query}\nUser Background: {personalized_features_text}\nDocument: {doc_content}</s>"
                    else:
                        formatted_text = f"<s>Instruction: {base_instruction}\nQuery: {query}\nDocument: {doc_content}</s>"
            return formatted_text

    def _get_personalized_template(self):
        return """
Task: Evaluate document relevance for the query, considering the user's background and interests.
Aspects: 1. Technical relevance. 2. Alignment with user's profile. 3. Usefulness.

Query: {query}

User Background and Interests: 
{personalized_features}

Document: {document_text}"""

    def _get_baseline_template(self):
        return "Task: Evaluate document relevance for the query based on content.\n\nQuery: {query}\n\nDocument: {document_text}"

def batch_rerank_docs(model, tokenizer, q_text: str,
                      pf_text: Optional[str], 
                      docs: List[Dict[str, Any]], dev: str,
                      r_type:str, r_mode:str, b_size:int, max_len:int):
    if not docs: return []
    formatter = RerankerPromptFormatter()
    reranked = []
    for i in range(0, len(docs), b_size):
        batch_data = docs[i:i+b_size]; batch_inputs = []; orig_docs_batch = []
        for doc_d in batch_data:
            fmt = formatter.format_input(
                query=q_text,
                personalized_features_text=(pf_text if r_mode != 'baseline' and pf_text else None),
                document_text_dict=doc_d,
                reranker_type=r_type,
                rerank_mode=r_mode
            )
            if fmt: batch_inputs.append(fmt); orig_docs_batch.append(doc_d)
        if not batch_inputs: continue
        try:
            with torch.no_grad():
                if r_type == "jina": 
                    scores = model.compute_score(batch_inputs, max_length=max_len)
                else:
                    inputs = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors='pt', max_length=max_len).to(dev)
                    scores = model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
            
            for j, doc_d_orig in enumerate(orig_docs_batch):
                s_val = float(scores[j]) if j < len(scores) else 0.0
                res_doc = {"text_id":doc_d_orig.get("text_id",""), "title":doc_d_orig.get("title",""), "text":doc_d_orig.get("text",""), "score":s_val}
                if "full_paper" in doc_d_orig and doc_d_orig["full_paper"]: 
                    res_doc["full_paper"] = doc_d_orig["full_paper"]
                reranked.append(res_doc)
        except Exception as e:
            logger.error(f"批处理重排错误 (批次 {i}): {e}", exc_info=True)
            for doc_d_orig in orig_docs_batch: # 回退
                 res_doc = {"text_id":doc_d_orig.get("text_id",""), "title":doc_d_orig.get("title",""), "text":doc_d_orig.get("text",""), "score":doc_d_orig.get("score",0.0)}
                 if "full_paper" in doc_d_orig and doc_d_orig["full_paper"]: 
                     res_doc["full_paper"] = doc_d_orig["full_paper"]
                 reranked.append(res_doc)
    return reranked

def get_model_and_tokenizer(config, reranker_type):
    model_path = config.reranker_path # reranker_path 应该已在 config.update 中根据 reranker_type 设置
    logger.info(f"正在加载 {reranker_type} 从 {model_path} 到 {config.device}")
    dtype = torch.float16 if torch.cuda.is_available() and "cuda" in str(config.device) else torch.float32
    try:
        trust_remote = reranker_type in ["jina", "minicpm", "gte"] # GTE有时也需要
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=trust_remote, torch_dtype=dtype).to(config.device).eval()
        return model, tokenizer
    except Exception as e: 
        logger.error(f"加载模型 {model_path} 时出错: {e}", exc_info=True); raise

def run_reranking(config):
    rerank_mode = "personalized" if config.use_personalized_features else "baseline"
    final_output_path = config.final_results_path # 这个路径现在会包含 _topX 后缀
    
    # 叙述文件路径取决于 config.narrative_query_mode_suffix_for_rerank_input
    # 和 config.length_suffix
    personalized_queries_path_for_run = os.path.join(
        config.results_dir,
        config.dataset_name,
        f"{config._personalized_queries_base}{config.length_suffix}{config.narrative_query_mode_suffix_for_rerank_input}.jsonl"
    )

    logger.info(f"--- 文档重排 (提示简化版) ---")
    logger.info(f"数据集: {config.dataset_name}, 重排器: {config.reranker_type}, 模式: {rerank_mode}")
    logger.info(f"个性化叙述特征来源: {personalized_queries_path_for_run}")
    logger.info(f"最终重排结果将保存到: {final_output_path}")
    if config.test_query_limit is not None:
        logger.info(f"测试模式: 将尝试限制处理约 {config.test_query_limit} 个查询。")

    retrieved_data = load_retrieved_results(config.retrieved_results_path)
    all_p_data = {} 
    if config.use_personalized_features:
        all_p_data = load_personalized_features(personalized_queries_path_for_run)
        if not all_p_data and rerank_mode == "personalized":
             logger.warning("请求了个性化特征但未加载任何特征。实际模式: 基线。")
             rerank_mode = "baseline" # 动态调整模式

    if not retrieved_data: 
        logger.error("没有检索结果。退出。"); return

    try: 
        model, tokenizer = get_model_and_tokenizer(config, config.reranker_type)
    except Exception as e: 
        logger.error(f"加载重排器模型失败: {e}。退出。"); return

    # --- 确定要处理的查询ID ---
    initial_query_ids_from_retrieval = list(retrieved_data.keys())
    queries_to_process = []

    if config.use_personalized_features:
        # 如果使用个性化特征，则只处理那些有叙述的查询
        # 由于叙述文件本身可能已受 test_query_limit 限制，这里的交集也会受限
        queries_with_narratives = set(all_p_data.keys())
        queries_to_process = [qid for qid in initial_query_ids_from_retrieval if qid in queries_with_narratives]
        logger.info(f"个性化模式: 将处理 {len(queries_to_process)} 个在检索结果和叙述文件中都存在的查询。")
        if config.test_query_limit is not None and len(queries_to_process) > config.test_query_limit:
            logger.warning(f"个性化模式下，具有叙述的查询数量 ({len(queries_to_process)}) 超过 test_query_limit ({config.test_query_limit})。将按叙述文件中的数量处理。")
            # 实际上，由于叙述文件在生成时已限制，这里 queries_to_process 通常不会超过 test_query_limit
    else: # 基线模式
        if config.test_query_limit is not None and config.test_query_limit > 0:
            if len(initial_query_ids_from_retrieval) > config.test_query_limit:
                queries_to_process = initial_query_ids_from_retrieval[:config.test_query_limit]
                logger.info(f"基线测试模式: 从检索结果中选取前 {len(queries_to_process)} 个查询进行处理。")
            else:
                queries_to_process = initial_query_ids_from_retrieval
                logger.info(f"基线测试模式: test_query_limit 大于或等于可用查询数。将处理所有 {len(queries_to_process)} 个检索到的查询。")
        else:
            queries_to_process = initial_query_ids_from_retrieval
            logger.info(f"基线模式: 将处理所有 {len(queries_to_process)} 个检索到的查询。")

    # MedCorpus 特定过滤 (如果适用)
    if config.dataset_type == "medcorpus":
        original_count_before_medcorpus_filter = len(queries_to_process)
        queries_to_process = [qid for qid in queries_to_process if qid.endswith("_3")]
        logger.info(f"MedCorpus: 从 {original_count_before_medcorpus_filter} 个查询中筛选出 {len(queries_to_process)} 个第3轮查询进行重排。")

    if not queries_to_process:
        logger.warning("没有查询需要处理。退出重排。")
        return
    # --- 查询ID确定结束 ---

    final_data_list = []
    for qid in tqdm(queries_to_process, desc=f"重排查询 (总计 {len(queries_to_process)})"):
        if qid not in retrieved_data: 
            logger.warning(f"查询ID {qid} 在已过滤的待处理列表中，但未在 retrieved_data 中找到。跳过。")
            continue
        
        q_info = retrieved_data[qid]; q_text = q_info["query"]
        cand_docs = q_info["results"][:config.initial_top_k] # 取初始 top_k 进行重排

        p_data_q = all_p_data.get(qid, {})
        pf_text_q = p_data_q.get("personalized_features", "") if config.use_personalized_features else ""
        
        eff_mode_for_query = rerank_mode # 初始有效模式
        if rerank_mode == "personalized" and not pf_text_q: # 如果是个性化模式但该查询没有叙述
            eff_mode_for_query = "baseline"
            logger.debug(f"查询 {qid}: 未找到个性化叙述文本，此查询将使用基线模式。")

        max_len = getattr(config, 'reranker_max_length', 512)

        ranked_docs_output = batch_rerank_docs(
            model=model, tokenizer=tokenizer, q_text=q_text,
            pf_text=pf_text_q, 
            docs=cand_docs, dev=config.device,
            r_type=config.reranker_type, r_mode=eff_mode_for_query, # 使用此查询的有效模式
            b_size=config.batch_size, max_len=max_len
        )
        # 根据 config.final_top_k 排序和截断
        final_ranked_docs_for_query = sorted(ranked_docs_output, key=lambda x: x['score'], reverse=True)[:config.final_top_k]
        
        final_data_list.append({
            "query_id": qid, "query": q_text,
            "personalized_features_used": pf_text_q if config.use_personalized_features else "N/A (Baseline)",
            "rerank_mode_effective": eff_mode_for_query,
            "ranked_results": final_ranked_docs_for_query
        })

    try:
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        with open(final_output_path, 'w', encoding='utf-8') as fout:
            for data_item in final_data_list: 
                fout.write(json.dumps(data_item, ensure_ascii=False) + "\n")
        logger.info(f"重排完成。结果已保存到 {final_output_path}")
    except IOError as e: 
        logger.error(f"写入 {final_output_path} 失败: {e}")

    del model, tokenizer; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info("--- 文档重排完成 ---")

def main():
    # 这部分主要用于脚本直接被调用时 (python rerank.py ...)
    # 如果是通过 run.py --mode rerank 调用，则 run.py 中的 main 会准备 config
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="PersLitRank 重排器 (独立运行)")
        parser.add_argument("--dataset_name", type=str, required=True)
        parser.add_argument("--reranker_type", type=str, default="minicpm")
        parser.add_argument("--reranker_path", type=str, help="重排器模型的显式路径")
        parser.add_argument("--data_dir", type=str, default="/workspace/PerMed/data")
        parser.add_argument("--results_dir", type=str, default="./results")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--initial_top_k", type=int, default=50)
        parser.add_argument("--final_top_k", type=int, default=10, help="重排后保留的文档数")
        parser.add_argument("--max_length", type=int, default=512, help="重排器最大序列长度")
        parser.add_argument("--gpu_id", type=int, default=0)
        parser.add_argument("--use_personalized_features", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--personalized_text_target_length", type=int, default=200, choices=[100,150,200,300])
        parser.add_argument("--narrative_query_mode_suffix_for_rerank_input", type=str, default="_wq", choices=["_wq", "_nq"], help="用于加载的叙述文件后缀")
        parser.add_argument("--test_query_limit", type=int, default=None, help="测试时限制处理的查询数量")


        args = parser.parse_args()

        config = get_config() # 获取全局config实例
        config.update(args) # 使用命令行参数更新它
        
        # reranker_max_length 和 reranker_path 可能需要特殊处理，因为它们不在 Config 的标准循环更新属性列表中
        if hasattr(args, 'max_length') and args.max_length is not None: # 对应 reranker_max_length
             config.reranker_max_length = args.max_length
        if hasattr(args, 'reranker_path') and args.reranker_path: 
             config.reranker_path = args.reranker_path
        
        run_reranking(config)
    else: # 如果是被 run.py 导入
        config = get_config() # 获取已由 run.py 更新的全局 config
        # 确保 reranker_max_length 有一个值，如果 run.py 没有设置它
        if not hasattr(config, 'reranker_max_length') or config.reranker_max_length is None:
             config.reranker_max_length = getattr(config, 'local_model_max_tokens', 512) # 使用一个回退值
        run_reranking(config)

if __name__ == "__main__":
    main()
