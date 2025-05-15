# rerank_history_baseline.py (AttributeError 和 TypeError 修正)
import gc
import json
import logging
import os
import torch
import argparse
from typing import Dict, List, Optional

try:
    from utils import get_config, logger, Query, load_queries as load_queries_util
    from rerank import RerankerPromptFormatter, get_model_and_tokenizer, batch_rerank_docs as original_batch_rerank_docs
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("RerankHistoryBaseline_Fallback")
    logger.error(f"无法导入模块: {e}。请确保 utils.py 和 rerank.py 可访问。")
    class Query:
        def __init__(self, query_id: str, query: str, topic_id: Optional[str] = None, turn_id: Optional[int] = None):
            self.query_id = query_id; self.query = query; self.topic_id = topic_id; self.turn_id = turn_id
    class DummyConfig:
        def __init__(self):
            self.device = 'cpu'; self.base_data_dir = "./data"; self.results_dir = "./results"
            self.batch_size = 8; self.initial_top_k = 50; self.final_top_k = 10
            self.reranker_max_length = 512; self.local_model_max_tokens = 512
            self.dataset_name = "unknown"; self.reranker_type = "unknown"
            self.jina_reranker_path = "/workspace/.cache/modelscope/hub/jinaai/jina-reranker-v2-base-multilingual" # 添加默认路径
            self.minicpm_reranker_path = "/workspace/.cache/modelscope/hub/OpenBMB/MiniCPM-Reranker-Light" # 添加默认路径
            self.reranker_path = "/workspace/.cache/modelscope/hub/iic/gte_passage-ranking_multilingual-base" # 添加默认路径


        def update(self, args_dict_or_obj):
            args_dict = vars(args_dict_or_obj) if isinstance(args_dict_or_obj, argparse.Namespace) else args_dict_or_obj
            for key, value in args_dict.items():
                if value is not None: setattr(self, key, value)
        def __getattr__(self, name):
             if name == 'medcorpus_eval_turn': return 2
             if name == 'history_window_litsearch': return 0
             if name == 'history_window_medcorpus': return 2
             if name == 'history_window_default': return 1
             if name == 'retrieved_results_path':
                 return os.path.join(getattr(self, 'results_dir', './results'), getattr(self, 'dataset_name', 'unknown'), "retrieved.jsonl")
             raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    def get_config(): return DummyConfig()
    def load_queries_util(config) -> List[Query]: return []
    class RerankerPromptFormatter:
         def format_input(self, query, personalized_features, document_text, reranker_type, rerank_mode):
            doc_text_cleaned = " ".join(filter(None, [document_text.get("title", ""), document_text.get("text", "")])).strip()
            if reranker_type == "jina": return (query, doc_text_cleaned)
            elif reranker_type == "minicpm":
                 return f"<s>{self._get_baseline_template_minicpm().format(query=query, document_text=doc_text_cleaned)}</s>"
            else: return self._get_baseline_template().format(query=query, document_text=doc_text_cleaned)
         def _get_baseline_template(self): return "Query: {query}\nDocument: {document_text}"
         def _get_baseline_template_minicpm(self): return "Instruction: Evaluate the relevance of the document for the query.\nQuery: {query}\nDocument: {document_text}"
    def get_model_and_tokenizer(config, reranker_type): raise NotImplementedError
    def original_batch_rerank_docs(*args, **kwargs): raise NotImplementedError

from tqdm import tqdm

def get_historical_queries_text(current_query: Query,
                                all_queries_dict: Dict[str, Query],
                                history_window: int) -> str:
    if not current_query.topic_id or current_query.turn_id is None or current_query.turn_id == 0 or history_window == 0:
        return ""
    history_texts = []
    for i in range(max(0, current_query.turn_id - history_window), current_query.turn_id):
        historical_query_id = f"{current_query.topic_id}_{i}"
        if historical_query_id in all_queries_dict:
            history_texts.append(all_queries_dict[historical_query_id].query)
        else:
            logger.debug(f"当前查询 {current_query.query_id} 的历史查询ID {historical_query_id} 未在 all_queries_dict 中找到")
    return " [SEP] ".join(history_texts) if history_texts else ""

def load_all_queries_as_dict(config, dataset_name_for_load: str) -> Dict[str, Query]:
    original_dataset_name = getattr(config, 'dataset_name', None)
    original_queries_path = getattr(config, 'queries_path', None)
    config.dataset_name = dataset_name_for_load
    queries_file_path = os.path.join(getattr(config, 'base_data_dir', './data'), dataset_name_for_load, "queries.jsonl")
    config.queries_path = queries_file_path 
    logger.info(f"正在加载所有查询以构建历史记录，来源: {config.queries_path}")
    queries_list = []
    try:
        queries_list = load_queries_util(config)
    except Exception as e:
        logger.error(f"为 {dataset_name_for_load} 加载查询时出错: {e}")
    finally:
        if original_dataset_name is not None: config.dataset_name = original_dataset_name
        if original_queries_path is not None: config.queries_path = original_queries_path
    if not queries_list: logger.warning(f"为 {dataset_name_for_load} 构建历史映射时未加载任何查询。")
    return {q.query_id: q for q in queries_list}


def run_history_baseline_reranking_for_dataset(
        config,
        dataset_name: str,
        reranker_type_arg: str, # 使用不同的参数名以避免与 config.reranker_type 混淆
        all_queries_map_for_dataset: Dict[str, Query]
    ):
    if dataset_name.lower() == "litsearch":
        history_window = getattr(config, 'history_window_litsearch', 0)
    elif dataset_name.lower() == "medcorpus":
        history_window = getattr(config, 'history_window_medcorpus', 2)
    else:
        history_window = getattr(config, 'history_window_default', 1)
    # 确保 window_suffix 总是包含窗口大小，即使是0
    window_suffix = f"_win{history_window}"


    logger.info(f"--- 开始为 {dataset_name} 进行历史感知基线重排 ---")
    logger.info(f"重排器: {reranker_type_arg}, 历史窗口: {history_window}")

    original_config_dataset_name = config.dataset_name
    original_reranker_type_in_config = config.reranker_type # 保存 config 中当前的 reranker_type
    original_reranker_path_in_config = config.reranker_path # 保存 config 中当前的 reranker_path
    original_retrieved_results_path = config.retrieved_results_path
    
    model = None 
    tokenizer = None

    try:
        # 临时设置 config 中的 dataset_name 和 reranker_type 以便加载正确的模型和数据
        config.dataset_name = dataset_name
        config.reranker_type = reranker_type_arg # 将命令行传入的 reranker 类型设置到 config 中
                                                # 以便 get_model_and_tokenizer 使用

        current_retrieved_path = os.path.join(getattr(config, 'results_dir', './results'), dataset_name, "retrieved.jsonl")
        # 注意：我们不直接修改 config.retrieved_results_path，而是在加载时使用 current_retrieved_path
        
        retrieved_data = {}
        if not os.path.exists(current_retrieved_path):
            logger.error(f"未找到 {dataset_name} 的检索结果文件: {current_retrieved_path}")
            return
        with open(current_retrieved_path, 'r', encoding='utf-8') as f:
            for line in f:
                try: 
                     data = json.loads(line)
                     query_id_str = str(data["query_id"])
                     retrieved_data[query_id_str] = {
                         "query_text_original": data["query"],
                         "results": data.get("results", []) 
                     }
                except json.JSONDecodeError:
                     logger.warning(f"跳过 {current_retrieved_path} 中的无效JSON行")
                     continue
                except KeyError:
                     logger.warning(f"跳过 {current_retrieved_path} 中缺少 'query_id' 或 'query' 的行")
                     continue
        logger.info(f"为 {dataset_name} 加载了 {len(retrieved_data)} 条检索结果。")
        if not retrieved_data:
            logger.warning(f"未为 {dataset_name} 加载检索数据。跳过重排。")
            return

        # --- 重排器路径确定 ---
        # 如果 config.reranker_path 没有被命令行参数 reranker_path 覆盖，
        # 或者 reranker_type_arg 与 config 中上次检查的类型不同，则更新它
        if not config.reranker_path or config.reranker_type != getattr(config, '_last_reranker_type_for_path_check', None):
            if config.reranker_type == "jina":
                 config.reranker_path = getattr(config, 'jina_reranker_path')
            elif config.reranker_type == "minicpm":
                 config.reranker_path = getattr(config, 'minicpm_reranker_path')
            elif config.reranker_type == "gte":
                 config.reranker_path = getattr(config, 'reranker_path') # GTE的路径通常是默认的config.reranker_path
            else:
                 logger.error(f"未知的重排器类型 '{config.reranker_type}' 且未提供显式路径。")
                 raise ValueError(f"无效的重排器类型: {config.reranker_type}")
            logger.info(f"为 {config.reranker_type} 使用确定的路径: {config.reranker_path}")
            setattr(config, '_last_reranker_type_for_path_check', config.reranker_type)
        else:
            logger.info(f"为 {config.reranker_type} 使用已配置的重排器路径: {config.reranker_path}")
        
        model, tokenizer = get_model_and_tokenizer(config, config.reranker_type) # 使用 config.reranker_type
        formatter = RerankerPromptFormatter() 
        final_results_data = []
        queries_to_process = list(retrieved_data.keys())

        if dataset_name.lower() == "medcorpus":
            medcorpus_eval_turn_index = getattr(config, 'medcorpus_eval_turn', 2) 
            medcorpus_eval_turn_suffix = f"_{medcorpus_eval_turn_index + 1}" 
            logger.info(f"应用MedCorpus过滤: 仅处理轮次 {medcorpus_eval_turn_index + 1} 的查询 (ID以 '{medcorpus_eval_turn_suffix}' 结尾)。")
            queries_to_process = [qid for qid in queries_to_process if qid.endswith(medcorpus_eval_turn_suffix)]
            logger.info(f"MedCorpus: 处理 {len(queries_to_process)} 个轮次 {medcorpus_eval_turn_index + 1} 的查询。")

        if not queries_to_process:
            logger.warning(f"过滤后没有要为 {dataset_name} 处理的查询。跳过。")
            return

        for query_id in tqdm(queries_to_process, desc=f"正在重排 {dataset_name} ({config.reranker_type})"):
            if query_id not in all_queries_map_for_dataset:
                logger.warning(f"查询ID {query_id} 未在 {dataset_name} 的 all_queries_map 中找到。跳过。")
                continue
            if query_id not in retrieved_data: 
                logger.warning(f"查询ID {query_id} 未在 retrieved_data 中找到。跳过。")
                continue

            current_query_obj = all_queries_map_for_dataset[query_id]
            if current_query_obj.turn_id is None:
                 logger.warning(f"查询 {query_id} 的 turn_id 无效 (None)。无法确定历史。跳过。")
                 continue

            candidate_docs = retrieved_data[query_id]["results"][:config.initial_top_k]
            if not candidate_docs:
                logger.debug(f"查询ID {query_id} 没有候选文档。跳过此查询的重排。")
                continue

            historical_text = get_historical_queries_text(current_query_obj, all_queries_map_for_dataset, history_window)
            combined_query_for_reranker = current_query_obj.query
            if historical_text:
                combined_query_for_reranker = historical_text + " [SEP] " + current_query_obj.query
            
            reranker_max_len = getattr(config, 'reranker_max_length', 512)

            # 修正调用 original_batch_rerank_docs 的参数名
            ranked_docs = original_batch_rerank_docs(
                model=model,
                tokenizer=tokenizer,
                q_text=combined_query_for_reranker, # 使用 q_text
                pf_text=None,                       # 使用 pf_text，对于基线模式为 None
                docs=candidate_docs,
                dev=config.device,
                r_type=config.reranker_type,        # 使用 config.reranker_type
                r_mode="baseline",
                b_size=config.batch_size,
                max_len=reranker_max_len
            )

            final_ranked_docs = sorted(ranked_docs, key=lambda x: x['score'], reverse=True)[:config.final_top_k]
            output_data = {
                "query_id": query_id, "query": current_query_obj.query, 
                "combined_query_for_reranker": combined_query_for_reranker, 
                "ranked_results": final_ranked_docs
            }
            final_results_data.append(output_data)

        if not final_results_data:
            logger.info(f"未为 {dataset_name} ({config.reranker_type}) 生成任何结果。不会创建输出文件。")
            return

        # 此脚本生成特定名称的输出文件
        output_filename = f"ranked_{config.reranker_type}_history_baseline{window_suffix}.jsonl"
        results_base_dir = getattr(config, 'results_dir', './results')
        dataset_results_dir = os.path.join(results_base_dir, dataset_name) 
        os.makedirs(dataset_results_dir, exist_ok=True)
        final_output_path_for_script = os.path.join(dataset_results_dir, output_filename)

        with open(final_output_path_for_script, 'w', encoding='utf-8') as fout:
            for data in final_results_data:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        logger.info(f"{dataset_name} ({config.reranker_type}) 的历史感知基线重排完成。结果已保存到 {final_output_path_for_script}")

    except Exception as e:
        logger.error(f"为 {dataset_name} ({config.reranker_type}) 重排期间发生错误: {e}", exc_info=True)
    finally:
        if 'original_config_dataset_name' in locals(): config.dataset_name = original_config_dataset_name
        if 'original_reranker_type_in_config' in locals(): config.reranker_type = original_reranker_type_in_config
        if 'original_reranker_path_in_config' in locals(): config.reranker_path = original_reranker_path_in_config
        if 'original_retrieved_results_path' in locals(): config.retrieved_results_path = original_retrieved_results_path
        if hasattr(config, '_last_reranker_type_for_path_check'): # 清理临时属性
            delattr(config, '_last_reranker_type_for_path_check')

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="为特定数据集和重排器运行历史感知的基线重排器")
    parser.add_argument("--dataset", type=str, required=True, help="要处理的数据集名称")
    parser.add_argument("--reranker", type=str, required=True, help="要使用的重排器类型 (jina, gte, minicpm)")
    parser.add_argument("--reranker_path", type=str, default=None, help="重排器模型的显式路径 (覆盖默认值)")
    parser.add_argument("--data_dir", type=str, default="./data", help="基础数据目录")
    parser.add_argument("--results_dir", type=str, default="./results", help="基础结果目录")
    parser.add_argument("--gpu_id", type=int, default=0, help="要使用的GPU ID。设置为-1表示CPU。")
    parser.add_argument("--history_window_litsearch", type=int, default=0, help="LitSearch历史基线的过去查询数量")
    parser.add_argument("--history_window_medcorpus", type=int, default=2, help="MedCorpus历史基线的过去查询数量")
    parser.add_argument("--history_window_default", type=int, default=1, help="其他数据集的默认过去查询数量")
    parser.add_argument("--medcorpus_eval_turn", type=int, default=2, help="对于MedCorpus，要评估的0索引轮次")
    parser.add_argument("--batch_size", type=int, default=8, help="重排的批处理大小")
    parser.add_argument("--initial_top_k", type=int, default=50, help="要重排的候选数量")
    parser.add_argument("--final_top_k", type=int, default=10, help="最终结果的数量")
    parser.add_argument("--max_length", type=int, default=512, help="重排器模型的最大序列长度")
    args = parser.parse_args()

    config = get_config()
    config.base_data_dir = args.data_dir
    config.results_dir = args.results_dir
    config.batch_size = args.batch_size
    config.initial_top_k = args.initial_top_k
    config.final_top_k = args.final_top_k
    config.reranker_max_length = args.max_length
    
    # 将命令行指定的 reranker 类型和路径更新到 config 中
    # reranker_type 将被 run_history_baseline_reranking_for_dataset 函数内部使用来设置 config.reranker_type
    # 如果命令行提供了 reranker_path，它会覆盖基于 reranker 类型的默认路径
    if args.reranker_path:
        config.reranker_path = args.reranker_path
    # config.reranker_type 会在 run_history_baseline_reranking_for_dataset 中被 args.reranker 覆盖

    config.history_window_litsearch = args.history_window_litsearch
    config.history_window_medcorpus = args.history_window_medcorpus
    config.history_window_default = args.history_window_default
    config.medcorpus_eval_turn = args.medcorpus_eval_turn 
    
    if args.gpu_id >= 0 and torch.cuda.is_available():
        config.device = f"cuda:{args.gpu_id}"
    else:
        config.device = "cpu"
    logger.info(f"使用设备: {config.device}")

    dataset_name_to_run = args.dataset
    reranker_type_from_arg = args.reranker # 从命令行获取 reranker 类型

    logger.info(f"准备为数据集运行: {dataset_name_to_run}，使用重排器: {reranker_type_from_arg}")
    logger.info(f"为数据集预加载查询: {dataset_name_to_run}")
    all_queries_map_for_current_dataset = load_all_queries_as_dict(config, dataset_name_to_run)

    if not all_queries_map_for_current_dataset:
        logger.error(f"为 {dataset_name_to_run} 加载查询失败。无法继续历史基线处理。")
        return 

    run_history_baseline_reranking_for_dataset(
        config,
        dataset_name_to_run,
        reranker_type_from_arg, # 传递命令行指定的 reranker 类型
        all_queries_map_for_current_dataset
    )
    logger.info(f"{dataset_name_to_run} 使用 {reranker_type_from_arg} 的历史感知基线重排运行完成。")

if __name__ == "__main__":
    if not logger.hasHandlers(): 
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
         if logger.name == "RerankHistoryBaseline_Fallback":
             logger = logging.getLogger("RerankHistoryBaseline_Main")
    main()
