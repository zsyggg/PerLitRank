# run.py
import os
import argparse
import logging
import time
import json 
from tqdm import tqdm 
import gc 
import torch 

from utils import get_config, logger, Query 
# PersonalizedGenerator 和 CognitiveFeatureExtractor 将在其各自的阶段函数中导入

def parse_args():
    parser = argparse.ArgumentParser(description="运行 PersLitRank 系统")
    parser.add_argument("--mode", type=str, 
                        choices=["all", "extract_cognitive_features", "generate_narratives", "retrieve", "rerank"],
                        required=True, help="处理模式")

    # 通用参数
    parser.add_argument("--dataset_name", type=str, default="MedCorpus", choices=["MedCorpus", "LitSearch", "CORAL"])
    parser.add_argument("--data_dir", type=str, default="/workspace/PerMed/data", help="基础数据目录")
    parser.add_argument("--results_dir", type=str, default="./results", help="基础结果目录")
    parser.add_argument("--gpu_id", type=int, default=0, help="当前进程的GPU ID")
    parser.add_argument("--llm_gpu_id", type=int, default=None, help="用于LLM的特定GPU ID")
    parser.add_argument("--batch_size", type=int, help="适用的批处理大小")

    # 'generate_narratives' 模式参数
    parser.add_argument("--personalized_text_target_length", type=int, default=200, choices=[100, 150, 200, 300],
                        help="个性化叙述的目标字符长度")
    parser.add_argument("--include_query_in_narrative_prompt", action=argparse.BooleanOptionalAction, default=True,
                        help="在个性化叙述生成提示中是否包含原始查询")

    # LLM 参数
    parser.add_argument("--local_model_path", type=str, help="本地LLM的路径")
    parser.add_argument("--enable_thinking", action="store_true", help="启用LLM思考模式")
    parser.add_argument("--temperature", type=float, help="LLM temperature")
    parser.add_argument("--top_p", type=float, help="LLM top_p")
    parser.add_argument("--top_k", type=int, help="LLM top_k") # LLM top_k for sampling
    parser.add_argument("--presence_penalty", type=float, help="LLM presence penalty")
    parser.add_argument("--repetition_penalty", type=float, help="LLM repetition penalty")
    
    # Reranker 参数
    parser.add_argument("--reranker_type", type=str, choices=["gte", "jina", "minicpm"], help="重排器类型")
    parser.add_argument("--reranker_path", type=str, help="重排器模型的显式路径")
    parser.add_argument("--reranker_max_length", type=int, help="重排器的最大序列长度")
    parser.add_argument("--use_personalized_features", action=argparse.BooleanOptionalAction, default=True,
                         help="在重排器中使用LLM生成的叙述")
    parser.add_argument("--narrative_query_mode_suffix_for_rerank_input", type=str, default="_wq", choices=["_wq", "_nq"],
                        help="用于重排时加载的个性化叙述文件的后缀")

    # 其他参数
    parser.add_argument("--feature_extractor", type=str, help="特征提取器类型")
    parser.add_argument("--memory_type", type=str, help="内存类型")
    parser.add_argument("--memory_components", type=str, help="要使用的内存组件，逗号分隔")
    parser.add_argument("--conversational", action="store_true", help="是否为对话模式")
    parser.add_argument("--initial_top_k", type=int, help="初始检索召回的文档数")
    parser.add_argument("--final_top_k", type=int, help="重排后最终保留的文档数 (例如 10 或 100 用于测试)")

    # 新增：测试时限制处理的查询数量
    parser.add_argument("--test_query_limit", type=int, default=None,
                        help="测试模式下限制处理的查询数量 (例如 100)。主要影响 generate_narratives 和 rerank 模式中实际处理的查询条目数。")

    return parser.parse_args()

def run_cognitive_feature_extraction_stage(config):
    """
    执行阶段1：认知特征提取。
    输出: config.cognitive_features_detailed_path
    """
    logger.info(f"--- 阶段1: 认知特征提取 ---")
    logger.info(f"数据集: {config.dataset_name}")
    logger.info(f"详细认知特征输出到: {config.cognitive_features_detailed_path}")
    stage_success = False
    try:
        from cognitive_retrieval import main as cognitive_main # cognitive_retrieval.py 包含 CognitiveFeatureExtractor
        cognitive_main() 
        logger.info(f"--- 阶段1: 认知特征提取完成 ---")
        stage_success = True
    except ImportError: 
        logger.error("无法导入 cognitive_retrieval (CognitiveFeatureExtractor)。跳过阶段1。", exc_info=True)
    except Exception as e: 
        logger.error(f"阶段1 (认知特征提取) 期间出错: {e}", exc_info=True)
    return stage_success

def run_narrative_generation_stage(config):
    """
    执行阶段2：个性化叙述生成。
    输入: config.cognitive_features_detailed_path
    输出: config.personalized_queries_path (带长度和查询模式后缀)
    如果 config.test_query_limit 被设置，则只处理前N条认知特征。
    """
    logger.info(f"--- 阶段2: 个性化叙述生成 ---")
    logger.info(f"从以下位置输入认知特征: {config.cognitive_features_detailed_path}")
    logger.info(f"叙述文件输出 (目标长度 {config.personalized_text_target_length}, 查询模式: {config.narrative_query_mode_suffix}): {config.personalized_queries_path}")
    logger.info(f"用于生成的LLM: {config.local_model_path} 在 {config.llm_device}上")
    logger.info(f"叙述生成提示中是否包含查询: {config.include_query_in_narrative_prompt}")
    if config.test_query_limit is not None:
        logger.info(f"测试模式: 将只为前 {config.test_query_limit} 个查询（来自认知特征文件）生成叙述。")
    stage_success = False

    try:
        from personalized_generator import PersonalizedGenerator
    except ImportError:
        logger.error("无法导入 PersonalizedGenerator。无法运行叙述生成阶段。", exc_info=True)
        return False

    if not os.path.exists(config.cognitive_features_detailed_path):
        logger.error(f"认知特征文件未找到: {config.cognitive_features_detailed_path}。请先运行 'extract_cognitive_features' 模式。")
        return False

    try:
        narrative_generator = PersonalizedGenerator(config=config)
        if narrative_generator.model is None or narrative_generator.tokenizer is None:
            logger.error("PersonalizedGenerator 模型/分词器未初始化。无法生成叙述。")
            return False
    except Exception as e:
        logger.error(f"初始化 PersonalizedGenerator 时出错: {e}", exc_info=True)
        return False

    generated_narratives_data = []
    queries_processed_count = 0
    
    try:
        cognitive_features_input_lines = []
        with open(config.cognitive_features_detailed_path, 'r', encoding='utf-8') as f_in:
            cognitive_features_input_lines = f_in.readlines()
        
        # 如果设置了 test_query_limit，则截断输入行
        # 注意：这假设 cognitive_features_detailed.jsonl 中的条目顺序是固定的，
        # 或者我们不关心具体是哪些查询被选中，只要数量正确即可。
        # 对于 MedCorpus 这种按 topic_id 后按 turn_id 排序处理认知特征的情况，
        # 这里的简单截断可能意味着只处理了部分 topic 的最后一轮。
        # 如果需要精确选择例如“前N个会话的最后一轮”，逻辑会更复杂。
        # 当前实现是简单地取文件的前N行。
        if config.test_query_limit is not None and config.test_query_limit > 0:
            if len(cognitive_features_input_lines) > config.test_query_limit:
                logger.info(f"由于 test_query_limit={config.test_query_limit}，将认知特征输入从 {len(cognitive_features_input_lines)} 条截断为前 {config.test_query_limit} 条。")
                cognitive_features_input_lines = cognitive_features_input_lines[:config.test_query_limit]
            else:
                logger.info(f"test_query_limit={config.test_query_limit} 大于或等于可用认知特征数 ({len(cognitive_features_input_lines)})。将处理所有可用的认知特征。")
        
        total_lines_to_process = len(cognitive_features_input_lines)

        for line in tqdm(cognitive_features_input_lines, total=total_lines_to_process, desc=f"生成 L{config.personalized_text_target_length}{config.narrative_query_mode_suffix} 叙述 (limit {total_lines_to_process})"):
            try:
                cognitive_data = json.loads(line)
                query_id = cognitive_data.get("query_id")
                original_query_text = cognitive_data.get("query")
                memory_features_for_generator = {"tagged_memory_features": cognitive_data.get("tagged_memory_features", [])}

                if not query_id or not original_query_text:
                    logger.warning(f"由于数据缺失跳过条目: {cognitive_data.get('query_id', 'N/A')}")
                    continue

                narrative_text = narrative_generator.generate_personalized_text(
                    query=original_query_text,
                    memory_results=memory_features_for_generator
                )
                
                output_entry = {
                    "query_id": query_id, "query": original_query_text,
                    "personalized_features": narrative_text,
                    "tagged_memory_features": cognitive_data.get("tagged_memory_features", []),
                    "continuity": cognitive_data.get("continuity", False)
                }
                generated_narratives_data.append(output_entry)
                queries_processed_count +=1

                if queries_processed_count % 50 == 0: # 定期日志
                    logger.info(f"已为 {queries_processed_count}/{total_lines_to_process} 个查询生成叙述 (L{config.personalized_text_target_length}{config.narrative_query_mode_suffix})...")
                        
            except json.JSONDecodeError: 
                logger.warning(f"跳过无效的JSON行: {line.strip()}")
            except Exception as e_inner: 
                logger.error(f"为查询 {cognitive_data.get('query_id', 'N/A')} 处理叙述时出错: {e_inner}", exc_info=True)
        
        output_narrative_file = config.personalized_queries_path 
        os.makedirs(os.path.dirname(output_narrative_file), exist_ok=True)
        with open(output_narrative_file, 'w', encoding='utf-8') as f_out:
            for entry in generated_narratives_data:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"已将 {len(generated_narratives_data)} 条个性化叙述保存到 {output_narrative_file}")
        stage_success = True
    except Exception as e:
        logger.error(f"在叙述生成阶段发生错误: {e}", exc_info=True)
    finally:
        del narrative_generator # 清理
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    logger.info(f"--- 阶段2: 个性化叙述生成 {'完成' if stage_success else '失败'} ---")
    return stage_success

def main():
    args = parse_args()
    config = get_config()
    config.update(args) # 使用所有解析的参数更新config

    start_time = time.time()
    logger.info(f"--- PersLitRank 运行 ---")
    logger.info(f"模式: {args.mode}, 数据集: {config.dataset_name} (类型: {config.dataset_type})")
    logger.info(f"当前进程 GPU: {config.device}, LLM GPU: {config.llm_device}") 
    if config.test_query_limit is not None:
        logger.info(f"*** 测试模式已激活: 后续相关阶段将限制处理前 {config.test_query_limit} 个查询。 ***")
    
    # 根据模式记录相关路径
    if args.mode == "extract_cognitive_features" or args.mode == "all":
        logger.info(f"认知特征 (阶段1) 输出: {config.cognitive_features_detailed_path}")
    
    if args.mode == "generate_narratives" or args.mode == "all":
        logger.info(f"本次运行的叙述目标长度: {config.personalized_text_target_length} -> 长度后缀: '{config.length_suffix}'")
        logger.info(f"叙述生成提示中是否包含查询: {config.include_query_in_narrative_prompt} -> 查询模式后缀: '{config.narrative_query_mode_suffix}'")
        logger.info(f"个性化叙述 (阶段2) 输出 (L{config.personalized_text_target_length}{config.narrative_query_mode_suffix}): {config.personalized_queries_path}")
    
    if args.mode == "retrieve" or args.mode == "all":
        logger.info(f"检索到的文档输出: {config.retrieved_results_path}")

    if args.mode == "rerank" or args.mode == "all":
        logger.info(f"重排器: {config.reranker_type}, 是否使用个性化叙述: {config.use_personalized_features}")
        logger.info(f"用于重排的叙述文件后缀: {config.narrative_query_mode_suffix_for_rerank_input}")
        # rerank_input_narrative_path 的构建需要与 config.personalized_queries_path 的逻辑一致，
        # 只是 narrative_query_mode_suffix 不同。
        rerank_input_narrative_path = os.path.join(
            config.results_dir, 
            config.dataset_name, 
            f"{config._personalized_queries_base}{config.length_suffix}{config.narrative_query_mode_suffix_for_rerank_input}.jsonl"
        )
        logger.info(f"个性化叙述 (重排输入 L{config.personalized_text_target_length}, 模式: {config.narrative_query_mode_suffix_for_rerank_input}): {rerank_input_narrative_path}")
        logger.info(f"重排输出 (L{config.personalized_text_target_length}, TopK: {config.final_top_k}): {config.final_results_path}")


    # --- 模式执行 ---
    if args.mode == "extract_cognitive_features":
        run_cognitive_feature_extraction_stage(config)

    elif args.mode == "generate_narratives":
        # config.test_query_limit 将在此函数内部应用
        run_narrative_generation_stage(config)

    elif args.mode == "retrieve":
        logger.info("--- 执行: 初始文档检索 ---")
        try:
            from feature_retrieval import main as retrieval_main
            retrieval_main() # retrieval_main 内部应使用 config
            logger.info("--- 初始文档检索完成 ---")
        except ImportError: 
            logger.error("无法导入 feature_retrieval。", exc_info=True)
        except Exception as e: 
            logger.error(f"检索期间出错: {e}", exc_info=True)

    elif args.mode == "rerank":
        logger.info("--- 执行: 文档重排 ---")
        # rerank.py 中的 load_retrieved_results 和 load_personalized_features
        # 需要能够处理可能由 test_query_limit 限制的输入文件。
        # 如果叙述文件只包含前N个查询，rerank.py 应该只处理这些查询。
        # config.final_top_k 将决定每个查询保留多少文档。
        
        retrieved_exists = os.path.exists(config.retrieved_results_path)
        # 构建实际用于重排的叙述文件路径
        path_to_narrative_for_rerank = os.path.join(
            config.results_dir, 
            config.dataset_name, 
            f"{config._personalized_queries_base}{config.length_suffix}{config.narrative_query_mode_suffix_for_rerank_input}.jsonl"
        )
        narratives_exist = os.path.exists(path_to_narrative_for_rerank)

        if not retrieved_exists:
             logger.error(f"重排输入缺失: {config.retrieved_results_path}。")
        elif config.use_personalized_features and not narratives_exist:
             logger.error(f"用于重排的叙述文件缺失: {path_to_narrative_for_rerank}。请使用目标长度 {config.personalized_text_target_length} 和查询模式后缀 '{config.narrative_query_mode_suffix_for_rerank_input}' 运行 'generate_narratives' (并考虑 --test_query_limit)。")
        
        if retrieved_exists and (not config.use_personalized_features or narratives_exist):
            try:
                from rerank import main as rerank_main # rerank.py 包含 Reranker 类
                # rerank_main 应该使用 config 来获取所有设置, 包括 test_query_limit 和 final_top_k
                rerank_main() 
                logger.info("--- 文档重排完成 ---")
            except ImportError: 
                logger.error("无法导入 rerank。", exc_info=True)
            except Exception as e: 
                logger.error(f"重排期间出错: {e}", exc_info=True)
        else:
            logger.warning("由于输入缺失，跳过重排。")
            
    elif args.mode == "all":
        logger.info("--- 执行所有阶段 ---")
        s1_ok = run_cognitive_feature_extraction_stage(config)
        
        s_retrieve_ok = False
        if s1_ok:
            logger.info("--- ALL 模式: 运行初始文档检索 ---")
            try:
                from feature_retrieval import main as retrieval_main
                retrieval_main()
                logger.info("--- ALL 模式: 初始文档检索完成 ---")
                s_retrieve_ok = True
            except Exception as e: 
                logger.error(f"ALL 模式检索期间出错: {e}", exc_info=True)
        else: 
            logger.error("由于阶段1失败，在ALL模式下跳过检索。")

        s2_ok = False
        if s1_ok:
            # config.test_query_limit 会在这里应用
            s2_ok = run_narrative_generation_stage(config) 
        else: 
            logger.error("由于阶段1失败，在ALL模式下跳过叙述生成。")
        
        if s_retrieve_ok and (not config.use_personalized_features or s2_ok) :
            logger.info("--- ALL 模式: 运行文档重排 ---")
            try:
                from rerank import main as rerank_main
                # rerank.py 需要能处理可能受限的输入，并使用 config.final_top_k
                rerank_main() 
                logger.info("--- ALL 模式: 文档重排完成 ---")
            except Exception as e: 
                logger.error(f"ALL 模式重排期间出错: {e}", exc_info=True)
        else: 
            logger.warning("由于输入缺失或先前阶段失败，在ALL模式下跳过重排。")

    end_time = time.time()
    logger.info(f"--- PersLitRank 处理模式: {args.mode} 已完成 ---")
    logger.info(f"总执行时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
