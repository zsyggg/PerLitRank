# utils.py
import os
import logging
import torch
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('perslitrank.log') # 所有日志将输出到此处
    ]
)
logger = logging.getLogger('PersLitRank')

@dataclass
class Document:
    text_id: str
    title: str = ""
    text: str = ""
    full_paper: Optional[str] = None
    full_text: Optional[str] = None # 组合字段，方便处理
    score: float = 0.0

@dataclass
class Query:
    query_id: str
    query: str
    personalized_features: str = ""
    tagged_memory_features: List[str] = field(default_factory=list) # 用于LLM输入
    # 内存组件的原始输出，用于详细存储
    sequential_results_raw: Optional[Dict] = field(default_factory=dict)
    working_memory_state_raw: Optional[Dict] = field(default_factory=dict)
    long_term_memory_results_raw: Optional[Dict] = field(default_factory=dict)
    # 连续性相关字段，在认知特征提取期间填充
    continuity_score: float = 0.0
    high_continuity_terms: List[str] = field(default_factory=list)
    continuity: bool = False # 查询的整体连续性标志
    topic_id: str = ""
    turn_id: int = 0

    def __post_init__(self):
        self.query_id = str(self.query_id)
        if "_" in self.query_id:
            parts = self.query_id.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                self.topic_id = "_".join(parts[:-1])
                try:
                    self.turn_id = int(parts[-1])
                except ValueError:
                     logger.warning(f"无法从 {self.query_id} 解析 turn_id。默认为 0。")
                     self.turn_id = 0
            else:
                 self.topic_id = self.query_id # 如果没有下划线或者最后一个部分不是数字，则整个query_id作为topic_id
                 self.turn_id = 0 # turn_id默认为0
        else: # 如果query_id中没有下划线
            self.topic_id = self.query_id
            self.turn_id = 0


class Config:
    def __init__(self):
        # 核心路径和标识符
        self.dataset_name = "MedCorpus"
        self.base_data_dir = "/workspace/PerMed/data"
        self.results_dir = "./results"

        # 设备配置
        self.gpu_id = 0
        self.device = None
        self.llm_device = None
        self._setup_device()

        # 内存系统和特征提取
        self.memory_components = ["sequential", "working", "long"]
        self.feature_extractor = "keybert"
        self.keybert_embedder_device = "cpu" # 默认 KeyBERT 使用 CPU
        if self.device and "cuda" in self.device: # 如果主设备是 GPU，KeyBERT 也用 GPU
            self.keybert_embedder_device = self.device
        
        self.keybert_model = "/workspace/.cache/modelscope/hub/iic/gte_sentence-embedding_multilingual-base"
        self.memory_type = "vector"
        self.sentence_transformer_model = "/workspace/.cache/modelscope/hub/iic/gte_sentence-embedding_multilingual-base"
        self.dataset_type = self._infer_dataset_type()
        self.embedding_path = "/workspace/.cache/modelscope/hub/iic/gte_sentence-embedding_multilingual-base"
        self.batch_size = 256
        self.initial_top_k = 50 # 初始检索召回数
        self.final_top_k = 10   # 重排后默认保留数

        # 个性化叙述相关
        self.personalized_text_target_length = 200
        self.length_suffix = f"_L{self.personalized_text_target_length}"
        self._update_text_length_constraints()
        self.include_query_in_narrative_prompt = True
        self.narrative_query_mode_suffix = "_wq" if self.include_query_in_narrative_prompt else "_nq"
        self.narrative_query_mode_suffix_for_rerank_input = "_wq" # 重排时默认加载带query的叙述

        # LLM 提示中的标记特征限制
        self.max_tagged_features_for_llm = 7
        self.max_features_per_memory_module = 2
        self.max_phrases_per_tag = 3

        # 文件名基础部分
        self._cognitive_features_detailed_base = "cognitive_features_detailed"
        self._personalized_queries_base = "personalized_queries"
        self._final_results_base = "ranked"
        self._initialize_data_paths() # 初始化路径

        # 重排器配置
        self.reranker_path = "/workspace/.cache/modelscope/hub/iic/gte_passage-ranking_multilingual-base"
        self.jina_reranker_path = "/workspace/.cache/modelscope/hub/jinaai/jina-reranker-v2-base-multilingual"
        self.minicpm_reranker_path = "/workspace/.cache/modelscope/hub/OpenBMB/MiniCPM-Reranker-Light"
        self.reranker_type = "gte"
        self.use_personalized_features = True
        self.reranker_max_length = 512

        # 本地 LLM 配置
        self.local_model_path = "/workspace/Qwen3-4B"
        self.local_model_tokenizer = None
        self.local_model_dtype = "float16"
        self.local_model_max_tokens = 300
        self.local_model_temperature = 0.6
        self.local_model_top_p = 0.95
        self.local_model_top_k = 20
        self.local_model_presence_penalty = None
        self.local_model_repetition_penalty = 1.0
        self.enable_thinking = False
        self.is_conversational = False
        self.continuity_threshold = 0.3
        
        # 新增：用于限制测试时处理的查询数量
        self.test_query_limit: Optional[int] = None

    def _initialize_data_paths(self):
        self.queries_path = self._get_data_path("queries.jsonl")
        self.corpus_path = self._get_data_path("corpus.jsonl")
        self.corpus_embeddings_path = self._get_data_path("corpus_embeddings.npy")
        self.retrieved_results_path = self._get_results_path_nosuffix("retrieved.jsonl")
        self.cognitive_features_detailed_path = self._get_results_path_nosuffix(f"{self._cognitive_features_detailed_base}.jsonl")

    def _update_text_length_constraints(self):
        self.personalized_text_min_length = int(self.personalized_text_target_length * 0.7)
        self.personalized_text_max_length = int(self.personalized_text_target_length * 1.3)
        logger.debug(f"个性化文本长度约束: 目标={self.personalized_text_target_length}, 最小/最大={self.personalized_text_min_length}/{self.personalized_text_max_length}, 后缀='{self.length_suffix}'")

    @property
    def personalized_queries_path(self):
        query_mode_suffix = getattr(self, 'narrative_query_mode_suffix', '_wq')
        # 如果设置了测试查询限制，并且文件名需要体现这一点（可选，当前不添加以保持简洁）
        # test_limit_suffix = f"_limit{self.test_query_limit}" if self.test_query_limit is not None else ""
        # return self._get_results_path_with_suffix(f"{self._personalized_queries_base}{self.length_suffix}{query_mode_suffix}{test_limit_suffix}.jsonl")
        return self._get_results_path_with_suffix(f"{self._personalized_queries_base}{self.length_suffix}{query_mode_suffix}.jsonl")


    @property
    def final_results_path(self):
        mode_suffix = "personalized" if self.use_personalized_features else "baseline"
        if self.use_personalized_features:
            query_mode_suffix_for_rerank_input = getattr(self, 'narrative_query_mode_suffix_for_rerank_input', '_wq')
            mode_suffix += query_mode_suffix_for_rerank_input
        
        type_suffix = f"_{self.reranker_type}" if self.reranker_type else ""
        
        # 文件名中总是包含 final_top_k 的值，除非它是默认值10且没有test_query_limit
        k_suffix = f"_top{self.final_top_k}"
        if self.final_top_k == 10 and self.test_query_limit is None: # 默认完整运行时，不加 _top10
            k_suffix = ""
        # 如果是测试（例如 final_top_k=100），则文件名会是 ..._top100.jsonl

        base_filename = f"{self._final_results_base}{type_suffix}_{mode_suffix}"
        return self._get_results_path_with_suffix(f"{base_filename}{self.length_suffix}{k_suffix}.jsonl")

    def _setup_device(self):
        if not torch.cuda.is_available():
            self.device = "cpu"
            self.llm_device = "cpu"
            logger.warning("CUDA 不可用，所有操作将使用 CPU。")
            return
        num_gpus = torch.cuda.device_count()
        if self.gpu_id >= num_gpus:
            logger.warning(f"提供的 GPU ID {self.gpu_id} 对于 {num_gpus} 个可用 GPU 无效。默认为 GPU 0。")
            self.gpu_id = 0
        self.device = f"cuda:{self.gpu_id}"
        self.llm_device = self.device 
        try:
            gpu_name = torch.cuda.get_device_name(self.gpu_id)
            logger.info(f"GPU 设置: 主设备设置为 {self.device} ('{gpu_name}')。LLM 将使用 {self.llm_device}。")
        except Exception as e:
            logger.error(f"无法获取 GPU ID {self.gpu_id} 的名称。错误: {e}")
            logger.info(f"GPU 设置: 主设备设置为 {self.device}。LLM 将使用 {self.llm_device}。")

    def _infer_dataset_type(self):
        name_lower = self.dataset_name.lower()
        if "coral" in name_lower: return "coral"
        elif "medcorpus" in name_lower: return "medcorpus"
        elif "litsearch" in name_lower: return "litsearch"
        return "unknown"

    def _get_data_path(self, filename: str) -> str:
        return os.path.join(self.base_data_dir, self.dataset_name, filename)

    def _get_results_path_with_suffix(self, filename_with_all_suffixes_ext: str) -> str:
        dataset_results_dir = os.path.join(self.results_dir, self.dataset_name)
        os.makedirs(dataset_results_dir, exist_ok=True)
        return os.path.join(dataset_results_dir, filename_with_all_suffixes_ext)

    def _get_results_path_nosuffix(self, filename_ext: str) -> str:
        dataset_results_dir = os.path.join(self.results_dir, self.dataset_name)
        os.makedirs(dataset_results_dir, exist_ok=True)
        return os.path.join(dataset_results_dir, filename_ext)

    def update(self, args):
        if hasattr(args, 'dataset_name') and args.dataset_name:
            if self.dataset_name != args.dataset_name:
                self.dataset_name = args.dataset_name
                self.dataset_type = self._infer_dataset_type()
                self._initialize_data_paths()
                logger.info(f"数据集已更改为 '{self.dataset_name}'。路径已重新初始化。")

        if hasattr(args, 'data_dir') and args.data_dir:
            self.base_data_dir = args.data_dir
            self._initialize_data_paths()
        if hasattr(args, 'results_dir') and args.results_dir:
            self.results_dir = args.results_dir

        if hasattr(args, 'gpu_id') and args.gpu_id is not None:
            if self.gpu_id != args.gpu_id or self.device is None:
                self.gpu_id = args.gpu_id
                self._setup_device()
                if self.device and "cuda" in self.device:
                    self.keybert_embedder_device = self.device
                else:
                    self.keybert_embedder_device = "cpu"
                logger.info(f"GPU ID 已更改。主设备: {self.device}, LLM 设备: {self.llm_device}, KeyBERT embedder: {self.keybert_embedder_device}")

        if hasattr(args, 'llm_gpu_id') and args.llm_gpu_id is not None and torch.cuda.is_available():
            if args.llm_gpu_id >= torch.cuda.device_count():
                logger.warning(f"LLM GPU ID {args.llm_gpu_id} 无效。LLM 将使用 {self.llm_device}。")
            else:
                self.llm_device = f"cuda:{args.llm_gpu_id}"
                logger.info(f"LLM 设备已明确设置为: {self.llm_device}")

        if hasattr(args, 'personalized_text_target_length') and args.personalized_text_target_length is not None:
            if self.personalized_text_target_length != args.personalized_text_target_length:
                self.personalized_text_target_length = args.personalized_text_target_length
                self.length_suffix = f"_L{self.personalized_text_target_length}"
                self._update_text_length_constraints()
                logger.info(f"个性化文本目标长度已设置为 {self.personalized_text_target_length}。")

        if hasattr(args, 'include_query_in_narrative_prompt'):
            self.include_query_in_narrative_prompt = args.include_query_in_narrative_prompt
            self.narrative_query_mode_suffix = "_wq" if self.include_query_in_narrative_prompt else "_nq"
            logger.info(f"叙述生成: include_query_in_narrative_prompt 设置为 {self.include_query_in_narrative_prompt}。文件名后缀: {self.narrative_query_mode_suffix}")
        else:
             if not hasattr(self, 'include_query_in_narrative_prompt'):
                 self.include_query_in_narrative_prompt = True
                 self.narrative_query_mode_suffix = "_wq"

        if hasattr(args, 'narrative_query_mode_suffix_for_rerank_input') and args.narrative_query_mode_suffix_for_rerank_input is not None:
            self.narrative_query_mode_suffix_for_rerank_input = args.narrative_query_mode_suffix_for_rerank_input
            logger.info(f"重排输入叙述后缀设置为: {self.narrative_query_mode_suffix_for_rerank_input}")
        else:
            if not hasattr(self, 'narrative_query_mode_suffix_for_rerank_input'):
                 self.narrative_query_mode_suffix_for_rerank_input = "_wq"

        if hasattr(args, 'max_tagged_features_for_llm') and args.max_tagged_features_for_llm is not None:
            self.max_tagged_features_for_llm = args.max_tagged_features_for_llm
        if hasattr(args, 'max_features_per_memory_module') and args.max_features_per_memory_module is not None:
            self.max_features_per_memory_module = args.max_features_per_memory_module
        if hasattr(args, 'max_phrases_per_tag') and args.max_phrases_per_tag is not None:
            self.max_phrases_per_tag = args.max_phrases_per_tag
        
        if hasattr(args, 'final_top_k') and args.final_top_k is not None:
            self.final_top_k = args.final_top_k
            logger.info(f"配置: final_top_k 更新为 {self.final_top_k}。")
            
        # 新增：处理 test_query_limit 参数
        if hasattr(args, 'test_query_limit') and args.test_query_limit is not None:
            try:
                self.test_query_limit = int(args.test_query_limit)
                if self.test_query_limit <= 0: # 确保是正数
                    logger.warning(f"test_query_limit ({self.test_query_limit}) 无效，将被忽略。")
                    self.test_query_limit = None
                else:
                    logger.info(f"配置: test_query_limit 设置为 {self.test_query_limit}。后续阶段将只处理这么多查询。")
            except ValueError:
                logger.warning(f"无法将 test_query_limit '{args.test_query_limit}' 解析为整数。将被忽略。")
                self.test_query_limit = None


        for attr_name in [
            'feature_extractor', 'memory_type', 'keybert_model',
            'sentence_transformer_model', 'embedding_path', 'batch_size',
            'initial_top_k', 
            'reranker_type', 'reranker_path',
            'use_personalized_features', 'reranker_max_length', 'local_model_path',
            'local_model_tokenizer', 'local_model_dtype',
            'enable_thinking',
            'local_model_max_tokens',
            'local_model_temperature', 'local_model_top_p', 'local_model_top_k',
            'local_model_presence_penalty', 'local_model_repetition_penalty',
            'continuity_threshold'
        ]:
            if hasattr(args, attr_name) and getattr(args, attr_name) is not None:
                current_val_from_arg = getattr(args, attr_name)
                if attr_name == 'local_model_presence_penalty' and current_val_from_arg == 0:
                    setattr(self, attr_name, None)
                    logger.info(f"配置: {attr_name} 设置为 None (原为 0)。")
                else:
                    setattr(self, attr_name, current_val_from_arg)
                    logger.debug(f"配置: {attr_name} 更新为 {current_val_from_arg}。")
        
        if hasattr(args, 'enable_thinking') and isinstance(getattr(args, 'enable_thinking'), bool):
             self.enable_thinking = getattr(args, 'enable_thinking')
             logger.info(f"配置: enable_thinking 从参数设置为 {self.enable_thinking}。")

        if hasattr(args, 'memory_components') and args.memory_components:
            valid_components = ["sequential", "working", "long"]
            self.memory_components = [
                c.strip().lower() for c in args.memory_components.split(',')
                if c.strip().lower() in valid_components
            ]
            logger.info(f"内存组件已更新为: {self.memory_components}")

        if hasattr(args, 'exclude_memory') and args.exclude_memory and args.exclude_memory != 'none':
            excluded_mem_lower = args.exclude_memory.lower()
            if excluded_mem_lower in self.memory_components:
                self.memory_components.remove(excluded_mem_lower)
                logger.info(f"内存组件 '{excluded_mem_lower}' 已排除。活动组件: {self.memory_components}")
            else:
                logger.warning(f"试图排除内存组件 '{excluded_mem_lower}'，但它不是活动组件或无效。")

        if self.reranker_type and not (hasattr(args, 'reranker_path') and args.reranker_path):
            if self.reranker_type == "jina": self.reranker_path = self.jina_reranker_path
            elif self.reranker_type == "minicpm": self.reranker_path = self.minicpm_reranker_path
            logger.info(f"为 reranker_type '{self.reranker_type}' 使用默认路径: {self.reranker_path}")

        self.is_conversational = (hasattr(args, 'conversational') and args.conversational) or \
                                 self.dataset_type in ["coral", "medcorpus"]
        if self.is_conversational:
            logger.info(f"运行模式设置为对话式 (dataset_type: {self.dataset_type}, arg: {hasattr(args, 'conversational') and args.conversational})。")

        logger.info(f"配置已更新。最终相关路径:")
        logger.info(f"  查询文件: '{self.queries_path}'")
        logger.info(f"  语料库文件: '{self.corpus_path}'")
        logger.info(f"  认知特征 (阶段1输出): {self.cognitive_features_detailed_path}")
        logger.info(f"  个性化叙述 (阶段2输出, L{self.personalized_text_target_length}, 模式: {self.narrative_query_mode_suffix}): {self.personalized_queries_path}")
        logger.info(f"  初始检索文档 (检索输出): {self.retrieved_results_path}")
        logger.info(f"  最终重排文档 (重排输出, L{self.personalized_text_target_length}, 叙述模式: {self.narrative_query_mode_suffix_for_rerank_input}, TopK: {self.final_top_k}): {self.final_results_path}")
        logger.info(f"  KeyBERT embedder 设备: {self.keybert_embedder_device}")
        logger.info(f"  LLM enable_thinking: {self.enable_thinking}")
        logger.info(f"  LLM include_query_in_prompt: {self.include_query_in_narrative_prompt}")
        if self.test_query_limit is not None:
            logger.info(f"  测试模式: 后续阶段将限制处理前 {self.test_query_limit} 个查询。")


class FeatureExtractorRegistry:
    _extractors = {}
    @classmethod
    def register(cls, name):
        def decorator(extractor_class):
            cls._extractors[name] = extractor_class
            logger.debug(f"特征提取器 '{name}' 已注册。")
            return extractor_class
        return decorator

    @classmethod
    def get_extractor(cls, name, **kwargs):
        if name not in cls._extractors:
            logger.error(f"未知的特征提取器: {name}")
            raise ValueError(f"未知的特征提取器: {name}")
        logger.debug(f"获取特征提取器: {name}")
        return cls._extractors[name](**kwargs)

class MemorySystemRegistry:
    _memory_systems = {}
    @classmethod
    def register(cls, name):
        def decorator(memory_class):
            cls._memory_systems[name] = memory_class
            logger.debug(f"内存系统 '{name}' 已注册。")
            return memory_class
        return decorator

_config = None
def get_config():
    global _config
    if _config is None:
        _config = Config()
        logger.info("全局 Config 对象已创建。")
    return _config

def load_queries(config: Config) -> List[Query]:
    queries = []
    if not os.path.exists(config.queries_path):
        logger.error(f"查询文件未找到: {config.queries_path}")
        return queries
    try:
        with open(config.queries_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    query_obj = Query(query_id=data['query_id'], query=data['query'])
                    if 'continuity' in data:
                        query_obj.continuity = bool(data['continuity'])
                    queries.append(query_obj)
                except KeyError as e:
                    logger.error(f"查询文件 {config.queries_path} 第 {line_num} 行缺少键 {e}")
                except Exception as e:
                    logger.error(f"解析查询文件 {config.queries_path} 第 {line_num} 行时出错: {e}")
    except Exception as e:
        logger.error(f"从 {config.queries_path} 加载查询失败: {e}")
    logger.info(f"从 {config.queries_path} 加载了 {len(queries)} 个原始查询")
    return queries

def load_corpus(config: Config) -> Dict[str, Document]:
    documents = {}
    if not os.path.exists(config.corpus_path):
        logger.error(f"语料库文件未找到: {config.corpus_path}")
        return documents
    try:
        with open(config.corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    text_id_str = str(data['text_id'])
                    title = data.get('title','') or ""
                    text_content = data.get('text','') or ""
                    full_paper_content = data.get('full_paper','') or ""
                    text_parts = [title, text_content, full_paper_content]
                    full_text = " ".join(filter(None, text_parts)).strip()
                    documents[text_id_str] = Document(
                        text_id=text_id_str,
                        title=title,
                        text=text_content,
                        full_paper=full_paper_content if full_paper_content else None,
                        full_text=full_text
                    )
                except KeyError as e:
                    logger.error(f"语料库文件 {config.corpus_path} 第 {line_num} 行缺少键 {e}")
                except Exception as e:
                    logger.error(f"解析语料库文件 {config.corpus_path} 第 {line_num} 行时出错: {e}")
    except Exception as e:
        logger.error(f"从 {config.corpus_path} 加载语料库失败: {e}")
    logger.info(f"从 {config.corpus_path} 加载了 {len(documents)} 个文档")
    return documents
