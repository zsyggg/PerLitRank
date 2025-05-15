# personalized_generator.py
import logging
import torch
import re
import gc
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from utils import logger, get_config 
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('PersonalizedGenerator_Fallback')
    logger.warning("无法从 utils 导入 logger/get_config，使用回退机制。")
    class DummyConfig: 
        device="cpu"; llm_device="cpu"; local_model_path=None;
        personalized_text_target_length=200; personalized_text_min_length=140; personalized_text_max_length=260;
        local_model_temperature=0.6; local_model_top_p=0.95; local_model_top_k=20;
        local_model_presence_penalty=None; local_model_repetition_penalty=1.0;
        enable_thinking=False; local_model_dtype="float16"; local_model_max_tokens=300;
        include_query_in_narrative_prompt = True 
        def _update_text_length_constraints(self): pass
        def __getattr__(self, name): return None 
    def get_config(): return DummyConfig()


try:
    from prompt_templates import DynamicPromptTemplates 
except ImportError:
     logger.error("无法从 prompt_templates.py 导入 DynamicPromptTemplates。")
     class DynamicPromptTemplates: 
         @staticmethod
         def format_memory_features(features): return "\n".join(features) if features else "没有相关的内存特征可用。"


class PersonalizedGenerator:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.device = getattr(self.config, 'llm_device', getattr(self.config, 'device', 'cpu'))
        logger.info(f"PersonalizedGenerator 初始化，LLM 设备: {self.device}")

        self.local_model_path = getattr(self.config, 'local_model_path', None)
        self.local_model_tokenizer_path = getattr(self.config, 'local_model_tokenizer', self.local_model_path)
        self.local_model_dtype = getattr(self.config, 'local_model_dtype', "float16")
        self.local_model_max_tokens = getattr(self.config, 'local_model_max_tokens', 300) 

        self.target_length = getattr(self.config, 'personalized_text_target_length', 200)
        self.min_length = getattr(self.config, 'personalized_text_min_length', int(self.target_length * 0.7))
        self.max_length = getattr(self.config, 'personalized_text_max_length', int(self.target_length * 1.3))
        logger.info(f"生成器使用文本长度约束: 目标={self.target_length}, 最小={self.min_length}, 最大={self.max_length}")

        self.temperature = getattr(self.config, 'local_model_temperature', 0.6)
        self.top_p = getattr(self.config, 'local_model_top_p', 0.95)
        self.top_k = getattr(self.config, 'local_model_top_k', 20)
        self.presence_penalty = getattr(self.config, 'local_model_presence_penalty', None)
        self.repetition_penalty = getattr(self.config, 'local_model_repetition_penalty', 1.0)
        self.enable_thinking = getattr(self.config, 'enable_thinking', False)
        self.include_query_in_prompt = getattr(self.config, 'include_query_in_narrative_prompt', True)
        logger.info(f"PersonalizedGenerator: 在提示中包含查询设置为: {self.include_query_in_prompt}")
        logger.info(f"生成器LLM参数: temp={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, presence_penalty={self.presence_penalty}, repetition_penalty={self.repetition_penalty}, enable_thinking={self.enable_thinking}")

        self.model = None
        self.tokenizer = None
        if self.local_model_path:
            self.setup_model()
        else:
            logger.error("config中没有local_model_path。PersonalizedGenerator将无法工作。")

    def setup_model(self):
        if not self.local_model_path:
            logger.error("无法设置模型: local_model_path未设置。")
            return
        try:
            logger.info(f"正在加载LLM: {self.local_model_path} 到设备 {self.device}")
            if torch.cuda.is_available() and 'cuda' in str(self.device):
                logger.info(f"正在为设备清除缓存: {self.device} 在模型加载前...")
                try:
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                except Exception: 
                    torch.cuda.empty_cache()
                gc.collect()

            tokenizer_path = self.local_model_tokenizer_path or self.local_model_path
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side="left")
            if self.tokenizer.pad_token is None: 
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.warning("Tokenizer pad_token已设置为eos_token。")

            dtype = torch.float16 if (self.local_model_dtype == "float16" and torch.cuda.is_available() and 'cuda' in str(self.device)) else torch.float32
            model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
            
            self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, **model_kwargs)

            if 'cuda' in str(self.device) and torch.cuda.is_available():
                try:
                    self.model.to(self.device)
                    logger.info(f"模型已移至 {self.device}。")
                except Exception as e:
                    logger.error(f"无法将模型移至 {self.device}: {e}", exc_info=True)
                    self.device = 'cpu' 
                    self.model.to(self.device)
                    logger.warning(f"LLM模型回退到CPU。原始设备是 {self.device}")

            self.model.eval() 
            logger.info(f"LLM设置完成。模型在 {self.model.device}上，评估模式。")
            if torch.cuda.is_available() and 'cuda' in str(self.model.device): 
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"从 {self.local_model_path} 加载LLM失败: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None

    def generate_personalized_text(self, query: str, memory_results: Dict, excluded_memory: Optional[str] = None):
        if self.model is None or self.tokenizer is None:
            return f"错误: 生成器未就绪 (模型/分词器缺失)。"

        try:
            tagged_features_list = memory_results.get("tagged_memory_features", [])
            logger.debug(f"正在为查询 '{query[:50]}...' 生成个性化短语，使用 {len(tagged_features_list)} 个标记特征。目标总长度: {self.target_length}, enable_thinking: {self.enable_thinking}")

            formatted_features_for_prompt = DynamicPromptTemplates.format_memory_features(tagged_features_list)

            prompt_parts = ["Context:"]
            if self.include_query_in_prompt:
                prompt_parts.append(f"User Query: {query}")
            else:
                logger.debug(f"PersonalizedGenerator: 根据配置，从提示中排除了 User Query。")
            
            prompt_parts.append(f"Memory Features (derived from user's interaction history):\n{formatted_features_for_prompt}")
            prompt_parts.append(f"""
Instruction:
Generate 3 to 5 concise, descriptive phrases or keywords in English based on the provided User Query (if any) and Memory Features. These phrases should summarize the user's current research focus, specific interests, key themes, or relevant entities.
- Each phrase MUST be on a new line.
- The total character length of all generated phrases combined MUST be approximately {self.target_length} characters (strictly between {self.min_length} and {self.max_length} characters).
- Output *ONLY* the phrases. Do NOT include any explanations, labels, preambles, or markdown formatting like bullet points.
- After generating the 3 to 5 phrases, stop. Do not add any further text.

Example of desired output (content will vary based on input):
Graph neural networks for material science
Predictive accuracy and reliability
Computational chemistry integration
Domain-specific knowledge application
""") # 在指令中增加了明确的停止要求
            prompt_content = "\n".join(prompt_parts)

            is_qwen_model = self.local_model_path and "qwen" in self.local_model_path.lower()
            messages = [{"role": "user", "content": prompt_content}]

            apply_chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if is_qwen_model:
                 apply_chat_kwargs["enable_thinking"] = self.enable_thinking

            final_prompt_text = self.tokenizer.apply_chat_template(messages, **apply_chat_kwargs)
            
            max_model_len = getattr(self.tokenizer, 'model_max_length', 4096)
            effective_max_prompt_len = max_model_len - self.local_model_max_tokens - 10 
            if effective_max_prompt_len <=0:
                logger.warning(f"计算出的 effective_max_prompt_len ({effective_max_prompt_len}) 太小。使用512。检查 local_model_max_tokens 和 model_max_length。")
                effective_max_prompt_len = 512

            inputs = self.tokenizer(final_prompt_text, return_tensors="pt", padding=False, truncation=True, max_length=effective_max_prompt_len).to(self.model.device)
            input_length = inputs.input_ids.shape[1]

            avg_chars_per_token = 3.5 
            estimated_min_tokens_for_min_length = int(self.min_length / avg_chars_per_token)
            min_new_tokens_val = max(10, estimated_min_tokens_for_min_length) 

            generation_config_dict = {
                "max_new_tokens": self.local_model_max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": True, 
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id or self.tokenizer.convert_tokens_to_ids("<|endoftext|>") or self.tokenizer.eos_token_id, # 确保有eos_token_id
                "repetition_penalty": self.repetition_penalty,
                "min_new_tokens": min_new_tokens_val,
            }
            if self.presence_penalty is not None:
                generation_config_dict["presence_penalty"] = self.presence_penalty

            logger.debug(f"生成配置: {generation_config_dict}")

            with torch.no_grad():
                outputs = self.model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, **generation_config_dict)

            output_ids = outputs[0][input_length:].tolist()
            # 先解码一次，不跳过特殊token，以便检查</think>等
            generated_text_with_special_tokens = self.tokenizer.decode(output_ids, skip_special_tokens=False).strip()
            logger.debug(f"LLM原始输出 (包含特殊token):\n'{generated_text_with_special_tokens}'")

            # 移除 <think>...</think> 块 (如果存在)
            generated_text = generated_text_with_special_tokens
            if is_qwen_model and "<think>" in generated_text: # 仅当是Qwen模型且包含<think>时处理
                think_pattern = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
                generated_text_no_think = think_pattern.sub("", generated_text).strip()
                if len(generated_text_no_think) < len(generated_text):
                    logger.info("从Qwen输出中移除了<think>块。")
                generated_text = generated_text_no_think
            
            # 在移除了<think>之后，再用skip_special_tokens=True解码一次，以去除其他如<|endoftext|>等
            generated_text = self.tokenizer.decode(self.tokenizer.encode(generated_text, add_special_tokens=False), skip_special_tokens=True).strip()
            logger.debug(f"解码后文本 (移除<think>和特殊token后):\n'{generated_text}'")


            # --- 增强的后处理逻辑 ---
            # 1. 寻找停止模式并截断
            stop_patterns = [
                r"\nHuman:", r"\nUser:", r"\nAssistant:", r"\nResearcher:", r"\nOkay, here",
                r"\nHere are the phrases", r"Example of desired output",
                # 对于Qwen等模型，有时会在末尾重复部分输入或生成模板化的回答开头
                r"You are a research assistant", r"You are a helpful assistant",
                r"Please answer the", r"Instruction:", r"Context:"
            ]
            earliest_stop_index = len(generated_text)
            detected_stop_pattern = None
            for pattern in stop_patterns:
                # 使用 re.IGNORECASE 进行不区分大小写的匹配
                match = re.search(pattern, generated_text, re.IGNORECASE | re.DOTALL)
                if match:
                    if match.start() < earliest_stop_index:
                        earliest_stop_index = match.start()
                        detected_stop_pattern = pattern
            
            if earliest_stop_index < len(generated_text):
                logger.info(f"由于检测到停止模式 '{detected_stop_pattern}' 在索引 {earliest_stop_index}，截断生成的文本。")
                generated_text = generated_text[:earliest_stop_index].strip()
            
            logger.debug(f"移除停止模式后的文本:\n'{generated_text}'")

            # 2. 移除代码块标记和迭代清理填充词
            generated_text = re.sub(r'^```[\s\S]*?```', '', generated_text, flags=re.MULTILINE).strip()
            common_fillers_patterns = [
                r"^\s*Okay, let me try.*?\n", r"^\s*Okay, let's tackle this\.?",
                r"^\s*Here are some descriptive phrases.*?:", r"^\s*Based on the query and memory features.*?:",
                r"^\s*Personalized Description Phrases:?", r"^\s*Output Phrases:?", r"^\s*Okay,\s*",
                r"^\s*Here are the phrases:", r"^\s*Generated phrases:",
                r"^\s*-\s*", r"^\s*\*\s*", # 移除列表标记
                r"^\s*1\.\s*", r"^\s*2\.\s*", r"^\s*3\.\s*", r"^\s*4\.\s*", r"^\s*5\.\s*" # 移除数字列表标记
            ]
            for _ in range(3): # 多次迭代以处理嵌套情况
                old_text_len = len(generated_text)
                for pattern in common_fillers_patterns:
                    generated_text = re.sub(pattern, '', generated_text, flags=re.IGNORECASE | re.MULTILINE).strip()
                if len(generated_text) == old_text_len:
                    break 
            
            generated_text = generated_text.strip() # 最后再清理一次首尾空白
            logger.debug(f"清理填充词后的文本:\n'{generated_text}'")

            # 3. 按行分割并过滤空短语
            phrases = [p.strip() for p in generated_text.split('\n') if p.strip() and len(p.strip()) > 2]
            generated_text = "\n".join(phrases) 
            logger.debug(f"按行分割并过滤后的短语文本:\n'{generated_text}'")

            # 4. 处理空输出的回退
            current_length = len(generated_text)
            if current_length == 0 and query:
                 logger.warning(f"查询 '{query}' 生成的短语为空。使用回退机制。")
                 fallback_query_part = query[:50] if self.include_query_in_prompt else "general context"
                 fallback_phrases = [f"Focus related to: {fallback_query_part}...", f"Key aspects of: {', '.join(query.split()[:4]) if query.split() else 'N/A'}"]
                 generated_text = "\n".join(fallback_phrases).strip()
                 current_length = len(generated_text)
                 logger.info(f"回退生成的文本 (长度 {current_length}):\n'{generated_text}'")


            # 5. 长度控制 (截断)
            if current_length > self.max_length:
                logger.debug(f"生成的短语长度 ({current_length}) > 最大长度 ({self.max_length})。正在截断...")
                truncated_phrases_list = []
                accumulated_length = 0
                # 重新从清理后的 phrases 列表开始截断
                phrases_for_truncation = [p.strip() for p in generated_text.split('\n') if p.strip()]

                for p_idx, p_content in enumerate(phrases_for_truncation):
                    # 当前短语长度 + 换行符长度 (如果不是最后一个短语)
                    len_p_content = len(p_content)
                    len_newline = 1 if p_idx < len(phrases_for_truncation) - 1 else 0
                    
                    if accumulated_length + len_p_content + len_newline <= self.max_length:
                        truncated_phrases_list.append(p_content)
                        accumulated_length += (len_p_content + len_newline)
                    else: # 如果添加当前短语会超出长度
                        remaining_char_budget = self.max_length - accumulated_length
                        if remaining_char_budget > 5: # 只有当剩余空间足够有意义时才尝试添加部分短语
                            partial_phrase_candidate = p_content[:remaining_char_budget]
                            # 尝试在词边界截断
                            last_space_idx = partial_phrase_candidate.rfind(' ')
                            # 如果能在词边界截断，并且不会损失太多（例如，只差几个字符就到词尾），则在词边界截断
                            if last_space_idx > 0 and (remaining_char_budget - last_space_idx) < 15 : 
                                final_partial_phrase = partial_phrase_candidate[:last_space_idx].strip()
                                if final_partial_phrase: # 确保截断后不是空字符串
                                     truncated_phrases_list.append(final_partial_phrase + "...")
                            else: # 否则，硬截断
                                final_partial_phrase = partial_phrase_candidate[:max(0, remaining_char_budget - 3)].strip()
                                if final_partial_phrase:
                                    truncated_phrases_list.append(final_partial_phrase + "...")
                        break # 停止添加短语
                generated_text = "\n".join(truncated_phrases_list).strip()
                logger.info(f"截断后的文本 (长度 {len(generated_text)}):\n'{generated_text}'")

            elif current_length < self.min_length and current_length > 0 :
                 logger.warning(f"生成的短语 ({current_length}) 比最小长度 ({self.min_length}) 短:\n'{generated_text}'")
            
            logger.info(f"最终个性化短语 (目标总长度: {self.target_length}, 实际: {len(generated_text)}):\n'{generated_text}'")
            return generated_text

        except Exception as e:
            logger.error(f"为查询 '{query}' 生成个性化文本（短语）时出错: {e}", exc_info=True)
            fallback_query_part = query[:40] if self.include_query_in_prompt else "context"
            return f"Focus: Related to query '{fallback_query_part}...'\nError: Could not generate personalized phrases."

    def generate_personalized_text_batch(self, queries: List[str], memory_results_list: List[Dict], excluded_memory_list: Optional[List[str]] = None):
        if self.model is None or self.tokenizer is None:
            return ["错误: 生成器未就绪。"] * len(queries)

        descriptions = []
        for i, query_text in enumerate(queries):
            mem_res = memory_results_list[i] if i < len(memory_results_list) else {}
            excl_mem = excluded_memory_list[i] if excluded_memory_list and i < len(excluded_memory_list) else None
            desc = self.generate_personalized_text(query_text, mem_res, excl_mem)
            descriptions.append(desc)
            if (i + 1) % 20 == 0: 
                logger.info(f"批量生成了 {i+1}/{len(queries)} 组个性化短语。")
        return descriptions
