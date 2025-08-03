# ==============================================================================
# 脚本名称: 通用推理性能与正确率测试脚本
#
# 脚本描述:
# 本脚本旨在测试不同推理后端（本地vLLM或外部API）在K&K逻辑谜题上的
# 推理性能和基本正确率。它从指定的Parquet文件中加载谜题数据，对每个谜题
# 进行一次采样推理，并统计成功解析并正确回答的谜题数量。
#
# 环境依赖:
# # 通用依赖
# pip install torch pandas tqdm datasets pyarrow
# # 如果使用vLLM模式
# pip install vllm
# # 如果使用API模式
# pip install openai "tqdm>=4.65.0" # tqdm需要较新版本以支持asyncio
#
# 使用方法:
# 1. 确保所有依赖库已安装。
# 2. 修改下面的 CONFIGURATION (配置) 部分。
# 3. 设置 INFERENCE_MODE 为 'VLLM' 或 'API'。
# 4. 根据选择的模式，填写对应的 VLLM_CONFIG 或 API_CONFIG。
# 5. (推荐)为API模式设置环境变量 'SILICONFLOW_API_KEY'。
# 6. 在终端中运行: python your_test_script_name.py
# ==============================================================================

import json
import os
import re
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm as asyncio_tqdm # 用于异步操作的tqdm
from tqdm import tqdm
from datasets import Dataset
import logging
import time
import asyncio

# --- CONFIGURATION (配置) ---

# 模式选择: 'VLLM' 或 'API'
INFERENCE_MODE = 'API' # <-- 在这里切换 'VLLM' 或 'API'

# vLLM 配置 (当 INFERENCE_MODE = 'VLLM' 时使用)
VLLM_CONFIG = {
    "MODEL_ID": "/root/autodl-tmp/myverl/ckpts/DAPO/DAPO-Qwen3-0.6b-think-free2/global_step_12/actor_inference/",
    "TENSOR_PARALLEL_SIZE": 1,
    "LOAD_FORMAT": "pt"
}

# API 配置 (当 INFERENCE_MODE = 'API' 时使用)
API_CONFIG = {
    "API_KEY": os.getenv("SILICONFLOW_API_KEY", "sk-lqzxjcmncefsrrejwbupgatcnpbsjhejuqfqneviobtjkefg"),
    "BASE_URL": "https://api.siliconflow.cn/v1",
    "MODEL_NAME": "Pro/deepseek-ai/DeepSeek-R1-0120",
    # "MODEL_NAME": "deepseek-ai/DeepSeek-V2-Chat",
    
    # !!! 新增：API并发请求数限制 !!!
    # 用于控制同时向API发送的请求数量，防止因速率过快而被服务器拒绝。
    # 建议从一个较低的数值开始（如5-10），根据API提供商的限制进行调整。
    "CONCURRENCY_LIMIT": 10,
}

# 通用文件和采样配置
BASE_PATH = "/root/autodl-tmp/myverl/data"
INPUT_PARQUET_PATHS = [
    os.path.join(BASE_PATH, "kk/4ppl", "test.parquet"),
]
K_SAMPLES = 1
MAX_NEW_TOKENS = 4096

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 推理后端抽象 ---

class VLLMRunner:
    """使用vLLM在本地进行推理"""
    def __init__(self, config):
        try:
            from vllm import LLM
        except ImportError:
            logging.error("错误: vLLM 库未安装。请运行 'pip install vllm'")
            exit()
            
        logging.info(f"使用vLLM加载模型 '{config['MODEL_ID']}'...")
        start_load_time = time.time()
        
        self.llm = LLM(
            model=config['MODEL_ID'],
            tensor_parallel_size=config.get('TENSOR_PARALLEL_SIZE', 1),
            trust_remote_code=True,
            load_format=config.get('LOAD_FORMAT', 'auto')
        )
        self.tokenizer = self.llm.get_tokenizer()
        end_load_time = time.time()
        logging.info(f"模型加载完成，耗时: {end_load_time - start_load_time:.2f} 秒。")

    def prepare_prompts(self, dataset, system_prompt):
        """为vLLM准备格式化的完整prompt字符串"""
        all_prompts = []
        for item in dataset:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['quiz']}
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt_text)
        return all_prompts

    def generate(self, prompts, sampling_params):
        """执行vLLM批量生成"""
        from vllm import SamplingParams
        
        vllm_params = SamplingParams(
            n=sampling_params.get('n', 1),
            temperature=sampling_params.get('temperature', 0.7),
            top_p=sampling_params.get('top_p', 0.9),
            max_tokens=sampling_params.get('max_tokens', 8192)
        )
        
        # vLLM 内部自动处理批量化，效率很高
        outputs = self.llm.generate(prompts, vllm_params)
        return [output.outputs[0].text for output in outputs]

class APIRunner:
    """使用外部API进行推理"""
    def __init__(self, config):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logging.error("错误: OpenAI 库未安装。请运行 'pip install openai'")
            exit()

        logging.info(f"配置API推理后端，目标模型: '{config['MODEL_NAME']}'")
        if not config.get("API_KEY") or config.get("API_KEY") == "your_api_key_here":
            logging.error("API 密钥未配置。请在 API_CONFIG 中设置 API_KEY 或设置环境变量。")
            exit()
        
        self.config = config
        self.client = AsyncOpenAI(
            api_key=self.config['API_KEY'],
            base_url=self.config['BASE_URL']
        )
        self.model = self.config['MODEL_NAME']

    def prepare_prompts(self, dataset, system_prompt):
        """为API准备messages列表"""
        all_messages = []
        for item in dataset:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['quiz']}
            ]
            all_messages.append(messages)
        return all_messages

    async def _fetch_one(self, messages, sampling_params, semaphore):
        """异步获取单个API响应，并受信号量控制"""
        async with semaphore: # 在请求前获取一个"令牌"
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=sampling_params.get('temperature', 0.7),
                    top_p=sampling_params.get('top_p', 0.9),
                    max_tokens=sampling_params.get('max_tokens', 4096),
                    n=sampling_params.get('n', 1),
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error(f"API请求失败: {e}")
                return f"<error>API call failed: {e}</error>"
            # semaphore 在 "async with" 块结束时自动释放令牌

    async def generate(self, messages_list, sampling_params):
        """并发执行所有API请求，并使用信号量控制并发数"""
        # 从配置中读取并发限制，提供一个安全的默认值
        concurrency_limit = self.config.get("CONCURRENCY_LIMIT", 10)
        logging.info(f"API并发限制设置为: {concurrency_limit}")
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        tasks = [
            self._fetch_one(messages, sampling_params, semaphore)
            for messages in messages_list
        ]
        
        results = await asyncio_tqdm.gather(*tasks, desc="通过API生成答案")
        return results

# --- 数据处理与验证逻辑 (保持不变) ---

def load_data_from_parquets(paths):
    """从多个Parquet文件加载数据并合并。"""
    all_dfs = []
    logging.info("开始从Parquet文件加载数据...")
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                for col in ['prompt', 'solution', 'names']: # 确保这些列是对象而不是字符串
                    if col in df.columns and df[col].apply(lambda x: isinstance(x, str)).any():
                         df[col] = df[col].apply(json.loads)
                all_dfs.append(df)
                logging.info(f"  - 成功加载 {len(df)} 条数据从 {path}")
            except Exception as e:
                logging.error(f"  - 加载文件 {path} 失败: {e}")
        else:
            logging.warning(f"  - 文件未找到，跳过: {path}")

    if not all_dfs:
        logging.error("未能加载任何数据，请检查INPUT_PARQUET_PATHS配置。")
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    full_dataset = Dataset.from_pandas(combined_df)
    logging.info(f"数据加载完成。总共 {len(full_dataset)} 个谜题。")
    return full_dataset

def parse_and_verify_answer(generated_text, names, ground_truth_solution):
    """解析模型生成的答案，并与标准答案进行比对。"""
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
        if not answer_match: return False
        
        answer_text = answer_match.group(1).strip()
        matches = re.findall(r'([\w\s]+?)\s+is\s+a\s+(knight|knave)', answer_text, re.IGNORECASE)
        if not matches: return False

        parsed_solution = {}
        for name_str, status in matches:
            name = re.sub(r'^\(?\d+\)?\s*', '', name_str.strip()).strip()
            found_canonical_name = next((c_name for c_name in names if c_name.lower() == name.lower()), None)
            if found_canonical_name:
                parsed_solution[found_canonical_name] = status.lower()

        gt_map = {name: "knight" if is_knight else "knave" for name, is_knight in zip(names, ground_truth_solution)}
        
        return parsed_solution == gt_map and len(parsed_solution) == len(gt_map)
    except Exception:
        return False

# --- 主函数 ---

async def main():
    """主函数，运行推理和正确率测试。"""
    
    # 1. 加载数据
    full_dataset = load_data_from_parquets(INPUT_PARQUET_PATHS)
    if full_dataset is None: return

    # 2. 根据模式初始化推理后端
    runner = None
    if INFERENCE_MODE == 'VLLM':
        runner = VLLMRunner(VLLM_CONFIG)
    elif INFERENCE_MODE == 'API':
        runner = APIRunner(API_CONFIG)
    else:
        logging.error(f"未知的 INFERENCE_MODE: '{INFERENCE_MODE}'。请选择 'VLLM' 或 'API'。")
        return

    # 3. 准备所有生成请求
    logging.info("正在准备所有生成请求...")
    system_prompt = "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>."
    
    prompts_or_messages = runner.prepare_prompts(full_dataset, system_prompt)
    
    # 定义通用采样参数
    sampling_params = {
        "n": K_SAMPLES,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": MAX_NEW_TOKENS
    }
    
    # 4. 执行批量生成
    logging.info(f"开始使用 '{INFERENCE_MODE}' 模式进行批量生成 (共 {len(prompts_or_messages)} 个谜题)...")
    start_infer_time = time.time()

    if asyncio.iscoroutinefunction(runner.generate):
        generated_outputs = await runner.generate(prompts_or_messages, sampling_params)
    else:
        generated_outputs = runner.generate(prompts_or_messages, sampling_params)
        
    end_infer_time = time.time()
    logging.info("批量生成完成。")
    
    inference_time = end_infer_time - start_infer_time
    logging.info(f"总推理请求数: {len(prompts_or_messages)}")
    logging.info(f"推理耗时: {inference_time:.2f} 秒。")
    if inference_time > 0:
        logging.info(f"平均每秒推理请求数 (QPS): {len(prompts_or_messages) / inference_time:.2f}")

    # 5. 解析结果并计算正确率
    logging.info("正在解析结果并计算正确谜题数量...")
    correct_puzzles_count = 0
    for i, generated_text in enumerate(tqdm(generated_outputs, desc="验证结果")):
        item_data = full_dataset[i]
        is_correct = parse_and_verify_answer(generated_text, item_data['names'], item_data['solution'])
        if is_correct:
            correct_puzzles_count += 1

    # 6. 打印总结
    logging.info(f"\n--- 推理结果总结 ---")
    logging.info(f"测试模式: {INFERENCE_MODE}")
    if INFERENCE_MODE == 'VLLM':
        logging.info(f"测试模型: {VLLM_CONFIG['MODEL_ID']}")
    else:
        logging.info(f"测试模型: {API_CONFIG['MODEL_NAME']}")
        
    logging.info(f"总共谜题数量: {len(full_dataset)}")
    logging.info(f"成功正确回答的谜题数量: {correct_puzzles_count}")
    
    if len(full_dataset) > 0:
        accuracy = correct_puzzles_count / len(full_dataset) * 100
        logging.info(f"正确率: {accuracy:.2f}%")
        
    logging.info("脚本执行完毕。")


if __name__ == "__main__":
    try:
        import pyarrow
    except ImportError:
        logging.error("错误: 缺少 'pyarrow' 库。请运行 'pip install pyarrow' 来支持Parquet文件读取。")
    else:
        asyncio.run(main())