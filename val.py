# ==============================================================================
# 脚本名称: vLLM 推理性能与正确率测试
#
# 脚本描述:
# 本脚本旨在测试使用vLLM推理引擎的模型在K&K逻辑谜题上的推理性能和基本正确率。
# 它从指定的Parquet文件中加载谜题数据，对每个谜题进行一次采样推理，
# 并统计成功解析并正确回答的谜题数量。
#
# 环境依赖:
# pip install torch pandas tqdm datasets pyarrow vllm
#
# 使用方法:
# 1. 确保所有依赖库已安装。
# 2. 修改下面的 CONFIGURATION (配置) 部分，特别是模型ID和文件路径。
# 3. 在终端中运行: python your_test_script_name.py
# ==============================================================================

import json
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import logging
import time # 导入time模块用于计时
import torch
# try:
#     # 我们把新旧两个不被信任的全局变量都加到安全列表里
#     safe_globals_to_add = [
#         np.dtype,
#         np._core.multiarray._reconstruct,
#         np.ndarray  # 加上 ndarray 通常也是一个好主意
#     ]
#     torch.serialization.add_safe_globals(safe_globals_to_add)
# except AttributeError:
#     # 如果 PyTorch 版本较旧，可能没有这个函数，直接忽略即可
#     pass
# 导入vLLM库
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("错误: 未找到 vLLM 库。请运行 'pip install vllm' 进行安装。")
    exit()

# --- CONFIGURATION (配置) ---

# 模型配置
# 如果你想测试你训练好的模型，请将 MODEL_ID 替换为你的检查点路径：
# MODEL_ID = "/root/autodl-tmp/myverl/ckpts/DAPO/DAPO-Qwen3-0.6b-SingleGPU-DirectRun/global_step_12/actor_inference/"
# MODEL_ID = "/root/autodl-tmp/myverl/mymodels/qwen2.5-0.5b-ins"  # 或者你想要测试的Hugging Face模型ID
MODEL_ID = "/root/autodl-tmp/myverl/ckpts/Qwen2.5-0.5/GRPO_KK_few/global_step_8/actor_inference/"
# MODEL_ID = "/root/autodl-tmp/myverl/ckpts/DAPO/DAPO-Qwen3-0.6b-think-free2/global_step_12/actor_inference/"
# 输入文件路径 (Parquet文件列表)
# !!! 重要: 请根据你的实际文件存放位置修改这里的路径 !!!
BASE_PATH = "/root/autodl-tmp/myverl/data"
INPUT_PARQUET_PATHS = [
    os.path.join(BASE_PATH, "kk/3ppl", "test.parquet"),
    # os.path.join(BASE_PATH, "train_first_95.parquet"),
    # 可以添加更多文件路径进行测试
    # os.path.join(BASE_PATH, "3ppl", "train.parquet"),
    # os.path.join(BASE_PATH, "4ppl", "train.parquet"),
]

# 采样参数 (为性能测试，只需要一次采样)
K_SAMPLES = 1          # 每个谜题生成1个答案
MAX_NEW_TOKENS = 8192    # 模型生成答案部分的最大Token数

# --- 脚本主体 ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_from_parquets(paths):
    """从多个Parquet文件加载数据并合并成一个Hugging Face Dataset对象。"""
    all_dfs = []
    logging.info("开始从Parquet文件加载数据...")
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                # 检查并解析可能为JSON字符串的列
                for col in ['prompt', 'solution', 'names']:
                    if col in df.columns and df[col].apply(lambda x: isinstance(x, str)).any(): # 只对有字符串的列尝试解析
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

    # 合并所有DataFrame并转换为Hugging Face Dataset
    combined_df = pd.concat(all_dfs, ignore_index=True)
    full_dataset = Dataset.from_pandas(combined_df)
    logging.info(f"数据加载完成。总共 {len(full_dataset)} 个谜题。")
    return full_dataset

# def parse_and_verify_answer(generated_text, names, ground_truth_solution):
#     """
#     解析模型生成的答案，并与标准答案进行比对。
#     返回 True 代表完全匹配，否则返回 False。
#     """
#     try:
#         answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
#         if not answer_match:
#             return False
#         answer_text = answer_match.group(1).strip()

#         matches = re.findall(r'([\w\s]+?)\s+is\s+a\s+(knight|knave)', answer_text, re.IGNORECASE)
        
#         if not matches:
#             return False

#         parsed_solution = {}
#         for name_str, status in matches:
#             name = name_str.strip()
#             name = re.sub(r'^\(?\d+\)?\s*', '', name).strip() # 移除名字前的序号

#             # 与标准名字列表进行匹配
#             found_canonical_name = None
#             for canonical_name in names:
#                 if canonical_name.lower() == name.lower():
#                     found_canonical_name = canonical_name
#                     break
#             if found_canonical_name:
#                 parsed_solution[found_canonical_name] = status.lower()
#             # else:
#             #     # 如果解析出的名字不在预期名字列表中，则认为解析失败
#             #     return False 
        
#         # 准备标准答案的 Map
#         gt_map = {
#             name: "knight" if is_knight else "knave"
#             for name, is_knight in zip(names, ground_truth_solution)
#         }

#         # 最终比对：确保解析出的答案和标准答案完全一致
#         # 还要确保解析出的名字数量与预期一致，且所有预期名字都被解析
#         return parsed_solution == gt_map and len(parsed_solution) == len(gt_map)
        
#     except Exception as e:
#         # logging.error(f"解析或验证答案时发生错误: {e}") # 调试时可以打开
#         return False





def parse_and_verify_answer(generated_text: str, names: list, ground_truth_solution_bools: list) -> bool:
    """
    一个更健壮的解析和验证函数，采用从后向前搜索的策略来获取最终答案。

    Args:
        generated_text: 模型生成的完整文本。
        names: 谜题中所有角色的名字列表 (e.g., ['Zoey', 'Amy', 'Sid'])。
        ground_truth_solution_bools: 标准答案的布尔值列表 (e.g., [True, False, True])。

    Returns:
        bool: 如果解析出的最终答案与标准答案完全匹配，则返回 True，否则返回 False。
    """
    # 步骤 1: 将布尔型的标准答案转换为标准的 "名字:角色" 字典，便于比较。
    try:
        gt_map = {
            name: "knight" if is_knight else "knave"
            for name, is_knight in zip(names, ground_truth_solution_bools)
        }
    except TypeError:
        # 如果标准答案格式不正确，则无法进行比较。
        logging.error(f"标准答案(Ground truth)格式错误: {ground_truth_solution_bools}")
        return False

    # 步骤 2: 从后向前解析模型生成的文本，以捕获最终结论。
    parsed_solution = {}
    found_names = set() # 用来记录已经找到身份的名字

    # 从后向前逐行扫描
    lines = generated_text.split('\n')
    for line in reversed(lines):
        # 如果已经找到了所有人的身份，就可以提前停止，提高效率。
        if len(found_names) == len(names):
            break

        # 对每个尚未找到身份的名字进行搜索
        for name in names:
            if name in found_names:
                continue

            # 使用正则表达式确保匹配的是完整的名字（\b是单词边界）
            # re.IGNORECASE 使得搜索不区分大小写
            if re.search(rf'\b{re.escape(name)}\b', line, re.IGNORECASE):
                # 如果在这一行找到了名字，就在同一行里寻找角色身份
                role_match = re.search(r'\b(knight|knave)\b', line, re.IGNORECASE)
                if role_match:
                    # 找到了！记录下来，并标记这个名字已找到。
                    parsed_solution[name] = role_match.group(1).lower()
                    found_names.add(name)
                    # 跳出内层循环，因为这个人的身份在这一行已经确定
                    break
    
    # 步骤 3: 进行最终比对
    # 必须满足两个条件：
    # 1. 成功解析出了所有角色的身份 (len(parsed_solution) == len(gt_map))
    # 2. 解析出的身份与标准答案完全一致 (parsed_solution == gt_map)
    is_correct = (len(parsed_solution) == len(gt_map)) and (parsed_solution == gt_map)

    return is_correct







def main():
    """主函数，运行推理和正确率测试。"""
    
    # --- 第1步: 从Parquet文件加载数据 ---
    full_dataset = load_data_from_parquets(INPUT_PARQUET_PATHS)
    if full_dataset is None:
        return

    # # --- 第2步: 使用vLLM加载模型 ---
    # logging.info(f"使用vLLM加载模型 '{MODEL_ID}'...")
    # start_load_time = time.time()
    # # tensor_parallel_size=1 代表使用单张GPU。trust_remote_code=True 对某些模型是必须的。
    # llm = LLM(model=MODEL_ID, tensor_parallel_size=1, trust_remote_code=True)
    # tokenizer = llm.get_tokenizer()
    # end_load_time = time.time()
    # logging.info(f"模型加载完成，耗时: {end_load_time - start_load_time:.2f} 秒。")



    # --- 第2步: 使用vLLM加载模型 ---
    logging.info(f"使用vLLM加载模型 '{MODEL_ID}'...")
    start_load_time = time.time()
    
    llm = LLM(
            model=MODEL_ID, 
            tensor_parallel_size=1, 
            trust_remote_code=True,
            # load_format="PT" # <-- 在这里修改
        )
    tokenizer = llm.get_tokenizer()
    end_load_time = time.time()
    logging.info(f"模型加载完成，耗时: {end_load_time - start_load_time:.2f} 秒。")











    
    # --- 第3步: 准备所有生成请求 ---
    logging.info("正在准备所有生成请求...")
    system_prompt = "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>."
    # system_prompt = "You are a helpful assistant. The reasoning process and answer are enclosed within <answer> </answer> tags, respectively, i.e.<answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem.  When you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>."    
    all_prompts = []
    for item in full_dataset:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item['quiz']}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(prompt_text)

    # 定义采样参数。K_SAMPLES=1，表示每个prompt只生成一个序列。
    sampling_params = SamplingParams(
        n=K_SAMPLES,
        temperature=0.7, # 保持一定的随机性，但可以调低到0以获取更确定性输出
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS
    )

    # --- 第4步: 执行批量生成 ---
    logging.info(f"开始使用vLLM进行批量生成 (共 {len(all_prompts)} 个谜题, 每个谜题采样 {K_SAMPLES} 次)...")
    start_infer_time = time.time()
    outputs = llm.generate(all_prompts, sampling_params)
    end_infer_time = time.time()
    logging.info("批量生成完成。")
    
    total_inferences = len(all_prompts) * K_SAMPLES
    logging.info(f"总推理请求数: {len(all_prompts)}")
    logging.info(f"总生成样本数: {total_inferences}")
    logging.info(f"推理耗时: {end_infer_time - start_infer_time:.2f} 秒。")
    logging.info(f"平均每秒推理请求数 (QPS): {len(all_prompts) / (end_infer_time - start_infer_time):.2f}")
    logging.info(f"平均每秒生成样本数: {total_inferences / (end_infer_time - start_infer_time):.2f}")


    # --- 第5步: 解析结果并计算正确率 ---
    logging.info("正在解析结果并计算正确谜题数量...")
    correct_puzzles_count = 0
    for i, item_output in enumerate(tqdm(outputs, desc="验证结果")):
        item_data = full_dataset[i]
        
        # 只需要 K_SAMPLES 中的第一个生成结果进行评估 (因为 K_SAMPLES=1)
        # 如果 K_SAMPLES > 1，这里可以取 outputs[0] 或其他策略
        generated_text = item_output.outputs[0].text 
        
        is_correct = parse_and_verify_answer(generated_text, item_data['names'], item_data['solution'])
        if is_correct:
            correct_puzzles_count += 1
        else: # 调试时可以打开，看看哪些谜题错了以及模型输出了什么
            logging.warning(f"谜题 {i} 回答错误或格式不正确。")
            logging.warning(f"  Prompt: {item_output.prompt[:200]}...") # 打印部分Prompt
            logging.warning(f"  Ground Truth: {item_data['solution']}")
            logging.warning(f"  Generated Text: {generated_text}")

    logging.info(f"\n--- 推理结果总结 ---")
    logging.info(f"总共谜题数量: {len(full_dataset)}")
    logging.info(f"成功正确回答的谜题数量: {correct_puzzles_count}")
    logging.info(f"正确率: {correct_puzzles_count / len(full_dataset) * 100:.2f}%")
    logging.info("脚本执行完毕。")


if __name__ == "__main__":
    try:
        import pyarrow
    except ImportError:
        logging.error("错误: 缺少 'pyarrow' 库。请运行 'pip install pyarrow' 来支持Parquet文件读取。")
    else:
        main()