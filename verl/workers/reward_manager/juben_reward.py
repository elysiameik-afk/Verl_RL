# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import defaultdict
import torch
from openai import OpenAI, RateLimitError, APIError

from verl import DataProto
from .registry import register

# --- 新增：LLMQualityRewardManager ---

# LLM 裁判（Judge）的配置常量
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JUDGE_MODEL = "gemini-2.5-flash"  # 使用最新且高效的 Flash 模型
# 这是给 LLM 裁判的系统指令（System Prompt），对于获得稳定、可靠的评分至关重要。
JUDGE_SYSTEM_PROMPT = """
You are a talented musical playwright. Now you need to rate other people's scripts. From 0 (very bad) to 10 (very good), 5 is considered medium. You should consider the following points comprehensively:
1. The structure of the story: its logic, fluency, and whether the conflicts are engaging.
2. Whether the character development is complete: the rationality of the character's motives, the overall integrity of the character arc, and the three-dimensionality of the characters.
3. The entertainment value and readability.
4. The emotional intensity in the lyrics.
The answer must be a single number, without any explanation or additional text.
"""


@register("llm_quality_reward")
class LLMQualityRewardManager:
    """
    一个使用外部大语言模型（如 Gemini）作为“裁判”来评估生成回答质量的 Reward Manager。
    LLM 裁判给出的分数将直接作为奖励（reward）。
    """

    def __init__(self, tokenizer, num_examine, gemini_api_key: str = "AIzaSyCRTfMX1QSXItSiP6VjwRXyp6wWWnZEUYI", **kwargs) -> None:
        """
        参数:
            tokenizer: 模型的 tokenizer。
            num_examine: 在控制台中打印多少个样本用于调试。
            gemini_api_key: 用于 Gemini 服务的 API 密钥。强烈建议通过配置文件传入，而不是硬编码。
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.print_counter = 0

        # --- 配置用于 Gemini 的 OpenAI 客户端 ---
        # 从环境变量中获取 API 密钥是更安全的方式。
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("未找到 GEMINI_API_KEY。请将其设置为环境变量或在配置文件中传入。")
        
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=GEMINI_BASE_URL
            )
        except Exception as e:
            raise ImportError(f"无法初始化 OpenAI 客户端。请确保已安装 'openai' 库 (`pip install openai`)。错误: {e}")

    def _get_llm_judge_score(self, prompt_str: str, response_str: str) -> float:
        """
        调用 LLM 裁判 API 来获取一个质量分数。
        
        返回:
            一个浮点数分数。如果发生 API 错误或解析失败，则返回 0.0。
        """
        # 如果回答为空，则直接返回 0 分，避免无效的 API 调用。
        if not response_str or not response_str.strip():
            return 0.0

        try:
            # 构建发送给 LLM 裁判的用户内容
            user_content = f"**PROMPT (用户提示):**\n{prompt_str}\n\n**RESPONSE TO EVALUATE (待评估的回答):**\n{response_str}"
            
            response = self.client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0, # 我们需要确定性的、可复现的评估，所以温度设为 0
                max_tokens=10,   # 只需要几个 token 来返回一个数字
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # 尝试将返回的文本转换为浮点数。
            # 由于我们的 prompt 强制要求只返回数字，这通常会成功。
            score = float(score_text)
            return score

        except (RateLimitError, APIError) as e:
            print(f"[警告] LLM 裁判 API 调用失败: {e}。返回奖励 0.0。")
            return 0.0
        except (ValueError, IndexError) as e:
            print(f"[警告] 无法解析 LLM 裁判返回的分数 ('{score_text}')。错误: {e}。返回奖励 0.0。")
            return 0.0
        except Exception as e:
            print(f"[警告] 调用 LLM 裁判时发生意外错误: {e}。返回奖励 0.0。")
            return 0.0


    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]

            # 解码 prompt 和 response
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            # 确保响应长度大于0，避免切片错误
            if valid_response_length <= 0:
                continue
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # --- 从 LLM 裁判获取分数 ---
            score = self._get_llm_judge_score(prompt_str, response_str)
            
            # 存储分数信息，并将其分配给奖励张量
            reward_extra_info["llm_judge_score"].append(score)
            # 奖励通常被分配给序列的最后一个 token
            reward_tensor[i, valid_response_length - 1] = score

            # 打印调试信息
            if self.print_counter < self.num_examine:
                self.print_counter += 1
                print("=" * 80)
                print(f"[用户提示 PROMPT]\n{prompt_str}")
                print(f"\n[模型回答 RESPONSE]\n{response_str}")
                print(f"\n[LLM 裁判评分 JUDGE SCORE]: {score:.2f}")
                print("=" * 80)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor