import re

# ==============================================================================
# 预编译正则表达式模式 (性能优化关键)
# 这些模式在模块加载时编译一次，而不是在每次函数调用时重复编译。
# ==============================================================================

# 用于检查标签是否存在和数量
_RE_THINK_START = re.compile(r'<think>')
_RE_THINK_END = re.compile(r'</think>')
_RE_ANSWER_START = re.compile(r'<answer>')
_RE_ANSWER_END = re.compile(r'</answer>')

# 用于捕获标签内部的内容 (re.DOTALL 确保 . 能匹配换行符)
_RE_THINK_CONTENT = re.compile(r'<think>(.*?)</think>', re.DOTALL)
_RE_ANSWER_CONTENT = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

def compute_score(model_response: str,ground_truth: str,) -> float:
    """
    高效计算模型响应的格式分数。
    该函数优先检查最关键的格式问题，并在发现严重错误时提前退出以优化性能。

    Args:
        model_response: 模型的原始响应字符串。

    Returns:
        一个浮点数，表示格式符合度的分数。
    """
    score = 0.0
    
    # 定义奖励和惩罚值
    # 可以根据重要性调整这些值，以引导模型行为。
    REWARD_TAG_PRESENT = 0.5        # 每个关键标签存在且唯一
    PENALTY_TAG_MISSING_OR_DUPLICATE = -2.0 # 标签缺失或重复 (非常严重)
    REWARD_ORDER_CORRECT = 1.0      # 标签顺序正确
    PENALTY_ORDER_INCORRECT = -3.0  # 标签顺序错误 (最严重)
    REWARD_CONTENT_NON_EMPTY_THINK = 0.2 # <think> 内容不为空
    PENALTY_CONTENT_EMPTY_THINK = -0.5 # <think> 内容为空
    REWARD_CONTENT_NON_EMPTY_ANSWER = 0.5 # <answer> 内容不为空
    PENALTY_CONTENT_EMPTY_ANSWER = -1.0 # <answer> 内容为空 (答案为空是严重问题)

    # 用于存储标签的起始位置，以便检查顺序
    tag_positions = {}

    # ==========================================================================
    # 阶段 1: 检查关键标签的存在性和唯一性
    # 这是最基础的结构检查，如果有问题，立即返回。
    # ==========================================================================
    critical_tags_data = [
        ('think_start', _RE_THINK_START),
        ('think_end', _RE_THINK_END),
        ('answer_start', _RE_ANSWER_START),
        ('answer_end', _RE_ANSWER_END)
    ]

    for tag_name, pattern in critical_tags_data:
        matches = list(pattern.finditer(model_response))
        
        if len(matches) == 1:
            score += REWARD_TAG_PRESENT
            tag_positions[tag_name] = matches[0].start() # 记录起始位置
        elif len(matches) == 0:
            score += PENALTY_TAG_MISSING_OR_DUPLICATE
            # 标签缺失，结构严重破坏，直接返回
            return score 
        else: # len(matches) > 1
            score += PENALTY_TAG_MISSING_OR_DUPLICATE
            # 标签重复，结构严重破坏，直接返回
            return score
            
    # 执行到这里，说明所有四个关键标签都存在且只出现一次。

    # ==========================================================================
    # 阶段 2: 检查标签的顺序
    # 只有当所有标签都存在且唯一时才检查顺序。
    # ==========================================================================
    ts_start = tag_positions['think_start']
    te_start = tag_positions['think_end']
    as_start = tag_positions['answer_start']
    ae_start = tag_positions['answer_end']

    # 期望的顺序：<think>... </think>... <answer>... </answer>
    if ts_start < te_start and te_start < as_start and as_start < ae_start:
        score += REWARD_ORDER_CORRECT
    else:
        score += PENALTY_ORDER_INCORRECT
        # 顺序错误是严重的结构问题，直接返回
        return score

    # ==========================================================================
    # 阶段 3: 检查标签内容是否为空
    # 这是在结构和顺序都正确的前提下进行的。
    # ==========================================================================

    # 检查 <think> 标签内容
    think_content_match = _RE_THINK_CONTENT.search(model_response)
    # 确保匹配成功且捕获组1 (即内容) 存在且去除空白后不为空
    if think_content_match and think_content_match.group(1).strip():
        score += REWARD_CONTENT_NON_EMPTY_THINK
    else:
        score += PENALTY_CONTENT_EMPTY_THINK

    # 检查 <answer> 标签内容
    answer_content_match = _RE_ANSWER_CONTENT.search(model_response)
    if answer_content_match and answer_content_match.group(1).strip():
        score += REWARD_CONTENT_NON_EMPTY_ANSWER
    else:
        score += PENALTY_CONTENT_EMPTY_ANSWER
        
    return score

# ==============================================================================
# 测试用例 (用于验证函数行为)
# ==============================================================================
if __name__ == "__main__":
    print("--- 格式验证分数计算测试 ---")

    # 完美格式
    response_perfect = """
This is some leading text if any.
<think>
This is the thinking process.
It can span multiple lines.
</think>
Some text in between.
<answer>
The final answer is this.
</answer>
Trailing text.
"""
    print("\n--- 测试用例 1: 完美格式 ---")
    final_score_1 = compute_format_score(response_perfect)
    # 期望: 0.5*4 (标签存在) + 1.0 (顺序) + 0.2 (think内容) + 0.5 (answer内容) = 2.0 + 1.0 + 0.2 + 0.5 = 3.7
    print(f"最终格式分数: {final_score_1}") 

    # 缺少结束标签
    response_missing_end_tag = """
<think>
Thinking...
<answer>
Answer.
</answer>
"""
    print("\n--- 测试用例 2: 缺少 </think> 标签 ---")
    final_score_2 = compute_format_score(response_missing_end_tag)
    # 期望: -2.0 (缺少 </think>) = -2.0 (并提前退出)
    print(f"最终格式分数: {final_score_2}")

    # 标签顺序错误
    response_wrong_order = """
<answer>
My answer is here.
</answer>
<think>
My thoughts.
</think>
"""
    print("\n--- 测试用例 3: 标签顺序错误 ---")
    final_score_3 = compute_format_score(response_wrong_order)
    # 期望: 0.5*4 (标签存在) - 3.0 (顺序错误) = 2.0 - 3.0 = -1.0 (并提前退出)
    print(f"最终格式分数: {final_score_3}")

    # <answer> 内容为空
    response_empty_answer = """
<think>
Some thoughts.
</think>
<answer>
  
</answer>
"""
    print("\n--- 测试用例 4: <answer> 内容为空 ---")
    final_score_4 = compute_format_score(response_empty_answer)
    # 期望: 0.5*4 (标签存在) + 1.0 (顺序) + 0.2 (think内容) - 1.0 (answer内容为空) = 2.0 + 1.0 + 0.2 - 1.0 = 2.2
    print(f"最终格式分数: {final_score_4}")

    # 完全没有标签
    response_no_tags = "Just plain text output."
    print("\n--- 测试用例 5: 完全没有标签 ---")
    final_score_5 = compute_format_score(response_no_tags)
    # 期望: 至少一个标签缺失 (-2.0) -> 提前退出。实际会因为第一个标签 <think> 不存在而直接返回 -2.0
    print(f"最终格式分数: {final_score_5}")

    # 标签重复
    response_duplicate_tag = """
<think>
思考1
</think>
<think>
思考2
</think>
<answer>
答案
</answer>
"""
    print("\n--- 测试用例 6: <think> 标签重复 ---")
    final_score_6 = compute_format_score(response_duplicate_tag)
    # 期望: -2.0 (因为 <think> 重复)
    print(f"最终格式分数: {final_score_6}")