import re
from typing import Dict, Any

# ==============================================================================
# 预编译正则表达式 (保持不变)
# ==============================================================================
_RE_THINK_START = re.compile(r'<think>')
_RE_THINK_END = re.compile(r'</think>')
_RE_ANSWER_START = re.compile(r'<answer>')
_RE_ANSWER_END = re.compile(r'</answer>')
_RE_THINK_CONTENT = re.compile(r'<think>(.*?)</think>', re.DOTALL)
_RE_ANSWER_CONTENT = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)


def _parse_gt_to_map(gt_text: str) -> Dict[str, str]:
    """
    一个辅助函数，将标准答案文本解析为 "名字: 角色" 的字典。(保持不变)
    它能从形如 "(1) Lily is a knave" 的行中提取信息。

    Args:
        gt_text: 标准答案的字符串。

    Returns:
        一个字典，例如: {'lily': 'knave', 'matthew': 'knight'}
    """
    gt_map = {}
    pattern = re.compile(r'\b([a-zA-Z]+)\b\s+is\s+a\s+\b(knight|knave)\b', re.IGNORECASE)
    for line in gt_text.splitlines():
        match = pattern.search(line)
        if match:
            name = match.group(1).lower()
            role = match.group(2).lower()
            gt_map[name] = role
    return gt_map


def compute_score(model_response: str, ground_truth: Dict[str, Any]) -> float:
    """
    一个评估思考“长度”与“质量”的增强版评分函数。

    - 保留了基础的格式和顺序检查。
    - 加大了对思考长度的奖惩范围，使“短思考”的代价更高。
    - 新增了对思考内容质量的检查：检查<think>中是否包含了谜题的关键角色名。
      一个有意义的思考过程至少应该提及它要分析的对象。
    - 答案评估部分总会被执行，确保模型因正确答案而获得奖励。
    """
    score = 0.0
    
    # --- 奖励和惩罚值定义 ---
    # 基础格式部分
    REWARD_TAG_PRESENT = 0.5
    PENALTY_TAG_MISSING_OR_DUPLICATE = -4.0  # 加大基础格式惩罚
    REWARD_ORDER_CORRECT = 1.0
    PENALTY_ORDER_INCORRECT = -5.0        # 加大基础格式惩罚
    
    # 答案评估部分
    PENALTY_CONTENT_EMPTY_ANSWER = -2.0
    REWARD_FOR_EACH_CORRECT_ASSERTION = 4

    # 思考过程评估 (长度 + 质量)
    # 1. 长度评估 (加大奖惩范围)
    MIN_THINK_REWARD = -4.0  # 从-2.0加大到-4.0
    MAX_THINK_REWARD = 4.0   # 从 2.0加大到 3.0
    TARGET_THINK_LENGTH = 2000.0 # 目标长度

    # 2. 质量评估 (新)
    REWARD_FOR_KEYWORD_IN_THINK = 1.0 # 每在<think>中提到一个关键角色名，就奖励1分

    # ==========================================================================
    # 阶段 1 & 2: 格式和顺序检查 (逻辑不变，数值调整)
    # ==========================================================================
    critical_tags_data = [
        ('think_start', _RE_THINK_START), ('think_end', _RE_THINK_END),
        ('answer_start', _RE_ANSWER_START), ('answer_end', _RE_ANSWER_END)
    ]
    tag_positions = {}
    for tag_name, pattern in critical_tags_data:
        matches = list(pattern.finditer(model_response))
        if len(matches) == 1:
            score += REWARD_TAG_PRESENT
            tag_positions[tag_name] = matches[0].start()
        else:
            return score + PENALTY_TAG_MISSING_OR_DUPLICATE

    if not (tag_positions['think_start'] < tag_positions['think_end'] < 
            tag_positions['answer_start'] < tag_positions['answer_end']):
        return score + PENALTY_ORDER_INCORRECT
    else:
        score += REWARD_ORDER_CORRECT
        
    # ==========================================================================
    # 阶段 3: <think> 内容质量评分 (长度 + 质量)
    # ==========================================================================
    think_content_match = _RE_THINK_CONTENT.search(model_response)
    think_content = think_content_match.group(1).strip() if think_content_match else ""
    think_length = len(think_content)
    
    # --- 3a. 长度评分 (使用新的、更大的奖惩范围) ---
    slope = (MAX_THINK_REWARD - MIN_THINK_REWARD) / TARGET_THINK_LENGTH
    think_length_score = (slope * think_length) + MIN_THINK_REWARD
    think_length_score = min(think_length_score, MAX_THINK_REWARD)
    score += think_length_score

    # --- 3b. 质量评分 (检查是否提及关键角色) ---
    gt_map = _parse_gt_to_map(ground_truth.get("solution_text_format", ""))
    # 如果标准答案为空或无法解析，则跳过此部分
    if gt_map:
        character_names = gt_map.keys() # 获取所有角色名, e.g., ['lily', 'matthew', 'riley']
        
        think_quality_score = 0
        think_content_lower = think_content.lower() 
        for name in character_names:
            # 使用单词边界 \b 确保匹配的是完整的名字 (例如, "son" 不会匹配到 "mason")
            if re.search(r'\b' + re.escape(name) + r'\b', think_content_lower):
                think_quality_score += REWARD_FOR_KEYWORD_IN_THINK
        score += think_quality_score
    
    # ==========================================================================
    # 阶段 4: <answer> 内容评估 (逻辑不变)
    # ==========================================================================
    answer_content_match = _RE_ANSWER_CONTENT.search(model_response)
    model_answer_text = answer_content_match.group(1).strip() if answer_content_match else ""
    
    if not model_answer_text or not gt_map:
        return score + (PENALTY_CONTENT_EMPTY_ANSWER if not model_answer_text else 0)

    num_correct_assertions = 0
    for name, correct_role in gt_map.items():
        try:
            pattern = re.compile(rf'.*\b{re.escape(name)}\b.*\b{correct_role}\b.*', re.IGNORECASE)
            if pattern.search(model_answer_text):
                num_correct_assertions += 1
        except re.error:
            continue
            
    answer_score = num_correct_assertions * REWARD_FOR_EACH_CORRECT_ASSERTION
    score += answer_score
            
    return score

# ==============================================================================
# 测试区
# ==============================================================================
if __name__ == "__main__":
    
    # 定义一个标准的谜题答案用于所有测试
    sample_ground_truth = {
        "solution_text_format": "(1) Lily is a knave\n(2) Matthew is a knight\n(3) Riley is a knight",
    }
    
    print("--- 增强版评分函数 compute_score_with_think_quality 测试 ---")

    # --- 测试用例 1: 理想情况 (思考充分且言之有物，答案全对) ---
    response_perfect = "<think>" + "a" * 600 + " Lily Matthew Riley " + "</think>" + """
    <answer>
    (1) Lily is a knave
    (2) Matthew is a knight
    (3) Riley is a knight
    </answer>
    """
    score_1 = compute_score_with_think_quality(response_perfect, sample_ground_truth)
    # 期望: 3.0(格式) + 3.0(think长度满分) + 3*1.0(think质量满分) + 3*2.0(答案满分) = 15.0
    print(f"\n[测试 1: 理想情况] -> 得分: {score_1:.2f} (期望: 15.00)")

    # --- 测试用例 2: “走过场”式思考 (长度不足，但提到了人名，答案部分正确) ---
    response_lazy_template = "<think>We need to analyze Lily, Matthew, and Riley.</think>" + """
    <answer>
    Lily is a knave, Matthew is a knight, and Riley is a knave.
    </answer>
    """
    score_2 = compute_score_with_think_quality(response_lazy_template, sample_ground_truth)
    # think长度约45。长度分 = ((3 - (-4))/500)*45 - 4 = 0.014*45-4 = 0.63-4 = -3.37
    # 质量分 = 3 * 1.0 = 3.0
    # 答案分 = 2 * 2.0 = 4.0
    # 期望: 3.0(格式) - 3.37(长度) + 3.0(质量) + 4.0(答案) = 6.63
    print(f"\n[测试 2: 走过场式思考] -> 得分: {score_2:.2f} (期望: 6.63)")
    print("   -> 分析: 即使提到了人名，但长度不足依然受到重罚。")

    # --- 测试用例 3: 思考过程言之无物 (模板句，答案部分正确) ---
    response_lazy_no_quality = "<think>Let's think step by step to solve this logic puzzle.</think>" + """
    <answer>
    Lily is a knave, Matthew is a knight, and Riley is a knave.
    </answer>
    """
    score_3 = compute_score_with_think_quality(response_lazy_no_quality, sample_ground_truth)
    # think长度约54。长度分 = 0.014*54-4 = 0.756-4 = -3.24
    # 质量分 = 0.0
    # 答案分 = 4.0
    # 期望: 3.0(格式) - 3.24(长度) + 0.0(质量) + 4.0(答案) = 3.76
    print(f"\n[测试 3: 言之无物的思考] -> 得分: {score_3:.2f} (期望: 3.76)")
    print("   -> 分析: 比测试2分数低很多，证明了思考质量的重要性。")
    
    # --- 测试用例 4: 思考充分但言之无物 (长度够，但没提人名) ---
    response_long_no_quality = "<think>" + "a" * 600 + "</think>" + """
    <answer>
    Lily is a knave, Matthew is a knight, and Riley is a knight.
    </answer>
    """
    score_4 = compute_score_with_think_quality(response_long_no_quality, sample_ground_truth)
    # 期望: 3.0(格式) + 3.0(长度) + 0.0(质量) + 6.0(答案) = 12.0
    print(f"\n[测试 4: 思考充分但言之无物] -> 得分: {score_4:.2f} (期望: 12.00)")
    print("   -> 分析: 比理想情况(15.0)低，因为它没有完成“提及角色”这个思考任务。")