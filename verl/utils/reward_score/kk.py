# import re
# from typing import Dict, Tuple, Optional

# # def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
# #     """Extracts the final answer from the model's response string.
    
# #     Args:
# #         solution_str: Raw response string from the language model
        
# #     Returns:
# #         Tuple containing (extracted_answer, processed_string)
# #     """
# #     # Split response to isolate assistant output
# #     if "Assistant:" in solution_str:
# #         processed_str = solution_str.split("Assistant:", 1)[1]
# #     elif "<|im_start|>assistant" in solution_str:
# #         processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
# #     else:
# #         print("[Error] Failed to locate model response header")
# #         return None, solution_str

# #     # Extract final answer using XML-style tags
# #     answer_pattern = r'<answer>(.*?)</answer>'
# #     matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
# #     if not matches:
# #         print("[Error] No valid answer tags found")
# #         return None, processed_str
        
# #     final_answer = matches[-1].group(1).strip()
# #     return final_answer, processed_str


# def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
#     """Extracts the final answer from the model's response string.
    
#     Args:
#         solution_str: Raw response string from the language model
        
#     Returns:
#         Tuple containing (extracted_answer, processed_string)
#     """
#     # **修改开始**
#     # 不再强制要求 'Assistant:' 或 '<|im_start|>assistant' 头部。
#     # 整个 solution_str 将作为 processed_str 进行后续的标签解析。
#     processed_str = solution_str
#     # 打印一个警告，以防用户确实期望有头部但模型没有生成
#     print("[Warning] 'Assistant:' or '<|im_start|>assistant' header not found. Proceeding with full response for tag extraction.")
#     # **修改结束**

#     # Extract final answer using XML-style tags
#     answer_pattern = r'<answer>(.*?)</answer>'
#     matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
#     if not matches:
#         print("[Error] No valid answer tags found")
#         return None, processed_str
        
#     final_answer = matches[-1].group(1).strip()
#     return final_answer, processed_str




# def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
#     """Parses ground truth solution text into status dictionary.
    
#     Args:
#         solution_text: Formatted solution text from dataset
        
#     Returns:
#         Dictionary mapping character names to their roles (knight/knave)
#     """
#     status_dict = {}
#     print("\n[Ground Truth Parsing]")
    
#     for line in solution_text.split('\n'):
#         line = line.strip()
#         if not line:
#             continue
            
#         match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
#         if match:
#             name, role = match.groups()
#             status_dict[name] = role.lower()
#             print(f"  Found: {name} → {role}")
#         else:
#             print(f"  [Warning] Unparseable line: '{line}'")
    
#     return status_dict

# def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
#     """Parses model's answer text into status dictionary.
    
#     Args:
#         answer_text: Text extracted from model's <answer> tags
#         expected_names: List of character names requiring identification
        
#     Returns:
#         Dictionary mapping character names to predicted roles, or None if incomplete
#     """
#     status_dict = {}
#     print("\n[Model Answer Parsing]")
#     print(f"  Expected characters: {expected_names}")

#     knight_count = answer_text.lower().count('knight')
#     knave_count = answer_text.lower().count('knave')

#     print(f"  Number of predicted roles: {knight_count + knave_count}")
#     if knight_count + knave_count != len(expected_names):
#         print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
#         return None

#     for name in expected_names:
#         pattern = re.compile(
#             rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
#             re.IGNORECASE
#         )
#         match = pattern.search(answer_text)
        
#         if match:
#             role = match.group(1).lower()
#             status_dict[name] = role
#             print(f"  Found: {name} → {role}")
#         else:
#             print(f"  [Error] Missing identification for {name}")
#             return None
    
#     return status_dict

# def validate_response_structure(processed_str: str) -> bool:
#     """Performs comprehensive validation of response structure.
    
#     Args:
#         processed_str: Processed response string from the model
        
#     Returns:
#         Boolean indicating whether all formatting requirements are met
#     """
#     print("\n[Structure Validation]")
#     validation_passed = True

#     # Check required tags
#     tags = {
#         'think_start': ('<think>', 1),
#         'think_end': ('</think>', 1),
#         'answer_start': ('<answer>', 1),
#         'answer_end': ('</answer>', 1)
#     }

#     positions = {}
#     for tag_name, (tag_str, expected_count) in tags.items():
#         count = processed_str.count(tag_str)
#         positions[tag_name] = pos = processed_str.find(tag_str)
        
#         print(f"  {tag_str}: count={count}, position={pos}")
        
#         if count != expected_count:
#             print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
#             validation_passed = False

#     # Verify tag order
#     if (positions['think_start'] > positions['think_end'] or
#         positions['think_end'] > positions['answer_start'] or
#         positions['answer_start'] > positions['answer_end']):
#         print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
#         validation_passed = False
#     else:
#         print("  Tag sequence validation passed")

#     return validation_passed

# def compute_score(solution_str: str, 
#                  ground_truth: Dict[str, str],
#                  format_reward: int = 1,
#                  answer_reward: float = 1.0) :
#     """Computes comprehensive score for model response.
    
#     Args:
#         solution_str: Raw model response string
#         ground_truth: Dictionary containing ground truth data
#         format_reward: Points awarded/deducted for format correctness
#         answer_reward: Points awarded/deducted for answer correctness
        
#     Returns:
#         Total score (sum of format and answer rewards)
#     """
#     print("\n" + "="*80)
#     print(" Processing New Sample ".center(80, '='))
    
#     # Parse ground truth data
#     solution_text = ground_truth.get('solution_text_format', '')
#     gt_status = parse_solution_text_format(solution_text)
#     expected_names = list(gt_status.keys())
#     print(f"[Ground Truth] Final identities: {gt_status}")

#     # Extract model answer
#     answer_text, processed_str = extract_solution(solution_str)
#     print(f"\n[Model Response]\n{processed_str}")

#     # Validate response structure
#     format_correct = validate_response_structure(processed_str)
#     format_score = format_reward if format_correct else -abs(format_reward)
#     print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
#     print(f"  Format score: {format_score}")

#     # Validate answer content
#     answer_score = 0
#     if format_correct and answer_text:
#         pred_status = parse_model_answer(answer_text, expected_names)
#         if pred_status:
#             print(f"\n[Content Validation]")
#             print(f"  Expected: {gt_status}")
#             print(f"  Predicted: {pred_status}")
            
#             if pred_status == gt_status:
#                 answer_score = 2
#                 print("  Content validation: FULL MATCH")
#             else:
#                 answer_score = -1.5
#                 print("  Content validation: MISMATCH")
#         else:
#             answer_score = -2
#             print( "Fail to parse answer")
#     else:
#         answer_score = -2
#         print("\n[Content Validation] Skipped due to format errors or missing answer")

#     total_score = format_score + answer_score
#     print("\n" + "-"*80)
#     print(f" Final Score ".center(80, '-'))
#     print(f"  Format: {format_score}")
#     print(f"  Answer: {answer_score}")
#     print(f"  Total: {total_score}")
#     print("="*80 + "\n")

#     return total_score




# import re


# from typing import Dict, Optional, List
# # --------------------------------------------------------------------------
# # 函数 'extract_solution' 和 'validate_response_structure' 已被移除，因为不再需要。
# # --------------------------------------------------------------------------


# def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
#     """解析数据集中的标准答案文本。 (此函数保持不变)"""
#     status_dict = {}
#     # print("\n[Ground Truth Parsing]")
    
#     for line in solution_text.split('\n'):
#         line = line.strip()
#         if not line:
#             continue
            
#         match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
#         if match:
#             name, role = match.groups()
#             status_dict[name] = role.lower()
#             # print(f"  Found: {name} → {role}")
#         else:
#             pass
#             # print(f"  [Warning] Unparseable line: '{line}'")
    
#     return status_dict
# def parse_model_answer(model_response: str, expected_names: List[str]) -> Optional[Dict[str, str]]:
#     """
#     更健壮的模型响应解析器。
#     它能处理 "Name is a role", "Name: role", "Name - role" 等多种格式。
#     """
#     status_dict = {}
#     # print("\n[Robust Model Answer Parsing]")
#     # print(f"  Expected characters: {expected_names}")

#     # 将整个响应文本按行分割，逐行处理
#     lines = model_response.split('\n')
    
#     for name in expected_names:
#         found_for_name = False
#         for line in lines:
#             # 检查当前行是否与当前要找的名字相关
#             if re.search(rf'\b{re.escape(name)}\b', line, re.IGNORECASE):
#                 # 如果相关，再查找角色
#                 match = re.search(r'\b(knight|knave)\b', line, re.IGNORECASE)
#                 if match:
#                     role = match.group(1).lower()
#                     if name not in status_dict: # 防止同一角色被多次赋值
#                         status_dict[name] = role
#                         # print(f"  Found: {name} → {role} in line: '{line.strip()}'")
#                     found_for_name = True
#                     break # 找到了这个名字的身份，跳出内层循环，继续找下一个人
        
#         # if not found_for_name:
#         #     print(f"  [Warning] Could not find identification for {name}")

#     # 只要解析出的角色数量与预期一致，就认为成功
#     if len(status_dict) == len(expected_names):
#         # print("  [Result] Successfully parsed a complete answer.")
#         return status_dict
#     else:
#         # print(f"  [Result] Failed to parse a complete answer. Found {len(status_dict)}/{len(expected_names)}.")
#         return None



# def compute_score(solution_str: str, 
#                  ground_truth: Dict[str, str],
#                  correct_answer_reward: float = 2.0,
#                  wrong_answer_penalty: float = -1.5) -> float:
#     """
#     计算模型响应的简化分数，只关心答案的正确性。

#     Args:
#         solution_str: 模型的原始响应字符串
#         ground_truth: 包含标准答案的字典
#         correct_answer_reward: 答案完全正确时获得的分数
#         wrong_answer_penalty: 答案不正确或无法解析时获得的分数
        
#     Returns:
#         最终的分数
#     """
#     # print("\n" + "="*80)
#     # print(" Processing New Sample (Simplified Scoring) ".center(80, '='))
    
#     # 1. 解析标准答案
#     solution_text = ground_truth.get('solution_text_format', '')
#     gt_status = parse_solution_text_format(solution_text)
#     expected_names = list(gt_status.keys())
#     # print(f"[Ground Truth] Final identities: {gt_status}")

#     # 2. 直接解析模型的完整输出
#     # print(f"\n[Model Response]\n{solution_str}")
#     pred_status = parse_model_answer(solution_str, expected_names)

#     # 3. 计算分数
#     # print("\n[Content Validation & Scoring]")
#     if pred_status:
#         # 如果成功解析出答案
#         # print(f"  Expected:  {gt_status}")
#         # print(f"  Predicted: {pred_status}")
        
#         if pred_status == gt_status:
#             total_score = correct_answer_reward
#             # print("  Result: FULL MATCH")
#         else:
#             total_score = wrong_answer_penalty
#             # print("  Result: MISMATCH")
#     else:
#         # 如果无法从模型输出中解析出完整的答案
#         total_score = wrong_answer_penalty
#     #     print("  Result: FAIL TO PARSE")

#     # print("\n" + "-"*80)
#     # print(f" Final Score: {total_score} ".center(80, '-'))
#     # print("="*80 + "\n")

#     return total_score










import re
from typing import Dict, Optional, List

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """解析数据集中的标准答案文本。 (此函数保持不变)"""
    status_dict = {}
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
    return status_dict

def parse_model_answer(model_response: str, expected_names: List[str]) -> Optional[Dict[str, str]]:
    """
    更健壮的模型响应解析器，从后向前查找以获取最终答案。
    这避免了将在中间推理过程中提到的身份误判为最终结论。
    """
    status_dict = {}
    found_names = set()
    
    # 从后向前逐行扫描模型的响应
    lines = model_response.split('\n')
    for line in reversed(lines):
        # 如果已经找到了所有人的身份，可以提前退出以提高效率
        if len(found_names) == len(expected_names):
            break
            
        # 在当前行中查找尚未确定身份的角色
        for name in expected_names:
            if name in found_names:
                continue

            # 使用\b确保匹配的是完整的单词，避免 "Abigail" 匹配 "Abigail's"
            if re.search(rf'\b{re.escape(name)}\b', line, re.IGNORECASE):
                # 如果找到了名字，就在同一行里寻找身份
                match = re.search(r'\b(knight|knave)\b', line, re.IGNORECASE)
                if match:
                    role = match.group(1).lower()
                    status_dict[name] = role
                    found_names.add(name)
                    # 找到了这个名字的身份，跳出内层循环，继续处理下一行
                    break 

    # 只有当所有角色的身份都被成功解析时，才认为这是一个有效的答案
    if len(status_dict) == len(expected_names):
        return status_dict
    else:
        return None

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 correct_answer_reward: float = 2.0,
                 wrong_answer_penalty: float = -1.5) -> float:
    """
    计算模型响应的简化分数，只关心答案的正确性。
    (此函数内部逻辑不变，但现在会调用新的、更可靠的解析器)
    """
    # 1. 解析标准答案
    solution_text = ground_truth.get('solution_text_format', '')
    gt_status = parse_solution_text_format(solution_text)
    if not gt_status:
        # 如果标准答案为空或无法解析，无法评分
        return 0.0
    expected_names = list(gt_status.keys())

    # 2. 使用新的、从后向前的解析器来解析模型的完整输出
    pred_status = parse_model_answer(solution_str, expected_names)

    # 3. 计算分数
    if pred_status:
        # 如果成功解析出答案
        if pred_status == gt_status:
            # 答案完全正确
            total_score = correct_answer_reward
        else:
            # 答案部分正确或完全错误
            total_score = wrong_answer_penalty
    else:
        # 如果无法从模型输出中解析出完整的答案
        total_score = wrong_answer_penalty

    return total_score