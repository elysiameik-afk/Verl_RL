import torch
import os
import shutil
import sys

# --- 请配置您的路径 ---
# 源目录：包含您所有原始文件的文件夹
source_directory = "/root/autodl-tmp/myverl/ckpts/Qwen2.5-0.5/GRPO_KK_few/global_step_8/actor/"

# 目标目录：一个全新的、只用于存放推理所需文件的文件夹
destination_directory = "/root/autodl-tmp/myverl/ckpts/Qwen2.5-0.5/GRPO_KK_few/global_step_8/actor_inference/"

# --- 脚本主逻辑 ---

def create_inference_deployment():
    """
    创建一个干净的、仅用于推理部署的模型目录。
    """
    if not os.path.isdir(source_directory):
        print(f"错误：源目录 '{source_directory}' 不存在。")
        sys.exit(1)

    if os.path.exists(destination_directory):
        print(f"警告：目标目录 '{destination_directory}' 已存在。将先删除它再重新创建。")
        shutil.rmtree(destination_directory)
    
    os.makedirs(destination_directory)
    print(f"已创建空的推理目录: '{destination_directory}'")
    print("-" * 50)

    # 遍历源目录
    for filename in os.listdir(source_directory):
        source_path = os.path.join(source_directory, filename)
        destination_path = os.path.join(destination_directory, filename)

        # 1. 如果是模型权重文件，进行净化转换
        if filename == "model_world_size_1_rank_0.pt":
            print(f"[转换中] 正在净化模型权重: {filename}...")
            try:
                state_dict = torch.load(source_path, map_location="cpu", weights_only=False)
                # 确保加载的是一个字典（state_dict）
                if isinstance(state_dict, dict):
                    torch.save(state_dict, destination_path)
                    print(f"[成  功] 已将纯净的模型权重保存到新目录。")
                else:
                    print(f"[失  败] {filename} 加载后不是一个有效的 state_dict！")
            except Exception as e:
                print(f"[失  败] 转换 {filename} 时发生错误: {e}")
        
        # 2. 如果是优化器或额外状态文件，则直接忽略
        elif filename.startswith("optim_") or filename.startswith("extra_"):
            print(f"[忽  略] 跳过训练状态文件: {filename}")
            continue

        # 3. 其他所有文件（主要是配置文件），直接复制
        elif os.path.isfile(source_path):
            print(f"[复制中] {filename}...")
            shutil.copy2(source_path, destination_path)
        
    print("-" * 50)
    print("推理专用模型创建完成！")
    print("\n最后一步:")
    print(f"请修改您的 val.py 脚本，将模型路径指向最终目录: '{destination_directory}'")


if __name__ == "__main__":
    create_inference_deployment()