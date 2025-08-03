from huggingface_hub import snapshot_download
# 下载模型到指定路径
model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # 替换为你的模型ID
local_dir = "./mymodels/Qwen2.5-1.5B-Instruct"  # 替换为目标路径
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 避免符号链接，直接保存文件
    resume_download=True,          # 支持断点续传
)