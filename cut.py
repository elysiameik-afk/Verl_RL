import pandas as pd

# --- 请根据您的实际情况修改这里的变量 ---
# 输入文件：您的大Parquet文件路径
input_file_path = "./data/kk/kk_few/train.parquet"

# 输出文件：您想保存的新Parquet文件名
output_file_path = "train_first_95.parquet"

# 要截取的数据行数
num_rows_to_keep = 95
# -----------------------------------------

try:
    # 1. 读取原始的Parquet文件
    print(f"正在读取原始 Parquet 文件: {input_file_path} ...")
    df = pd.read_parquet(input_file_path)
    
    original_rows = len(df)
    print(f"✅ 读取成功！原始文件共有 {original_rows} 条数据。")
    df_subset = df.head(2)
    print(df_subset)
    output_path = "testoutput.json"  # 你可以修改为你想要的路径和文件名
    df_subset.to_json(output_path, orient="records", indent=2, force_ascii=False)
    
    print(f"✅ 数据已成功保存到 {output_path}")
#     # 2. 截取前 num_rows_to_keep 条数据
#     print(f"\n正在截取前 {num_rows_to_keep} 条数据...")
#     # 使用 .head() 方法可以方便地获取前N行
#     df_subset = df.head(num_rows_to_keep)
    
#     # 打印提示信息，告知用户实际截取了多少行（以防原文件行数不足）
#     actual_rows = len(df_subset)
#     if original_rows < num_rows_to_keep:
#         print(f"⚠️ 警告：原始文件只有 {original_rows} 行，不足 {num_rows_to_keep} 行。将截取所有行。")
#     print(f"截取完成，新的数据集中将包含 {actual_rows} 条数据。")

#     # 3. 将截取后的数据保存为新的Parquet文件
#     print(f"\n正在将截取的数据保存到新的 Parquet 文件: {output_file_path} ...")
#     # 使用 .to_parquet() 方法保存
#     # index=False 表示不将DataFrame的索引（0, 1, 2...）保存为文件中的一列
#     df_subset.to_parquet(output_file_path, index=False)
        
#     print(f"\n🎉 操作成功！")
#     print(f"已将前 {actual_rows} 条数据保存到 '{output_file_path}' 文件中。")
#     print("新文件的内容和列格式与原文件完全一致。")

except FileNotFoundError:
    print(f"❌ 错误：找不到文件 '{input_file_path}'。请确保文件名和路径正确。")
except Exception as e:
    print(f"❌ 处理文件时发生错误: {e}")