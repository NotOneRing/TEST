# 文件路径
file_path = "tf_run_py_pretrain_hopper_medium_v2.txt"

# 打开文件并提取包含 "train loss" 的行
with open(file_path, "r") as file:
    # 逐行读取
    matching_lines = [line for line in file if ("train loss" in line or "Epoch" in line)]

# 打印包含 "train loss" 的行
for line in matching_lines:
    print(line.strip())  # 去除行尾的换行符

output_file_path = "train_loss_lines.txt"

with open(output_file_path, "w") as output_file:
    output_file.writelines(matching_lines)

print(f"提取完成，结果保存在 {output_file_path}")

