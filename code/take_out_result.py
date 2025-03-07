# file path
file_path = "tf_run_py_pretrain_hopper_medium_v2.txt"

# open file and take out lines with "train loss"
with open(file_path, "r") as file:
    # read line by line
    matching_lines = [line for line in file if ("train loss" in line or "Epoch" in line)]

# print lines containing "train loss"
for line in matching_lines:
    print(line.strip())  # Removes trailing newline

output_file_path = "train_loss_lines.txt"

with open(output_file_path, "w") as output_file:
    output_file.writelines(matching_lines)

print(f"Extraction completed. Results saved to {output_file_path}")
























