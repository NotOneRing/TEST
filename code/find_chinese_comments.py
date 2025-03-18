import os
import re

def find_chinese_comments(directory):
    chinese_pattern = re.compile(r'#.*[\u4e00-\u9fff]+')
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines, 1):
                        match = chinese_pattern.search(line)
                        if match:
                            print(f"File: {file_path}")
                            print(f"Line number: {i}")
                            print(f"Comments: {match.group().strip()}")
                            print()

current_directory = '.'
find_chinese_comments(current_directory)




























