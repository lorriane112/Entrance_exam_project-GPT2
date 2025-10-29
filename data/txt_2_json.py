# txt_to_json_lines.py
import json
import os

def txt_2_json(txt_file, json_file):
    """将TXT文件的每一行转换为JSON的一个条目"""
    
    print(f"正在转换: {txt_file} -> {json_file}")
    
    # 读取TXT文件
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"读取到 {len(lines)} 行")
    
    # 创建JSON数据
    json_data = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line and len(line) > 10:  # 只保留非空且长度大于10的行
            json_data.append({
                "id": i + 1,
                "text": line,
                "length": len(line)
            })
    
    print(f"有效数据条数: {len(json_data)}")
    
    # 保存为JSON文件
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ JSON文件已创建: {json_file}")
    
    # 显示样本
    if json_data:
        print("\n样本数据:")
        for i in range(min(3, len(json_data))):
            print(f"  {json_data[i]['text'][:50]}...")

if __name__ == "__main__":
    txt_2_json('input.txt', 'input.json')