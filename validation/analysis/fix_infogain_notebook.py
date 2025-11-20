#!/usr/bin/env python3
"""
修复 InfoGain 分析 Notebook 的 GT 值填充逻辑
"""

import json
import math

def create_fixed_fill_function():
    """创建修复后的 GT 值填充函数源代码"""
    return '''def fill_gt_perplexities(sample_infogains: List[Dict]) -> List[Dict]:
    """填充 GT 相关的 perplexity 值

    Ground truth 相关的对照组只在 sample_idx=0 时计算，
    需要将这些值复制到其他样本中。

    正确处理 inf, -inf, None, NaN 值。
    """
    import math

    def is_invalid_value(value):
        """检查值是否无效（inf, -inf, None, NaN）"""
        if value is None:
            return True
        if isinstance(value, float):
            if math.isinf(value) or math.isnan(value):
                return True
        return False

    # 找到 sample_idx=0 的 GT 值
    gt_values = None
    for sample in sample_infogains:
        if sample['sample_id'] == 0:
            gt_values = {
                'perplexity_gt_action_toolcall': sample.get('perplexity_gt_action_toolcall'),
                'perplexity_gt_thought_action_toolcall': sample.get('perplexity_gt_thought_action_toolcall'),
                'perplexity_gt_full_toolcall': sample.get('perplexity_gt_full_toolcall'),
                'perplexity_baseline_action': sample.get('perplexity_baseline_action'),
                'perplexity_gt_thought_action': sample.get('perplexity_gt_thought_action'),
                'perplexity_gt_full_action': sample.get('perplexity_gt_full_action'),
                'gt_action_infogain': sample.get('gt_action_infogain'),
                'gt_thought_action_infogain': sample.get('gt_thought_action_infogain'),
                'gt_full_infogain': sample.get('gt_full_infogain'),
                'gt_thought_to_action_infogain': sample.get('gt_thought_to_action_infogain'),
                'gt_full_to_action_infogain': sample.get('gt_full_to_action_infogain'),
            }
            break

    if gt_values is None:
        print("警告: 未找到 sample_idx=0 的 GT 值")
        return sample_infogains

    # 验证 sample_id=0 的值是否有效
    invalid_keys = [k for k, v in gt_values.items() if is_invalid_value(v)]
    if invalid_keys:
        print(f"警告: sample_id=0 中以下 GT 值无效: {invalid_keys}")
        print("这些值可能无法正确填充到其他样本")

    # 填充其他样本的 GT 值
    filled_samples = []
    for sample in sample_infogains:
        sample_copy = sample.copy()
        for key, value in gt_values.items():
            # 如果当前样本的值无效，用 sample_id=0 的值替换
            current_value = sample_copy.get(key)
            if is_invalid_value(current_value):
                sample_copy[key] = value
        filled_samples.append(sample_copy)

    return filled_samples'''

# 读取现有 notebook
notebook_path = "validation/analysis/_infogain_analysis_complete.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 找到并替换 fill_gt_perplexities 函数的单元格 (cell-4)
fill_function_code = create_fixed_fill_function()

for i, cell in enumerate(notebook['cells']):
    # 找到包含 "def fill_gt_perplexities" 的代码单元格
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source

        if 'def fill_gt_perplexities' in source_text:
            print(f"找到 fill_gt_perplexities 函数在 cell {i}")

            # 替换为修复后的函数
            new_source = fill_function_code + '''

# 填充所有 query 的 GT 值
for query_result in data['detailed_results']:
    query_result['sample_infogains'] = fill_gt_perplexities(query_result['sample_infogains'])

print("✓ GT perplexity 值已填充")'''

            # 将字符串转换为列表格式（每行一个元素）
            cell['source'] = [line + '\\n' for line in new_source.split('\\n')[:-1]]
            if new_source.split('\\n')[-1]:  # 最后一行不加 \n
                cell['source'].append(new_source.split('\\n')[-1])

            print("✓ 已更新 fill_gt_perplexities 函数")
            break

# 同时更新文件路径（确保指向正确的文件）
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source

        if 'result_file = Path(' in source_text and 'sampled_agentnetbench_test_infogain' in source_text:
            print(f"找到文件路径定义在 cell {i}")

            # 更新为正确的文件路径
            new_source = '''# 读取 InfoGain 结果
result_file = Path("../guir1/outputs/Qwen3-VL-4B-Instruct/analysis/agentnetbench_test_qwen3vl_sampling16_Thought:_infogain_multiGPU.json")

with open(result_file, 'r') as f:
    data = json.load(f)

print(f"总查询数: {data['summary']['total_queries']}")
print(f"总样本数: {data['summary']['total_samples']}")
print(f"详细结果数: {len(data['detailed_results'])}")'''

            cell['source'] = [line + '\\n' for line in new_source.split('\\n')[:-1]]
            if new_source.split('\\n')[-1]:
                cell['source'].append(new_source.split('\\n')[-1])

            print("✓ 已更新文件路径")
            break

# 保存修复后的 notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\n✓ Notebook 已修复并保存: {notebook_path}")
print("\n主要修复:")
print("1. 正确处理 inf, -inf, None, NaN 值")
print("2. 使用 is_invalid_value() 函数统一检测无效值")
print("3. 文件路径已更新为正确路径")
print("\n现在可以运行 notebook 了！")
