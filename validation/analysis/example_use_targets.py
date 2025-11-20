#!/usr/bin/env python3
"""
使用示例：如何使用包含 targets 信息的 InfoGain 结果文件
"""

import json
from pathlib import Path
from typing import Dict, List


def example_1_basic_access():
    """示例 1: 基本访问 - 读取和查看 targets"""
    print("=" * 80)
    print("示例 1: 基本访问 targets")
    print("=" * 80)

    file_path = Path("/data1/zlx/workspace/EasyR1/validation/guir1/outputs/Qwen3-VL-4B-Instruct/analysis/agentnetbench_test_qwen3vl_sampling16_Thought:_infogain_multiGPU_with_targets.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    # 获取第一个 query
    first_query = data['detailed_results'][0]
    query_id = first_query['query_id']

    print(f"\nQuery ID: {query_id}")

    # Per-query targets
    per_query_targets = first_query['inference_targets']['per_query_targets']
    print(f"\nPer-query targets (共 {len(per_query_targets)} 个):")
    for target_name in per_query_targets.keys():
        print(f"  - {target_name}")

    # Per-sample targets
    per_sample_targets = first_query['inference_targets']['per_sample_targets']
    print(f"\nPer-sample targets (共 {len(per_sample_targets)} 个 samples):")
    print(f"  每个 sample 有 {len(per_sample_targets[0]['targets'])} 个 targets")


def example_2_compare_targets():
    """示例 2: 比较不同 samples 的 targets"""
    print("\n" + "=" * 80)
    print("示例 2: 比较不同 samples 的模型 targets")
    print("=" * 80)

    file_path = Path("/data1/zlx/workspace/EasyR1/validation/guir1/outputs/Qwen3-VL-4B-Instruct/analysis/agentnetbench_test_qwen3vl_sampling16_Thought:_infogain_multiGPU_with_targets.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    first_query = data['detailed_results'][0]

    # 比较前 3 个 samples 的 model_thought_action_toolcall targets
    print(f"\n比较前 3 个 samples 的 'model_thought_action_toolcall' targets:\n")

    for i in range(min(3, len(first_query['inference_targets']['per_sample_targets']))):
        sample_targets = first_query['inference_targets']['per_sample_targets'][i]
        target = sample_targets['targets']['model_thought_action_toolcall']['target']

        print(f"Sample {i}:")
        # 只显示前 150 个字符
        print(f"  {target[:150]}...")
        print()


def example_3_count_inference_requirements():
    """示例 3: 统计推理需求"""
    print("=" * 80)
    print("示例 3: 统计推理需求")
    print("=" * 80)

    file_path = Path("/data1/zlx/workspace/EasyR1/validation/guir1/outputs/Qwen3-VL-4B-Instruct/analysis/agentnetbench_test_qwen3vl_sampling16_Thought:_infogain_multiGPU_with_targets.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    total_queries = len(data['detailed_results'])

    # 使用第一个 query 来确定结构
    first_query = data['detailed_results'][0]
    per_query_count = len(first_query['inference_targets']['per_query_targets'])
    per_sample_count = len(first_query['inference_targets']['per_sample_targets'][0]['targets'])
    num_samples = len(first_query['inference_targets']['per_sample_targets'])

    print(f"\n数据集统计:")
    print(f"  总 queries: {total_queries}")
    print(f"\n每个 query:")
    print(f"  Per-query targets: {per_query_count} 个 (只推理一次)")
    print(f"  Per-sample targets: {per_sample_count} 个 × {num_samples} samples = {per_sample_count * num_samples} 次")
    print(f"  每个 query 总推理: {per_query_count + per_sample_count * num_samples} 次")

    print(f"\n整个数据集:")
    total_per_query_inferences = total_queries * per_query_count
    total_per_sample_inferences = total_queries * per_sample_count * num_samples
    total_inferences = total_per_query_inferences + total_per_sample_inferences

    print(f"  Per-query 推理: {total_per_query_inferences:,} 次")
    print(f"  Per-sample 推理: {total_per_sample_inferences:,} 次")
    print(f"  总推理次数: {total_inferences:,} 次")


def example_4_extract_specific_targets():
    """示例 4: 提取特定类型的 targets"""
    print("\n" + "=" * 80)
    print("示例 4: 提取特定类型的 targets")
    print("=" * 80)

    file_path = Path("/data1/zlx/workspace/EasyR1/validation/guir1/outputs/Qwen3-VL-4B-Instruct/analysis/agentnetbench_test_qwen3vl_sampling16_Thought:_infogain_multiGPU_with_targets.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    # 提取所有 queries 的 GT Full targets
    print("\n提取前 3 个 queries 的 'gt_full_toolcall' targets:")

    for i, query_result in enumerate(data['detailed_results'][:3]):
        query_id = query_result['query_id']
        gt_full_target = query_result['inference_targets']['per_query_targets']['gt_full_toolcall']

        print(f"\nQuery {i+1}: {query_id}")
        print(f"  描述: {gt_full_target['description']}")
        print(f"  测量部分: {gt_full_target['measure_part']}")
        print(f"  Target 长度: {len(gt_full_target['target'])} 字符")


def example_5_group_by_measure_part():
    """示例 5: 按 measure_part 分组 targets"""
    print("\n" + "=" * 80)
    print("示例 5: 按 measure_part 分组 targets")
    print("=" * 80)

    file_path = Path("/data1/zlx/workspace/EasyR1/validation/guir1/outputs/Qwen3-VL-4B-Instruct/analysis/agentnetbench_test_qwen3vl_sampling16_Thought:_infogain_multiGPU_with_targets.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    first_query = data['detailed_results'][0]

    # 分类 per-query targets
    tool_call_targets = []
    action_targets = []

    for target_name, target_info in first_query['inference_targets']['per_query_targets'].items():
        if target_info['measure_part'] == 'tool_call':
            tool_call_targets.append(target_name)
        elif target_info['measure_part'] == 'action':
            action_targets.append(target_name)

    print("\nPer-query targets 分类:")
    print(f"  用于预测 tool_call 的 ({len(tool_call_targets)} 个):")
    for name in tool_call_targets:
        print(f"    - {name}")

    print(f"\n  用于预测 Action 的 ({len(action_targets)} 个):")
    for name in action_targets:
        print(f"    - {name}")

    # 分类 per-sample targets (使用第一个 sample)
    sample_targets = first_query['inference_targets']['per_sample_targets'][0]['targets']
    sample_tool_call = []
    sample_action = []

    for target_name, target_info in sample_targets.items():
        if target_info['measure_part'] == 'tool_call':
            sample_tool_call.append(target_name)
        elif target_info['measure_part'] == 'action':
            sample_action.append(target_name)

    print("\nPer-sample targets 分类:")
    print(f"  用于预测 tool_call 的 ({len(sample_tool_call)} 个):")
    for name in sample_tool_call:
        print(f"    - {name}")

    print(f"\n  用于预测 Action 的 ({len(sample_action)} 个):")
    for name in sample_action:
        print(f"    - {name}")


def example_6_calculate_target_lengths():
    """示例 6: 统计 target 长度分布"""
    print("\n" + "=" * 80)
    print("示例 6: 统计 target 长度分布")
    print("=" * 80)

    file_path = Path("/data1/zlx/workspace/EasyR1/validation/guir1/outputs/Qwen3-VL-4B-Instruct/analysis/agentnetbench_test_qwen3vl_sampling16_Thought:_infogain_multiGPU_with_targets.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    # 统计前 100 个 queries 的 target 长度
    print("\n前 100 个 queries 的平均 target 长度:")

    target_lengths = {}

    for query_result in data['detailed_results'][:100]:
        # Per-query targets
        for target_name, target_info in query_result['inference_targets']['per_query_targets'].items():
            if target_name not in target_lengths:
                target_lengths[target_name] = []
            target_lengths[target_name].append(len(target_info['target']))

        # Per-sample targets (只统计第一个 sample)
        if query_result['inference_targets']['per_sample_targets']:
            sample_targets = query_result['inference_targets']['per_sample_targets'][0]['targets']
            for target_name, target_info in sample_targets.items():
                key = f"{target_name}_sample0"
                if key not in target_lengths:
                    target_lengths[key] = []
                target_lengths[key].append(len(target_info['target']))

    # 打印平均长度
    print("\nPer-query targets:")
    for target_name in ['gt_action_toolcall', 'gt_thought_action_toolcall', 'gt_full_toolcall',
                       'baseline_action', 'gt_thought_action', 'gt_full_action']:
        if target_name in target_lengths:
            avg_len = sum(target_lengths[target_name]) / len(target_lengths[target_name])
            print(f"  {target_name:30s}: {avg_len:6.1f} 字符")

    print("\nPer-sample targets (sample 0):")
    for target_name in ['baseline_toolcall', 'model_action_toolcall',
                       'model_thought_action_toolcall', 'model_thought_action']:
        key = f"{target_name}_sample0"
        if key in target_lengths:
            avg_len = sum(target_lengths[key]) / len(target_lengths[key])
            print(f"  {target_name:30s}: {avg_len:6.1f} 字符")


def main():
    """运行所有示例"""
    try:
        example_1_basic_access()
        example_2_compare_targets()
        example_3_count_inference_requirements()
        example_4_extract_specific_targets()
        example_5_group_by_measure_part()
        example_6_calculate_target_lengths()

        print("\n" + "=" * 80)
        print("所有示例运行完成！")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"错误: 文件不存在 - {e}")
        print("请确保已经运行 add_targets_to_infogain_v2.py 生成带 targets 的文件")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
