#!/usr/bin/env python3
"""
为 InfoGain 结果文件添加推理时使用的 target 信息

按照 calculate_infogain_multiGPU.py 中的方式组织 target，
并区分 per-query 和 per-sample 的 target。
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_thought_action_toolcall(response: str) -> Tuple[str, str, str]:
    """
    从模型响应中提取 Thought、Action 和 tool_call

    与 calculate_infogain_multiGPU.py 中的实现保持一致
    """
    # 提取 Thought
    thought_match = re.search(r'Thought:(.*?)(?=Action:|<tool_call>|$)', response, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    # 提取 Action
    action_match = re.search(r'Action:(.*?)(?=<tool_call>|$)', response, re.DOTALL)
    action = action_match.group(1).strip() if action_match else ""

    # 提取 tool_call
    tool_call_match = re.search(r'<tool_call>.*?</tool_call>', response, re.DOTALL)
    tool_call = tool_call_match.group(0) if tool_call_match else ""

    # 格式化输出
    thought_text = f"Thought: {thought}" if thought else ""
    action_text = f"Action: {action}" if action else ""

    return thought_text, action_text, tool_call


def construct_targets_for_query(query_data: Dict, sampling_result: Dict) -> Dict:
    """
    为一个 query 构建所有的 targets

    返回:
    {
        "per_query_targets": {...},  # 每个 query 只需推理一次的 targets
        "per_sample_targets": [...]  # 每个 sample 都需要的 targets
    }
    """
    # 获取 ground truth 数据
    step_data = query_data["data"]
    gt_tool_call = query_data["ground_truth_tool_call"]

    # 从 ground truth 获取各部分
    gt_observation = step_data.get("inner_monologue", {}).get("observation", "")
    gt_thought = step_data.get("inner_monologue", {}).get("thought", "")
    gt_low_level_instruction = step_data.get("inner_monologue", {}).get("low_level_instruction", "")

    # 格式化 ground truth 部分
    gt_observation_text = f"Observation: {gt_observation}" if gt_observation else ""
    gt_thought_text = f"Thought: {gt_thought}" if gt_thought else ""
    gt_action_text = f"Action: {gt_low_level_instruction}" if gt_low_level_instruction else ""

    # ============ Per-Query Targets (只需推理一次) ============
    per_query_targets = {}

    # 1. gt_action_toolcall
    target = ""
    if gt_action_text:
        target += gt_action_text + "\n"
    target += gt_tool_call
    per_query_targets["gt_action_toolcall"] = {
        "target": target,
        "measure_part": "tool_call",
        "description": "GT Action + <tool_call>"
    }

    # 2. gt_thought_action_toolcall
    target = ""
    if gt_thought_text:
        target += gt_thought_text + "\n"
    if gt_action_text:
        target += gt_action_text + "\n"
    target += gt_tool_call
    per_query_targets["gt_thought_action_toolcall"] = {
        "target": target,
        "measure_part": "tool_call",
        "description": "GT Thought + GT Action + <tool_call>"
    }

    # 3. gt_full_toolcall
    target = ""
    if gt_observation_text:
        target += gt_observation_text + "\n"
    if gt_thought_text:
        target += gt_thought_text + "\n"
    if gt_action_text:
        target += gt_action_text + "\n"
    target += gt_tool_call
    per_query_targets["gt_full_toolcall"] = {
        "target": target,
        "measure_part": "tool_call",
        "description": "GT Observation + GT Thought + GT Action + <tool_call>"
    }

    # 对于预测 Action 的 GT targets
    if gt_action_text:
        # 4. baseline_action
        per_query_targets["baseline_action"] = {
            "target": gt_action_text,
            "measure_part": "action",
            "description": "GT Action only (baseline)"
        }

        # 5. gt_thought_action
        target = ""
        if gt_thought_text:
            target += gt_thought_text + "\n"
        target += gt_action_text
        per_query_targets["gt_thought_action"] = {
            "target": target,
            "measure_part": "action",
            "description": "GT Thought + GT Action"
        }

        # 6. gt_full_action
        target = ""
        if gt_observation_text:
            target += gt_observation_text + "\n"
        if gt_thought_text:
            target += gt_thought_text + "\n"
        target += gt_action_text
        per_query_targets["gt_full_action"] = {
            "target": target,
            "measure_part": "action",
            "description": "GT Observation + GT Thought + GT Action"
        }

    # ============ Per-Sample Targets (每个 sample 都需要) ============
    per_sample_targets = []

    for sample_idx, sample in enumerate(sampling_result["samples"]):
        # 从模型 sampling 结果中提取 Thought 和 Action
        response = "Thought:" + sample["response"]
        model_thought, model_action, _ = extract_thought_action_toolcall(response)

        sample_targets = {
            "sample_idx": sample_idx,
            "targets": {}
        }

        # 1. baseline_toolcall
        sample_targets["targets"]["baseline_toolcall"] = {
            "target": gt_tool_call,
            "measure_part": "tool_call",
            "description": "<tool_call> only (baseline)"
        }

        # 2. model_action_toolcall
        target = ""
        if model_action:
            target += model_action + "\n"
        target += gt_tool_call
        sample_targets["targets"]["model_action_toolcall"] = {
            "target": target,
            "measure_part": "tool_call",
            "description": "Model Action + <tool_call>"
        }

        # 3. model_thought_action_toolcall
        target = ""
        if model_thought:
            target += model_thought + "\n"
        if model_action:
            target += model_action + "\n"
        target += gt_tool_call
        sample_targets["targets"]["model_thought_action_toolcall"] = {
            "target": target,
            "measure_part": "tool_call",
            "description": "Model Thought + Model Action + <tool_call>"
        }

        # 4. model_thought_action (用于预测 Action)
        if gt_action_text:
            target = ""
            if model_thought:
                target += model_thought + "\n"
            target += gt_action_text
            sample_targets["targets"]["model_thought_action"] = {
                "target": target,
                "measure_part": "action",
                "description": "Model Thought + GT Action"
            }

        per_sample_targets.append(sample_targets)

    return {
        "per_query_targets": per_query_targets,
        "per_sample_targets": per_sample_targets
    }


def add_targets_to_infogain_file(
    infogain_file: Path,
    sampling_file: Path,
    output_file: Path
):
    """
    读取 InfoGain 结果文件和原始 sampling 文件，
    添加 target 信息并保存
    """
    print("=" * 80)
    print("为 InfoGain 结果添加 Target 信息")
    print("=" * 80)

    # 1. 读取 InfoGain 结果
    print(f"\n1. 读取 InfoGain 结果文件...")
    print(f"   文件: {infogain_file}")
    with open(infogain_file, 'r') as f:
        infogain_data = json.load(f)
    print(f"   ✓ 查询数: {len(infogain_data['detailed_results'])}")

    # 2. 读取原始 sampling 结果
    print(f"\n2. 读取原始 sampling 文件...")
    print(f"   文件: {sampling_file}")
    with open(sampling_file, 'r') as f:
        sampling_data = [json.loads(line) for line in f]
    print(f"   ✓ 查询数: {len(sampling_data)}")

    # 3. 创建 query_id 到 sampling_result 的映射
    print(f"\n3. 创建查询映射...")
    sampling_dict = {}
    for result in sampling_data:
        query_id = result["query_id"]
        sampling_dict[query_id] = result
    print(f"   ✓ 映射 {len(sampling_dict)} 个查询")

    # 4. 为每个 query 添加 target 信息
    print(f"\n4. 构建 target 信息...")
    added_count = 0
    missing_count = 0

    for query_result in infogain_data['detailed_results']:
        query_id = query_result['query_id']

        if query_id not in sampling_dict:
            print(f"   警告: 找不到 query_id={query_id} 的 sampling 结果")
            missing_count += 1
            continue

        sampling_result = sampling_dict[query_id]

        # 构建 targets
        targets_info = construct_targets_for_query(query_result, sampling_result)

        # 添加到 query_result
        query_result['inference_targets'] = targets_info
        added_count += 1

    print(f"   ✓ 成功添加: {added_count} 个查询")
    if missing_count > 0:
        print(f"   ✗ 缺失: {missing_count} 个查询")

    # 5. 添加元数据说明
    if 'metadata' not in infogain_data:
        infogain_data['metadata'] = {}

    infogain_data['metadata']['inference_targets_description'] = {
        "per_query_targets": "每个 query 只需推理一次的 targets (所有 samples 共享)",
        "per_sample_targets": "每个 sample 都需要推理的 targets (因为模型输出不同)",
        "target_types": {
            "gt_action_toolcall": "GT Action + <tool_call>",
            "gt_thought_action_toolcall": "GT Thought + GT Action + <tool_call>",
            "gt_full_toolcall": "GT Observation + GT Thought + GT Action + <tool_call>",
            "baseline_action": "GT Action only (baseline for action prediction)",
            "gt_thought_action": "GT Thought + GT Action",
            "gt_full_action": "GT Observation + GT Thought + GT Action",
            "baseline_toolcall": "<tool_call> only (baseline for tool_call prediction)",
            "model_action_toolcall": "Model Action + <tool_call>",
            "model_thought_action_toolcall": "Model Thought + Model Action + <tool_call>",
            "model_thought_action": "Model Thought + GT Action"
        },
        "measure_parts": {
            "tool_call": "测量 <tool_call>...</tool_call> 部分的 perplexity",
            "action": "测量 Action: ... 部分的 perplexity"
        }
    }

    # 6. 保存结果
    print(f"\n5. 保存结果...")
    print(f"   输出文件: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(infogain_data, f, indent=2, ensure_ascii=False)
    print(f"   ✓ 保存成功")

    # 7. 显示示例
    print(f"\n6. 示例 (第一个 query):")
    print(f"=" * 80)
    first_query = infogain_data['detailed_results'][0]
    print(f"Query ID: {first_query['query_id']}")

    print(f"\nPer-Query Targets (所有 samples 共享):")
    for target_type, target_info in first_query['inference_targets']['per_query_targets'].items():
        print(f"  - {target_type}:")
        print(f"    描述: {target_info['description']}")
        print(f"    测量: {target_info['measure_part']}")
        target_preview = target_info['target'][:100] + "..." if len(target_info['target']) > 100 else target_info['target']
        print(f"    Target 预览: {target_preview}")

    print(f"\nPer-Sample Targets (每个 sample 不同):")
    print(f"  Sample 0:")
    for target_type, target_info in first_query['inference_targets']['per_sample_targets'][0]['targets'].items():
        print(f"    - {target_type}:")
        print(f"      描述: {target_info['description']}")
        target_preview = target_info['target'][:80] + "..." if len(target_info['target']) > 80 else target_info['target']
        print(f"      Target: {target_preview}")

    print(f"\n" + "=" * 80)
    print(f"✓ 完成！Target 信息已添加到 InfoGain 结果文件")
    print(f"=" * 80)


def main():
    # 文件路径
    infogain_file = Path("../guir1/outputs/Qwen3-VL-4B-Instruct/analysis/agentnetbench_test_qwen3vl_sampling16_Thought:_infogain_multiGPU.json")
    sampling_file = Path("../guir1/outputs/Qwen3-VL-4B-Instruct/agentnetbench_test_qwen3vl_sampling16_Thought:.jsonl")
    output_file = Path("../guir1/outputs/Qwen3-VL-4B-Instruct/analysis/agentnetbench_test_qwen3vl_sampling16_Thought:_infogain_multiGPU_with_targets.json")

    # 检查文件是否存在
    if not infogain_file.exists():
        print(f"✗ InfoGain 文件不存在: {infogain_file}")
        return 1

    if not sampling_file.exists():
        print(f"✗ Sampling 文件不存在: {sampling_file}")
        return 1

    # 添加 target 信息
    add_targets_to_infogain_file(infogain_file, sampling_file, output_file)

    return 0


if __name__ == "__main__":
    exit(main())
