#!/usr/bin/env python3
"""
分析多次采样结果的脚本

功能：
1. 计算每个回复是否正确
2. 计算 pass@1, pass@k 和平均 Acc
3. 用 ground_truth 构造 <tool_call> 格式并计算 InfoGain
"""

import os
import json
import argparse
import importlib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "agentnetbench"))
from eval import ActionEvaluator


class SamplingResultAnalyzer:
    """分析多次采样结果的类"""

    def __init__(self, data_dir: str, agent_type: str = "qwen3vl"):
        self.data_dir = Path(data_dir)
        self.agent_type = agent_type
        self.evaluator = ActionEvaluator()
        self.agent_instance = self._load_agent()

    def _load_agent(self):
        """加载 agent 用于解析响应"""
        try:
            module_name = f"agent.{self.agent_type}"
            sys.path.insert(0, str(Path(__file__).parent.parent / "agentnetbench"))
            agent_module = importlib.import_module(module_name)

            for attr_name in dir(agent_module):
                attr = getattr(agent_module, attr_name)
                if isinstance(attr, type) and attr_name.lower() == self.agent_type.lower():
                    if self.agent_type.lower() == "qwen3vl":
                        class DummyParsingAgent(attr):
                            def __init__(self, model, client=None, **kwargs):
                                self.model = model
                                self.client = None
                                self.image_dir = str(Path(__file__).parent.parent.parent / "data/data/AgentNetBench/test_data/images")
                                self.image_cache = {}
                                self.message_cache = {}
                                self.history_n = 3
                                self.history_responses = []
                                self.history_images = []

                            def load_image(self, image_file, image_dir):
                                image_path = os.path.join(image_dir, image_file)
                                with open(image_path, "rb") as f:
                                    return f.read()

                        return DummyParsingAgent(model="dummy")
                    else:
                        return attr(model="dummy", client=None)

            print(f"Warning: Could not find agent class for {self.agent_type}")
            return None
        except Exception as e:
            print(f"Warning: Could not load agent: {e}")
            return None

    def load_trajectory(self, task_id: str):
        """加载某个task的trajectory"""
        trajectory_path = self.data_dir / f"{task_id}.json"
        if not trajectory_path.exists():
            return None

        with open(trajectory_path, "r") as f:
            return json.load(f)

    def evaluate_single_response(self, response: str, ground_truth_actions: list,
                                 trajectory: dict, step_num: int):
        """评估单个响应是否正确"""
        if not self.agent_instance:
            return None, None

        try:
            # 解析响应
            parsed_action = self.agent_instance.parse_response(response, trajectory, step_num)
            predicted_actions = self.agent_instance.extract_actions(parsed_action)

            # 评估
            eval_item = {
                "ground_truth_actions": ground_truth_actions,
                "predicted_actions": predicted_actions
            }
            eval_scores = self.evaluator.evaluate_action(eval_item)

            return eval_scores, predicted_actions
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return None, None

    def analyze_sampling_file(self, sampling_file: Path):
        """分析一个采样结果文件"""
        print(f"\n{'='*70}")
        print(f"Analyzing: {sampling_file.name}")
        print(f"{'='*70}\n")

        # 读取采样结果
        with open(sampling_file, "r") as f:
            sampling_results = [json.loads(line) for line in f]

        # 统计指标
        total_queries = len(sampling_results)
        num_samples = len(sampling_results[0]["samples"]) if sampling_results else 0

        # 为每个query收集评估结果
        all_eval_results = []
        pass_at_1_count = 0
        pass_at_k_count = 0

        for query_result in tqdm(sampling_results, desc="Evaluating samples"):
            query_id = query_result["query_id"]

            # 解析 query_id 获取 task_id 和 step_num
            # 假设 query_id 格式为 "task_id-step_num" 或其他格式
            if "-" in query_id:
                task_id, step_num_str = query_id.rsplit("-", 1)
                try:
                    step_num = int(step_num_str)
                except:
                    print(f"Warning: Cannot parse step_num from {query_id}")
                    continue
            else:
                print(f"Warning: Invalid query_id format: {query_id}")
                continue

            # 加载 trajectory
            trajectory = self.load_trajectory(task_id)
            if not trajectory:
                print(f"Warning: Cannot load trajectory for {task_id}")
                continue

            if step_num >= len(trajectory["steps"]):
                print(f"Warning: step_num {step_num} out of range for {task_id}")
                continue

            step_data = trajectory["steps"][step_num]
            ground_truth_actions = step_data.get("ground_truth_actions", [])

            # 评估所有samples
            sample_results = []
            for sample in query_result["samples"]:
                response = sample["response"]
                eval_scores, predicted_actions = self.evaluate_single_response(
                    response, ground_truth_actions, trajectory, step_num
                )

                sample_results.append({
                    "sample_id": sample["sample_id"],
                    "eval_scores": eval_scores,
                    "predicted_actions": predicted_actions,
                    "is_correct": eval_scores["total"] == 1.0 if eval_scores else False
                })

            # 计算 pass@1 和 pass@k
            is_correct_list = [r["is_correct"] for r in sample_results]
            pass_at_1 = is_correct_list[0] if len(is_correct_list) > 0 else False
            pass_at_k = any(is_correct_list)

            if pass_at_1:
                pass_at_1_count += 1
            if pass_at_k:
                pass_at_k_count += 1

            all_eval_results.append({
                "query_id": query_id,
                "task_id": task_id,
                "step_num": step_num,
                "ground_truth_actions": ground_truth_actions,
                "sample_results": sample_results,
                "pass_at_1": pass_at_1,
                "pass_at_k": pass_at_k,
                "avg_score": np.mean([r["eval_scores"]["total"] if r["eval_scores"] else 0.0
                                      for r in sample_results])
            })

        # 计算总体指标
        pass_at_1_acc = pass_at_1_count / total_queries if total_queries > 0 else 0.0
        pass_at_k_acc = pass_at_k_count / total_queries if total_queries > 0 else 0.0

        # 计算所有样本的平均准确率
        all_scores = []
        for result in all_eval_results:
            for sample_result in result["sample_results"]:
                if sample_result["eval_scores"]:
                    all_scores.append(sample_result["eval_scores"]["total"])

        avg_acc = np.mean(all_scores) if all_scores else 0.0

        metrics = {
            "total_queries": total_queries,
            "num_samples_per_query": num_samples,
            "pass@1_accuracy": pass_at_1_acc,
            f"pass@{num_samples}_accuracy": pass_at_k_acc,
            "average_accuracy_all_samples": avg_acc,
            "total_samples_evaluated": len(all_scores)
        }

        # 打印结果
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"Total queries: {total_queries}")
        print(f"Samples per query: {num_samples}")
        print(f"Total samples evaluated: {len(all_scores)}")
        print(f"\nAccuracy Metrics:")
        print(f"  Pass@1:  {pass_at_1_acc*100:.2f}% ({pass_at_1_count}/{total_queries})")
        print(f"  Pass@{num_samples}: {pass_at_k_acc*100:.2f}% ({pass_at_k_count}/{total_queries})")
        print(f"  Average (all samples): {avg_acc*100:.2f}%")
        print(f"{'='*70}\n")

        return {
            "metrics": metrics,
            "detailed_results": all_eval_results
        }

    def save_results(self, results: dict, output_file: Path):
        """保存分析结果"""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 保存详细结果
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 保存简化的metrics
        metrics_file = output_file.parent / f"{output_file.stem}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(results["metrics"], f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_file}")
        print(f"Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-sampling inference results")
    parser.add_argument("--sampling_file", type=str, required=True,
                       help="Path to the sampling results file (.jsonl)")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to the test data directory")
    parser.add_argument("--agent_type", type=str, default="qwen3vl",
                       help="Agent type (qwen3vl, qwen25vl, etc.)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for analysis results")

    args = parser.parse_args()

    # 创建分析器
    analyzer = SamplingResultAnalyzer(args.data_dir, args.agent_type)

    # 分析文件
    sampling_file = Path(args.sampling_file)
    results = analyzer.analyze_sampling_file(sampling_file)

    # 确定输出路径
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = sampling_file.parent / "analysis"

    output_file = output_dir / f"{sampling_file.stem}_analysis.json"

    # 保存结果
    analyzer.save_results(results, output_file)


if __name__ == "__main__":
    main()
