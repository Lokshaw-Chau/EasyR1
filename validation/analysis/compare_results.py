#!/usr/bin/env python3
"""
比较不同采样结果的脚本

用于对比不同 prefix 或不同模型的性能
"""

import json
import argparse
from pathlib import Path
import pandas as pd
from tabulate import tabulate


def load_metrics(file_path: Path):
    """加载 metrics 文件"""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_config_from_filename(filename: str):
    """从文件名中提取配置信息"""
    # 例如: agentnetbench_test_qwen3vl_sampling16_Thought:_metrics.json
    parts = filename.replace("_metrics.json", "").split("_")

    config = {
        "filename": filename,
        "full_name": filename.replace("_metrics.json", "")
    }

    # 提取采样数
    for i, part in enumerate(parts):
        if part.startswith("sampling"):
            config["num_samples"] = part.replace("sampling", "")
            if i + 1 < len(parts):
                config["prefix"] = parts[i + 1]
            break

    return config


def compare_accuracy(metrics_files: list):
    """比较准确率指标"""
    print(f"\n{'='*80}")
    print("ACCURACY COMPARISON")
    print(f"{'='*80}\n")

    data = []
    for file_path in metrics_files:
        metrics = load_metrics(file_path)
        config = extract_config_from_filename(file_path.name)

        # 提取关键指标
        pass_at_1 = metrics.get("pass@1_accuracy", 0.0) * 100
        pass_at_k_key = [k for k in metrics.keys() if k.startswith("pass@") and k != "pass@1_accuracy"]
        pass_at_k = metrics.get(pass_at_k_key[0], 0.0) * 100 if pass_at_k_key else 0.0
        avg_acc = metrics.get("average_accuracy_all_samples", 0.0) * 100

        data.append({
            "Configuration": config.get("prefix", "Unknown"),
            "Samples": config.get("num_samples", "?"),
            "Pass@1 (%)": f"{pass_at_1:.2f}",
            "Pass@k (%)": f"{pass_at_k:.2f}",
            "Avg Acc (%)": f"{avg_acc:.2f}",
            "Queries": metrics.get("total_queries", 0),
            "Total Samples": metrics.get("total_samples_evaluated", 0)
        })

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 打印表格
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    print()

    # 打印最佳配置
    if len(data) > 0:
        best_pass1_idx = max(range(len(data)),
                            key=lambda i: float(data[i]["Pass@1 (%)"].rstrip('%')))
        best_passk_idx = max(range(len(data)),
                            key=lambda i: float(data[i]["Pass@k (%)"].rstrip('%')))
        best_avg_idx = max(range(len(data)),
                          key=lambda i: float(data[i]["Avg Acc (%)"].rstrip('%')))

        print("Best Configurations:")
        print(f"  Pass@1:  {data[best_pass1_idx]['Configuration']} ({data[best_pass1_idx]['Pass@1 (%)']}%)")
        print(f"  Pass@k:  {data[best_passk_idx]['Configuration']} ({data[best_passk_idx]['Pass@k (%)']}%)")
        print(f"  Avg Acc: {data[best_avg_idx]['Configuration']} ({data[best_avg_idx]['Avg Acc (%)']}%)")
        print()


def compare_infogain(infogain_files: list):
    """比较 InfoGain 指标"""
    print(f"\n{'='*80}")
    print("INFOGAIN COMPARISON")
    print(f"{'='*80}\n")

    data = []
    for file_path in infogain_files:
        with open(file_path, "r") as f:
            results = json.load(f)

        config = extract_config_from_filename(file_path.name.replace("_infogain", "_metrics"))
        summary = results.get("summary", {})

        data.append({
            "Configuration": config.get("prefix", "Unknown"),
            "Avg InfoGain": f"{summary.get('avg_infogain', 0.0):.4f}",
            "Median InfoGain": f"{summary.get('median_infogain', 0.0):.4f}",
            "PPL Baseline": f"{summary.get('avg_perplexity_baseline', 0.0):.4f}",
            "PPL w/ Thinking": f"{summary.get('avg_perplexity_with_thinking', 0.0):.4f}",
            "Positive IG (%)": f"{summary.get('percentage_positive_infogain', 0.0):.2f}",
            "Samples": summary.get('total_samples', 0)
        })

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 打印表格
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    print()

    # 打印最佳配置
    if len(data) > 0:
        best_ig_idx = max(range(len(data)),
                         key=lambda i: float(data[i]["Avg InfoGain"]))
        best_positive_idx = max(range(len(data)),
                               key=lambda i: float(data[i]["Positive IG (%)"]))

        print("Best Configurations:")
        print(f"  Highest InfoGain: {data[best_ig_idx]['Configuration']} ({data[best_ig_idx]['Avg InfoGain']})")
        print(f"  Most Positive IG: {data[best_positive_idx]['Configuration']} ({data[best_positive_idx]['Positive IG (%)']}%)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Compare multiple sampling results")
    parser.add_argument("files", nargs="+", help="Metrics or InfoGain JSON files to compare")
    parser.add_argument("--type", choices=["auto", "accuracy", "infogain"], default="auto",
                       help="Type of comparison (default: auto-detect)")

    args = parser.parse_args()

    # 分类文件
    metrics_files = []
    infogain_files = []

    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        if "infogain" in path.name:
            infogain_files.append(path)
        elif "metrics" in path.name:
            metrics_files.append(path)
        else:
            # 尝试自动检测
            with open(path, "r") as f:
                data = json.load(f)
                if "summary" in data and "avg_infogain" in data.get("summary", {}):
                    infogain_files.append(path)
                elif "pass@1_accuracy" in data or "metrics" in data:
                    metrics_files.append(path)

    # 执行比较
    if args.type in ["auto", "accuracy"] and metrics_files:
        compare_accuracy(metrics_files)

    if args.type in ["auto", "infogain"] and infogain_files:
        compare_infogain(infogain_files)

    if not metrics_files and not infogain_files:
        print("Error: No valid metrics or infogain files found")
        return


if __name__ == "__main__":
    main()
