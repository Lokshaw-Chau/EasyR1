#!/usr/bin/env python3
"""
性能对比测试：批处理 vs 顺序处理
"""

import time
import torch
from pathlib import Path
import json
from calculate_infogain import InfoGainCalculator


def benchmark_infogain(sampling_file: str, model_path: str, data_dir: str, use_batch: bool = True):
    """
    测试 InfoGain 计算性能

    Args:
        sampling_file: 采样结果文件
        model_path: 模型路径
        data_dir: 数据目录
        use_batch: 是否使用批处理

    Returns:
        elapsed_time: 耗时（秒）
    """
    print(f"\n{'='*70}")
    print(f"模式: {'批处理' if use_batch else '顺序处理'}")
    print(f"{'='*70}")

    # 创建计算器
    calculator = InfoGainCalculator(model_path, data_dir, device="cuda")

    # 读取采样结果（只取前几个 query 用于测试）
    with open(sampling_file, "r") as f:
        sampling_results = [json.loads(line) for line in f]

    # 限制测试数量
    num_test_queries = min(5, len(sampling_results))
    test_results = sampling_results[:num_test_queries]

    print(f"测试 {num_test_queries} 个 queries...")
    print(f"每个 query 约 {len(test_results[0]['samples'])} 个 samples")

    # 保存到临时文件
    temp_file = Path(sampling_file).parent / f"temp_test_{num_test_queries}.jsonl"
    with open(temp_file, "w") as f:
        for result in test_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 计时
    torch.cuda.synchronize()
    start_time = time.time()

    # 运行 InfoGain 计算
    calculator.analyze_sampling_file(temp_file, output_file=None, use_batch=use_batch)

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    # 清理临时文件
    temp_file.unlink()

    print(f"\n✓ 完成！耗时: {elapsed_time:.2f} 秒")

    return elapsed_time


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark batch vs sequential processing")
    parser.add_argument("--sampling_file", type=str, required=True,
                       help="Path to sampling results file")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to test data directory")
    parser.add_argument("--num_queries", type=int, default=5,
                       help="Number of queries to test (default: 5)")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("InfoGain 计算性能对比测试")
    print("="*70)

    # 测试批处理模式
    print("\n🚀 测试 1: 批处理模式（加速）")
    batch_time = benchmark_infogain(
        args.sampling_file,
        args.model_path,
        args.data_dir,
        use_batch=True
    )

    # 测试顺序模式
    print("\n🐢 测试 2: 顺序处理模式（基线）")
    sequential_time = benchmark_infogain(
        args.sampling_file,
        args.model_path,
        args.data_dir,
        use_batch=False
    )

    # 性能对比
    speedup = sequential_time / batch_time if batch_time > 0 else 0

    print("\n" + "="*70)
    print("📊 性能对比结果")
    print("="*70)
    print(f"顺序处理耗时: {sequential_time:.2f} 秒")
    print(f"批处理耗时:   {batch_time:.2f} 秒")
    print(f"加速比:       {speedup:.2f}x")
    print("="*70)

    if speedup > 1:
        print(f"\n✅ 批处理模式比顺序模式快 {speedup:.2f} 倍！")
    elif speedup < 1:
        print(f"\n⚠️  批处理模式比顺序模式慢 {1/speedup:.2f} 倍")
        print("    可能原因：batch size 太小，overhead 较大")
    else:
        print("\n⚠️  性能相当，可能测试样本太少")

    print()


if __name__ == "__main__":
    main()
