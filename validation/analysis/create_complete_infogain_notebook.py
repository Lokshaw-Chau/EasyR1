#!/usr/bin/env python3
"""
创建完整的 InfoGain 分析 Jupyter Notebook (包含所有分析)
"""

import json

# 创建完整的 notebook 结构
cells = []

# ============================================================================
# 标题和介绍
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# InfoGain 完整分析\n",
        "\n",
        "分析模型推理链（Thought + Action）对预测 tool_call 和 Action 的影响。\n",
        "\n",
        "## 分析目标\n",
        "\n",
        "1. **推理链价值评估**: 统计不同推理链对 tool_call 和 Action 的帮助\n",
        "2. **模型 vs GT CoT 对比**: 评估模型生成的 CoT 质量\n",
        "3. **InfoGain 作为 RL Reward**: 探讨用 InfoGain 筛选 CoT 的可行性\n",
        "\n",
        "## InfoGain 指标说明\n",
        "\n",
        "### 对于预测 tool_call:\n",
        "- `model_action_infogain`: 模型 Action 的信息增益\n",
        "- `model_thought_action_infogain`: 模型 Thought+Action 的信息增益\n",
        "- `gt_action_infogain`: GT Action 的信息增益\n",
        "- `gt_thought_action_infogain`: GT Thought+Action 的信息增益\n",
        "- `gt_full_infogain`: GT Full (Observation+Thought+Action) 的信息增益\n",
        "\n",
        "### 对于预测 Action:\n",
        "- `model_thought_to_action_infogain`: 模型 Thought 对 Action 的信息增益\n",
        "- `gt_thought_to_action_infogain`: GT Thought 对 Action 的信息增益\n",
        "- `gt_full_to_action_infogain`: GT Full 对 Action 的信息增益"
    ]
})

# ============================================================================
# 导入库
# ============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import json\\n",
        "import numpy as np\\n",
        "import pandas as pd\\n",
        "import matplotlib.pyplot as plt\\n",
        "import seaborn as sns\\n",
        "from pathlib import Path\\n",
        "from typing import Dict, List, Tuple\\n",
        "import warnings\\n",
        "warnings.filterwarnings('ignore')\\n",
        "\\n",
        "# 设置中文字体\\n",
        "plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']\\n",
        "plt.rcParams['axes.unicode_minus'] = False\\n",
        "\\n",
        "# 设置显示选项\\n",
        "pd.set_option('display.max_columns', None)\\n",
        "pd.set_option('display.max_rows', 100)\\n",
        "pd.set_option('display.width', 1000)\\n",
        "\\n",
        "# 设置 seaborn 样式\\n",
        "sns.set_style('whitegrid')"
    ]
})

# ============================================================================
# Section 1: 数据加载和预处理
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. 数据加载和预处理"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 读取 InfoGain 结果\\n",
        "result_file = Path(\\\"../guir1/outputs/Qwen3-VL-4B-Instruct/analysis/sampled_agentnetbench_test_infogain_multiGPU.json\\\")\\n",
        "\\n",
        "with open(result_file, 'r') as f:\\n",
        "    data = json.load(f)\\n",
        "\\n",
        "print(f\\\"总查询数: {data['summary']['total_queries']}\\\")\\n",
        "print(f\\\"总样本数: {data['summary']['total_samples']}\\\")\\n",
        "print(f\\\"详细结果数: {len(data['detailed_results'])}\\\")"
    ]
})

# GT 填充函数
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def fill_gt_perplexities(sample_infogains: List[Dict]) -> List[Dict]:\\n",
        "    \\\"\\\"\\\"\\n",
        "    填充 GT 相关的 perplexity 值\\n",
        "    \\n",
        "    Ground truth 相关的对照组只在 sample_idx=0 时计算，\\n",
        "    需要将这些值复制到其他样本中。\\n",
        "    \\\"\\\"\\\"\\n",
        "    # 找到 sample_idx=0 的 GT 值\\n",
        "    gt_values = None\\n",
        "    for sample in sample_infogains:\\n",
        "        if sample['sample_id'] == 0:\\n",
        "            gt_values = {\\n",
        "                'perplexity_gt_action_toolcall': sample.get('perplexity_gt_action_toolcall'),\\n",
        "                'perplexity_gt_thought_action_toolcall': sample.get('perplexity_gt_thought_action_toolcall'),\\n",
        "                'perplexity_gt_full_toolcall': sample.get('perplexity_gt_full_toolcall'),\\n",
        "                'perplexity_baseline_action': sample.get('perplexity_baseline_action'),\\n",
        "                'perplexity_gt_thought_action': sample.get('perplexity_gt_thought_action'),\\n",
        "                'perplexity_gt_full_action': sample.get('perplexity_gt_full_action'),\\n",
        "                'gt_action_infogain': sample.get('gt_action_infogain'),\\n",
        "                'gt_thought_action_infogain': sample.get('gt_thought_action_infogain'),\\n",
        "                'gt_full_infogain': sample.get('gt_full_infogain'),\\n",
        "                'gt_thought_to_action_infogain': sample.get('gt_thought_to_action_infogain'),\\n",
        "                'gt_full_to_action_infogain': sample.get('gt_full_to_action_infogain'),\\n",
        "            }\\n",
        "            break\\n",
        "    \\n",
        "    if gt_values is None:\\n",
        "        print(\\\"警告: 未找到 sample_idx=0 的 GT 值\\\")\\n",
        "        return sample_infogains\\n",
        "    \\n",
        "    # 填充其他样本的 GT 值\\n",
        "    filled_samples = []\\n",
        "    for sample in sample_infogains:\\n",
        "        sample_copy = sample.copy()\\n",
        "        for key, value in gt_values.items():\\n",
        "            if sample_copy.get(key) is None or sample_copy.get(key) == float('inf'):\\n",
        "                sample_copy[key] = value\\n",
        "        filled_samples.append(sample_copy)\\n",
        "    \\n",
        "    return filled_samples\\n",
        "\\n",
        "# 填充所有 query 的 GT 值\\n",
        "for query_result in data['detailed_results']:\\n",
        "    query_result['sample_infogains'] = fill_gt_perplexities(query_result['sample_infogains'])\\n",
        "\\n",
        "print(\\\"✓ GT perplexity 值已填充\\\")"
    ]
})

# DataFrame 转换
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 将数据转换为 DataFrame 以便分析\\n",
        "rows = []\\n",
        "for query_result in data['detailed_results']:\\n",
        "    query_id = query_result['query_id']\\n",
        "    for sample in query_result['sample_infogains']:\\n",
        "        row = {'query_id': query_id}\\n",
        "        row.update(sample)\\n",
        "        rows.append(row)\\n",
        "\\n",
        "df = pd.DataFrame(rows)\\n",
        "\\n",
        "# 过滤掉无效样本（perplexity 为 inf 或 None）\\n",
        "df = df[df['perplexity_baseline_toolcall'] != float('inf')]\\n",
        "df = df[df['perplexity_baseline_toolcall'].notna()]\\n",
        "\\n",
        "print(f\\\"有效样本数: {len(df)}\\\")\\n",
        "print(f\\\"\\\\n数据列:\\\")\\n",
        "print(df.columns.tolist())"
    ]
})

# 统计摘要
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 查看数据摘要\\n",
        "print(\\\"=\\\" * 80)\\n",
        "print(\\\"InfoGain 统计摘要\\\")\\n",
        "print(\\\"=\\\" * 80)\\n",
        "\\n",
        "infogain_cols = [\\n",
        "    'model_action_infogain',\\n",
        "    'model_thought_action_infogain',\\n",
        "    'gt_action_infogain',\\n",
        "    'gt_thought_action_infogain',\\n",
        "    'gt_full_infogain',\\n",
        "    'model_thought_to_action_infogain',\\n",
        "    'gt_thought_to_action_infogain',\\n",
        "    'gt_full_to_action_infogain',\\n",
        "]\\n",
        "\\n",
        "df[infogain_cols].describe()"
    ]
})

# ============================================================================
# Section 2: 推理链价值分析 - tool_call
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. 推理链价值分析\\n",
        "\\n",
        "### 2.1 对于 tool_call 的推理链价值"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def classify_toolcall_benefit(row) -> str:\\n",
        "    \\\"\\\"\\\"根据不同推理链的 InfoGain 分类样本受益类型\\\"\\\"\\\"\\n",
        "    threshold = 0.05\\n",
        "    \\n",
        "    model_action_benefit = row['model_action_infogain'] > threshold\\n",
        "    model_thought_action_benefit = row['model_thought_action_infogain'] > threshold\\n",
        "    gt_action_benefit = row['gt_action_infogain'] > threshold\\n",
        "    gt_thought_action_benefit = row['gt_thought_action_infogain'] > threshold\\n",
        "    gt_full_benefit = row['gt_full_infogain'] > threshold\\n",
        "    \\n",
        "    benefits = [model_action_benefit, model_thought_action_benefit, gt_action_benefit, gt_thought_action_benefit, gt_full_benefit]\\n",
        "    benefit_count = sum(benefits)\\n",
        "    \\n",
        "    if benefit_count == 0:\\n",
        "        return \\\"无增益\\\"\\n",
        "    elif benefit_count == 1:\\n",
        "        if model_action_benefit:\\n",
        "            return \\\"仅模型Action有益\\\"\\n",
        "        elif model_thought_action_benefit:\\n",
        "            return \\\"仅模型Thought+Action有益\\\"\\n",
        "        elif gt_action_benefit:\\n",
        "            return \\\"仅GT Action有益\\\"\\n",
        "        elif gt_thought_action_benefit:\\n",
        "            return \\\"仅GT Thought+Action有益\\\"\\n",
        "        else:\\n",
        "            return \\\"仅GT Full有益\\\"\\n",
        "    else:\\n",
        "        return \\\"混合受益\\\"\\n",
        "\\n",
        "df['toolcall_benefit_type'] = df.apply(classify_toolcall_benefit, axis=1)\\n",
        "\\n",
        "# 统计各类型数量\\n",
        "benefit_counts = df['toolcall_benefit_type'].value_counts()\\n",
        "benefit_percentages = df['toolcall_benefit_type'].value_counts(normalize=True) * 100\\n",
        "\\n",
        "print(\\\"=\\\" * 80)\\n",
        "print(\\\"对于 tool_call 的推理链受益类型统计\\\")\\n",
        "print(\\\"=\\\" * 80)\\n",
        "for benefit_type in benefit_counts.index:\\n",
        "    count = benefit_counts[benefit_type]\\n",
        "    pct = benefit_percentages[benefit_type]\\n",
        "    print(f\\\"{benefit_type:30s}: {count:4d} ({pct:5.2f}%)\\\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 可视化：tool_call 推理链受益类型分布\\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\\n",
        "benefit_counts.plot(kind='bar', ax=ax, color='steelblue')\\n",
        "ax.set_title('对于 tool_call 的推理链受益类型分布', fontsize=14, fontweight='bold')\\n",
        "ax.set_xlabel('受益类型', fontsize=12)\\n",
        "ax.set_ylabel('样本数量', fontsize=12)\\n",
        "ax.grid(axis='y', alpha=0.3)\\n",
        "plt.xticks(rotation=45, ha='right')\\n",
        "\\n",
        "# 添加百分比标注\\n",
        "for i, (count, pct) in enumerate(zip(benefit_counts, benefit_percentages)):\\n",
        "    ax.text(i, count + 0.5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)\\n",
        "\\n",
        "plt.tight_layout()\\n",
        "plt.show()"
    ]
})

# ============================================================================
# Section 2.2: Action 推理链价值分析
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### 2.2 对于 Action 的推理链价值"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def classify_action_benefit(row) -> str:\\n",
        "    \\\"\\\"\\\"对于 Action 的 Thought 受益类型\\\"\\\"\\\"\\n",
        "    threshold = 0.05\\n",
        "    if pd.isna(row['model_thought_to_action_infogain']):\\n",
        "        return \\\"无Action数据\\\"\\n",
        "\\n",
        "    model_thought_benefit = row['model_thought_to_action_infogain'] > threshold\\n",
        "    gt_thought_benefit = row['gt_thought_to_action_infogain'] > threshold if not pd.isna(row['gt_thought_to_action_infogain']) else False\\n",
        "    gt_full_benefit = row['gt_full_to_action_infogain'] > threshold if not pd.isna(row['gt_full_to_action_infogain']) else False\\n",
        "\\n",
        "    benefits = [model_thought_benefit, gt_thought_benefit, gt_full_benefit]\\n",
        "    benefit_count = sum(benefits)\\n",
        "\\n",
        "    if benefit_count == 0:\\n",
        "        return \\\"无增益\\\"\\n",
        "    elif benefit_count == 1:\\n",
        "        if model_thought_benefit:\\n",
        "            return \\\"仅模型Thought有益\\\"\\n",
        "        elif gt_thought_benefit:\\n",
        "            return \\\"仅GT Thought有益\\\"\\n",
        "        else:\\n",
        "            return \\\"仅GT Full有益\\\"\\n",
        "    else:\\n",
        "        return \\\"混合受益\\\"\\n",
        "\\n",
        "df['action_benefit_type'] = df.apply(classify_action_benefit, axis=1)\\n",
        "\\n",
        "print(\\\"=\\\" * 80)\\n",
        "print(\\\"对于 Action 的推理链受益类型统计\\\")\\n",
        "print(\\\"=\\\" * 80)\\n",
        "print(df['action_benefit_type'].value_counts())\\n",
        "print(\\\"\\\\n百分比:\\\")\\n",
        "print(df['action_benefit_type'].value_counts(normalize=True) * 100)"
    ]
})

# ============================================================================
# Section 3: 模型 vs GT 对比分析
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 3. 模型 vs GT 对比分析"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Action InfoGain 对比\\n",
        "print(\\\"=\\\" * 80)\\n",
        "print(\\\"模型 Action vs GT Action (对于 tool_call)\\\")\\n",
        "print(\\\"=\\\" * 80)\\n",
        "\\n",
        "model_action_mean = df['model_action_infogain'].mean()\\n",
        "gt_action_mean = df['gt_action_infogain'].mean()\\n",
        "\\n",
        "print(f\\\"模型 Action InfoGain 平均值: {model_action_mean:.4f}\\\")\\n",
        "print(f\\\"GT Action InfoGain 平均值: {gt_action_mean:.4f}\\\")\\n",
        "print(f\\\"差距: {gt_action_mean - model_action_mean:.4f}\\\")\\n",
        "print(f\\\"模型相对GT的比例: {(model_action_mean / gt_action_mean * 100):.2f}%\\\")\\n",
        "\\n",
        "# 统计模型优于 GT 的样本\\n",
        "model_better = (df['model_action_infogain'] > df['gt_action_infogain']).sum()\\n",
        "gt_better = (df['gt_action_infogain'] > df['model_action_infogain']).sum()\\n",
        "\\n",
        "print(f\\\"\\\\n模型 Action 优于 GT: {model_better} ({model_better/len(df)*100:.2f}%)\\\")\\n",
        "print(f\\\"GT Action 优于模型: {gt_better} ({gt_better/len(df)*100:.2f}%)\\\")\\n",
        "\\n",
        "# Thought+Action InfoGain 对比\\n",
        "print(\\\"\\\\n\\\" + \\\"=\\\" * 80)\\n",
        "print(\\\"模型 Thought+Action vs GT Thought+Action (对于 tool_call)\\\")\\n",
        "print(\\\"=\\\" * 80)\\n",
        "\\n",
        "model_ta_mean = df['model_thought_action_infogain'].mean()\\n",
        "gt_ta_mean = df['gt_thought_action_infogain'].mean()\\n",
        "\\n",
        "print(f\\\"模型 Thought+Action InfoGain 平均值: {model_ta_mean:.4f}\\\")\\n",
        "print(f\\\"GT Thought+Action InfoGain 平均值: {gt_ta_mean:.4f}\\\")\\n",
        "print(f\\\"差距: {gt_ta_mean - model_ta_mean:.4f}\\\")\\n",
        "print(f\\\"模型相对GT的比例: {(model_ta_mean / gt_ta_mean * 100):.2f}%\\\")\\n",
        "\\n",
        "model_better_ta = (df['model_thought_action_infogain'] > df['gt_thought_action_infogain']).sum()\\n",
        "gt_better_ta = (df['gt_thought_action_infogain'] > df['model_thought_action_infogain']).sum()\\n",
        "\\n",
        "print(f\\\"\\\\n模型 Thought+Action 优于 GT: {model_better_ta} ({model_better_ta/len(df)*100:.2f}%)\\\")\\n",
        "print(f\\\"GT Thought+Action 优于模型: {gt_better_ta} ({gt_better_ta/len(df)*100:.2f}%)\\\")"
    ]
})

# ============================================================================
# Section 4: 散点图对比
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 4. 模型 vs GT 散点图对比"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 模型 vs GT InfoGain 散点图\\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\\n",
        "\\n",
        "# Action InfoGain\\n",
        "axes[0].scatter(df['gt_action_infogain'], df['model_action_infogain'], alpha=0.5)\\n",
        "axes[0].plot([df['gt_action_infogain'].min(), df['gt_action_infogain'].max()],\\n",
        "             [df['gt_action_infogain'].min(), df['gt_action_infogain'].max()],\\n",
        "             'r--', label='y=x')\\n",
        "axes[0].set_xlabel('GT Action InfoGain')\\n",
        "axes[0].set_ylabel('模型 Action InfoGain')\\n",
        "axes[0].set_title('Action InfoGain: 模型 vs GT')\\n",
        "axes[0].legend()\\n",
        "axes[0].grid(alpha=0.3)\\n",
        "\\n",
        "# Thought+Action InfoGain\\n",
        "axes[1].scatter(df['gt_thought_action_infogain'], df['model_thought_action_infogain'],\\n",
        "               alpha=0.5, color='orange')\\n",
        "axes[1].plot([df['gt_thought_action_infogain'].min(), df['gt_thought_action_infogain'].max()],\\n",
        "             [df['gt_thought_action_infogain'].min(), df['gt_thought_action_infogain'].max()],\\n",
        "             'r--', label='y=x')\\n",
        "axes[1].set_xlabel('GT Thought+Action InfoGain')\\n",
        "axes[1].set_ylabel('模型 Thought+Action InfoGain')\\n",
        "axes[1].set_title('Thought+Action InfoGain: 模型 vs GT')\\n",
        "axes[1].legend()\\n",
        "axes[1].grid(alpha=0.3)\\n",
        "\\n",
        "plt.tight_layout()\\n",
        "plt.show()"
    ]
})

# ============================================================================
# Section 5: 双方都无增益的样本分析
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 5. 双方都无增益的样本分析"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 找出双方 Thought 都无增益的样本\\n",
        "threshold = 0.05\\n",
        "\\n",
        "both_no_benefit_toolcall = df[\\n",
        "    (df['model_thought_action_infogain'] <= threshold) &\\n",
        "    (df['gt_thought_action_infogain'] <= threshold)\\n",
        "]\\n",
        "\\n",
        "print(\\\"=\\\" * 80)\\n",
        "print(\\\"模型和GT的Thought+Action都无增益的样本 (对于 tool_call)\\\")\\n",
        "print(\\\"=\\\" * 80)\\n",
        "print(f\\\"样本数: {len(both_no_benefit_toolcall)} ({len(both_no_benefit_toolcall)/len(df)*100:.2f}%)\\\")\\n",
        "print(f\\\"\\\\n这些样本的特征:\\\")\\n",
        "print(f\\\"平均 baseline perplexity: {both_no_benefit_toolcall['perplexity_baseline_toolcall'].mean():.4f}\\\")\\n",
        "print(f\\\"平均 model action perplexity: {both_no_benefit_toolcall['perplexity_model_action_toolcall'].mean():.4f}\\\")\\n",
        "print(f\\\"平均 model thought+action perplexity: {both_no_benefit_toolcall['perplexity_model_thought_action_toolcall'].mean():.4f}\\\")\\n",
        "\\n",
        "# 分析这些样本是否是因为任务太简单\\n",
        "print(f\\\"\\\\n与全体样本对比:\\\")\\n",
        "print(f\\\"全体平均 baseline perplexity: {df['perplexity_baseline_toolcall'].mean():.4f}\\\")\\n",
        "print(f\\\"全体平均 model action perplexity: {df['perplexity_model_action_toolcall'].mean():.4f}\\\")\\n",
        "print(f\\\"全体平均 model thought+action perplexity: {df['perplexity_model_thought_action_toolcall'].mean():.4f}\\\")"
    ]
})

# ============================================================================
# Section 6: InfoGain 分布分析
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 6. InfoGain 分布分析"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# InfoGain 分布直方图\\n",
        "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\\n",
        "\\n",
        "infogain_metrics = [\\n",
        "    ('model_action_infogain', '模型Action InfoGain'),\\n",
        "    ('model_thought_action_infogain', '模型Thought+Action InfoGain'),\\n",
        "    ('gt_action_infogain', 'GT Action InfoGain'),\\n",
        "    ('gt_thought_action_infogain', 'GT Thought+Action InfoGain'),\\n",
        "    ('model_thought_to_action_infogain', '模型Thought对Action InfoGain'),\\n",
        "    ('gt_thought_to_action_infogain', 'GT Thought对Action InfoGain'),\\n",
        "]\\n",
        "\\n",
        "for idx, (col, title) in enumerate(infogain_metrics):\\n",
        "    ax = axes[idx // 3, idx % 3]\\n",
        "    data = df[col].dropna()\\n",
        "\\n",
        "    ax.hist(data, bins=50, alpha=0.7, edgecolor='black')\\n",
        "    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='零点')\\n",
        "    ax.axvline(data.mean(), color='green', linestyle='--', linewidth=2, label=f'均值={data.mean():.3f}')\\n",
        "    ax.set_xlabel('InfoGain')\\n",
        "    ax.set_ylabel('样本数')\\n",
        "    ax.set_title(title, fontweight='bold')\\n",
        "    ax.legend()\\n",
        "    ax.grid(alpha=0.3)\\n",
        "\\n",
        "    # 统计正负样本\\n",
        "    positive = (data > 0).sum()\\n",
        "    negative = (data <= 0).sum()\\n",
        "    ax.text(0.02, 0.98, f'正样本: {positive} ({positive/len(data)*100:.1f}%)\\\\n负样本: {negative} ({negative/len(data)*100:.1f}%)',\\n",
        "            transform=ax.transAxes, fontsize=9, verticalalignment='top',\\n",
        "            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\\n",
        "\\n",
        "plt.tight_layout()\\n",
        "plt.show()"
    ]
})

# ============================================================================
# Section 7: Thought 增量价值分析
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 7. Thought 增量价值分析"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Thought 增量价值分析\\n",
        "df['model_thought_incremental'] = df['model_thought_action_infogain'] - df['model_action_infogain']\\n",
        "df['gt_thought_incremental'] = df['gt_thought_action_infogain'] - df['gt_action_infogain']\\n",
        "\\n",
        "print(\\\"=\\\" * 80)\\n",
        "print(\\\"Thought 增量价值分析\\\")\\n",
        "print(\\\"=\\\" * 80)\\n",
        "\\n",
        "# 模型 Thought 增量\\n",
        "model_thought_positive = (df['model_thought_incremental'] > 0.05).sum()\\n",
        "model_thought_negative = (df['model_thought_incremental'] < -0.05).sum()\\n",
        "model_thought_neutral = len(df) - model_thought_positive - model_thought_negative\\n",
        "\\n",
        "print(f\\\"模型 Thought 增量价值:\\\")\\n",
        "print(f\\\"  有额外价值（>0.05）: {model_thought_positive} ({model_thought_positive/len(df)*100:.1f}%)\\\")\\n",
        "print(f\\\"  有负面影响（<-0.05）: {model_thought_negative} ({model_thought_negative/len(df)*100:.1f}%)\\\")\\n",
        "print(f\\\"  基本无影响: {model_thought_neutral} ({model_thought_neutral/len(df)*100:.1f}%)\\\")\\n",
        "print(f\\\"  平均增量: {df['model_thought_incremental'].mean():.4f}\\\")\\n",
        "\\n",
        "# GT Thought 增量\\n",
        "gt_thought_positive = (df['gt_thought_incremental'] > 0.05).sum()\\n",
        "gt_thought_negative = (df['gt_thought_incremental'] < -0.05).sum()\\n",
        "gt_thought_neutral = len(df) - gt_thought_positive - gt_thought_negative\\n",
        "\\n",
        "print(f\\\"\\\\nGT Thought 增量价值:\\\")\\n",
        "print(f\\\"  有额外价值（>0.05）: {gt_thought_positive} ({gt_thought_positive/len(df)*100:.1f}%)\\\")\\n",
        "print(f\\\"  有负面影响（<-0.05）: {gt_thought_negative} ({gt_thought_negative/len(df)*100:.1f}%)\\\")\\n",
        "print(f\\\"  基本无影响: {gt_thought_neutral} ({gt_thought_neutral/len(df)*100:.1f}%)\\\")\\n",
        "print(f\\\"  平均增量: {df['gt_thought_incremental'].mean():.4f}\\\")"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Thought 增量分布可视化\\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\\n",
        "\\n",
        "# 模型 Thought 增量\\n",
        "axes[0].hist(df['model_thought_incremental'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')\\n",
        "axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='零点')\\n",
        "axes[0].axvline(df['model_thought_incremental'].mean(), color='green', linestyle='--', \\n",
        "               linewidth=2, label=f\\\"均值={df['model_thought_incremental'].mean():.3f}\\\")\\n",
        "axes[0].set_xlabel('Thought 增量 InfoGain')\\n",
        "axes[0].set_ylabel('样本数')\\n",
        "axes[0].set_title('模型 Thought 增量价值分布', fontweight='bold')\\n",
        "axes[0].legend()\\n",
        "axes[0].grid(alpha=0.3)\\n",
        "\\n",
        "# GT Thought 增量\\n",
        "axes[1].hist(df['gt_thought_incremental'], bins=50, alpha=0.7, edgecolor='black', color='orange')\\n",
        "axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='零点')\\n",
        "axes[1].axvline(df['gt_thought_incremental'].mean(), color='green', linestyle='--',\\n",
        "               linewidth=2, label=f\\\"均值={df['gt_thought_incremental'].mean():.3f}\\\")\\n",
        "axes[1].set_xlabel('Thought 增量 InfoGain')\\n",
        "axes[1].set_ylabel('样本数')\\n",
        "axes[1].set_title('GT Thought 增量价值分布', fontweight='bold')\\n",
        "axes[1].legend()\\n",
        "axes[1].grid(alpha=0.3)\\n",
        "\\n",
        "plt.tight_layout()\\n",
        "plt.show()"
    ]
})

# ============================================================================
# Section 8: InfoGain 作为 RL Reward 的可行性评估
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 8. InfoGain 作为 RL Reward 的可行性评估"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\\\"=\\\" * 80)\\n",
        "print(\\\"InfoGain 作为 RL Reward 的可行性评估\\\")\\n",
        "print(\\\"=\\\" * 80)\\n",
        "\\n",
        "# 1. 动态范围\\n",
        "print(\\\"\\\\n1. 动态范围分析:\\\")\\n",
        "for col, title in infogain_metrics:\\n",
        "    data = df[col].dropna()\\n",
        "    dynamic_range = data.max() - data.min()\\n",
        "    std = data.std()\\n",
        "    print(f\\\"  {title:35s}: 范围=[{data.min():.3f}, {data.max():.3f}], 动态范围={dynamic_range:.3f}, 标准差={std:.3f}\\\")\\n",
        "\\n",
        "# 2. 正负样本比例\\n",
        "print(\\\"\\\\n2. 正负样本比例:\\\")\\n",
        "for col, title in infogain_metrics:\\n",
        "    data = df[col].dropna()\\n",
        "    positive_ratio = (data > 0).sum() / len(data) * 100\\n",
        "    print(f\\\"  {title:35s}: 正样本={positive_ratio:.1f}%\\\")\\n",
        "\\n",
        "# 3. 与 baseline perplexity 的关系\\n",
        "print(\\\"\\\\n3. InfoGain 与 baseline perplexity 的相关性:\\\")\\n",
        "print(f\\\"  model_action_infogain: {df['model_action_infogain'].corr(df['perplexity_baseline_toolcall']):.3f}\\\")\\n",
        "print(f\\\"  model_thought_action_infogain: {df['model_thought_action_infogain'].corr(df['perplexity_baseline_toolcall']):.3f}\\\")\\n",
        "\\n",
        "# 4. InfoGain 稳定性（按 query 分组）\\n",
        "print(\\\"\\\\n4. InfoGain 稳定性（样本内方差）:\\\")\\n",
        "query_groups = df.groupby('query_id')\\n",
        "model_action_std_within = query_groups['model_action_infogain'].std().mean()\\n",
        "model_ta_std_within = query_groups['model_thought_action_infogain'].std().mean()\\n",
        "print(f\\\"  model_action_infogain 样本内平均标准差: {model_action_std_within:.4f}\\\")\\n",
        "print(f\\\"  model_thought_action_infogain 样本内平均标准差: {model_ta_std_within:.4f}\\\")"
    ]
})

# ============================================================================
# Section 9: 基于 InfoGain 的样本筛选策略
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 9. 基于 InfoGain 的样本筛选策略"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\\\"=\\\" * 80)\\n",
        "print(\\\"基于 InfoGain 的样本筛选策略\\\")\\n",
        "print(\\\"=\\\" * 80)\\n",
        "\\n",
        "# 策略1: 选择高 InfoGain 的样本（前25%）\\n",
        "high_infogain_threshold = df['model_thought_action_infogain'].quantile(0.75)\\n",
        "high_infogain_samples = df[df['model_thought_action_infogain'] > high_infogain_threshold]\\n",
        "\\n",
        "print(f\\\"\\\\n策略1: 选择高 InfoGain 样本（前25%)\\\")\\n",
        "print(f\\\"  阈值: {high_infogain_threshold:.4f}\\\")\\n",
        "print(f\\\"  样本数: {len(high_infogain_samples)} ({len(high_infogain_samples)/len(df)*100:.1f}%)\\\")\\n",
        "\\n",
        "# 策略2: Thought 有明显增量价值\\n",
        "thought_valuable_samples = df[df['model_thought_incremental'] > 0.1]\\n",
        "print(f\\\"\\\\n策略2: Thought 有明显增量价值（>0.1)\\\")\\n",
        "print(f\\\"  样本数: {len(thought_valuable_samples)} ({len(thought_valuable_samples)/len(df)*100:.1f}%)\\\")\\n",
        "\\n",
        "# 策略3: 优于 baseline\\n",
        "better_than_baseline = df[df['model_thought_action_infogain'] > 0.05]\\n",
        "print(f\\\"\\\\n策略3: 优于 baseline（InfoGain>0.05)\\\")\\n",
        "print(f\\\"  样本数: {len(better_than_baseline)} ({len(better_than_baseline)/len(df)*100:.1f}%)\\\")\\n",
        "\\n",
        "# 策略4: 接近或优于 GT\\n",
        "model_competitive = df[df['model_thought_action_infogain'] >= 0.8 * df['gt_thought_action_infogain']]\\n",
        "print(f\\\"\\\\n策略4: 接近GT（≥80% GT InfoGain)\\\")\\n",
        "print(f\\\"  样本数: {len(model_competitive)} ({len(model_competitive)/len(df)*100:.1f}%)\\\")\\n",
        "\\n",
        "# 策略5: 模型优于 GT\\n",
        "model_better_than_gt = df[df['model_thought_action_infogain'] > df['gt_thought_action_infogain']]\\n",
        "print(f\\\"\\\\n策略5: 模型优于GT\\\")\\n",
        "print(f\\\"  样本数: {len(model_better_than_gt)} ({len(model_better_than_gt)/len(df)*100:.1f}%)\\\")"
    ]
})

# ============================================================================
# Section 10: 导出结果
# ============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 10. 导出分析结果"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 导出带分类标签的结果\\n",
        "output_file = result_file.parent / \\\"infogain_analysis_results.csv\\\"\\n",
        "df.to_csv(output_file, index=False)\\n",
        "print(f\\\"✓ 分析结果已保存到: {output_file}\\\")\\n",
        "\\n",
        "# 导出筛选后的高质量样本\\n",
        "high_quality_samples = df[\\n",
        "    (df['model_thought_action_infogain'] > 0.05) &  # 优于 baseline\\n",
        "    (df['model_thought_incremental'] > 0)  # Thought 有正面贡献\\n",
        "]\\n",
        "high_quality_file = result_file.parent / \\\"high_quality_samples.csv\\\"\\n",
        "high_quality_samples.to_csv(high_quality_file, index=False)\\n",
        "print(f\\\"✓ 高质量样本已保存到: {high_quality_file}\\\")\\n",
        "print(f\\\"  数量: {len(high_quality_samples)} ({len(high_quality_samples)/len(df)*100:.1f}%)\\\")"
    ]
})

# 保存 notebook
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = "validation/analysis/_infogain_analysis_complete.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✓ 完整分析 Notebook 创建成功: {output_path}")
print(f"  包含 {len(cells)} 个单元格")
print(f"\n包含的分析章节:")
print("  1. 数据加载和预处理")
print("  2. 推理链价值分析 (tool_call 和 Action)")
print("  3. 模型 vs GT 对比分析")
print("  4. 散点图对比")
print("  5. 双方都无增益的样本分析")
print("  6. InfoGain 分布分析")
print("  7. Thought 增量价值分析")
print("  8. InfoGain 作为 RL Reward 的可行性评估")
print("  9. 基于 InfoGain 的样本筛选策略")
print(" 10. 导出分析结果")
