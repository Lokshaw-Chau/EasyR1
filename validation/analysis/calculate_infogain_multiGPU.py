#!/usr/bin/env python3
"""
多 GPU 数据并行 InfoGain 计算

第一阶段：预处理数据，组织所有需要计算的样本
第二阶段：使用多进程在多个 GPU 上并行推理
"""

import os
import json
import argparse
import torch
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import multiprocessing as mp
from functools import partial
import pickle
from torch.utils.data import Dataset, DataLoader


# 导入原始的辅助函数
import sys
sys.path.insert(0, str(Path(__file__).parent))

WEB_SYS_PROMPT = (
    "You are a helpful assistant.\n\n"
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{"
        "\"type\": \"function\", "
        "\"function\": {"
            "\"name_for_human\": \"computer_use\", "
            "\"name\": \"computer_use\", "
            "\"description\": \"Use a mouse and keyboard to interact with a computer, and take screenshots.\\n"
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n"
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\\n"
            "* The screen's resolution is 1000x1000.\\n"
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n"
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", "
        "\"parameters\": {"
            "\"properties\": {"
                "\"action\": {"
                    "\"description\": \"The action to perform. The available actions are:\\n"
                    "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n"
                    "* `type`: Type a string of text on the keyboard.\\n"
                    "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n"
                    "* `left_click`: Click the left mouse button.\\n"
                    "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n"
                    "* `right_click`: Click the right mouse button.\\n"
                    "* `double_click`: Double-click the left mouse button.\\n"
                    "* `scroll`: Performs a scroll of the mouse scroll wheel.\\n"
                    "* `terminate`: Terminate the current task and report its completion status.\", "
                    "\"enum\": [\"key\", \"type\", \"mouse_move\", \"left_click\", \"left_click_drag\", \"right_click\", \"double_click\", \"scroll\", \"terminate\"], \"type\": \"string\"}, "
                "\"keys\": {\"description\": \"Required only by `action=key`.\", \"type\": \"array\"}, "
                "\"text\": {\"description\": \"Required only by `action=type`.\", \"type\": \"string\"}, "
                "\"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`, `action=right_click`, `action=double_click`, `action=middle_click`, `action=mouse_move` and `action=left_click_drag`.\", \"type\": \"array\"}, "
                "\"pixels\": {\"description\": \"The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.\", \"type\": \"number\"}, "
                "\"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, "
            "\"required\": [\"action\"], "
            "\"type\": \"object\"}, "
        "\"args_format\": \"Format the arguments as a JSON object.\"}"
    "}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    "# Response format\n\n"
    "Response unit for every step:\n"
    "1) Thought: one concise sentence explaining the next move (no multi-step reasoning).\n"
    "2) Action: a short imperative describing what to do in the UI.\n"
    "3) A single <tool_call>...</tool_call> block containing only the JSON: {\"name\": <function-name>, \"arguments\": <args-json-object>}.\n\n"
    "Rules:\n"
    "- Thought is optional, omit if not needed if next action is obvious.\n"
    "- Output exactly in the order: Thought, Action, <tool_call>.\n"
    "- Be brief: one sentence for Thought, one for Action.\n"
    "- Do not output anything else outside those three parts.\n"
    "- If finishing, use action=terminate in the tool call."
)

class DataPreprocessor:
    """第一阶段：数据预处理"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_trajectory(self, task_id: str):
        """加载 trajectory"""
        trajectory_path = self.data_dir / f"{task_id}.json"
        if not trajectory_path.exists():
            return None
        with open(trajectory_path, "r") as f:
            return json.load(f)

    def load_image(self, task_id: str, step_num: int):
        """加载图像"""
        image_path = self.data_dir / "images" / f"{task_id}_{step_num}.png"
        if not image_path.exists():
            return None
        return Image.open(image_path)

    def construct_tool_call_from_gt(self, ground_truth_actions: List[Dict]) -> str:
        """
        从 ground_truth 构造 tool_call
        参考 qwen3vl.py 中的 _format_action_response 实现
        """
        if not ground_truth_actions:
            return ""

        # 初始化 action 对象
        action_obj = {
            "name": "computer_use",
            "arguments": {}
        }

        # 检查复合动作 (moveTo + scroll 或 moveTo + dragTo)
        if len(ground_truth_actions) >= 2:
            action1 = ground_truth_actions[0]
            action2 = ground_truth_actions[1]

            # moveTo + scroll 复合动作
            if action1.get('type') == 'moveTo' and action2.get('type') == 'scroll':
                action_obj["arguments"]["action"] = "scroll"

                # 从 moveTo 获取坐标
                if 'params' in action1 and 'position' in action1['params']:
                    pos = action1['params']['position']
                    if isinstance(pos, dict):
                        x = pos.get('x', 0)
                        y = pos.get('y', 0)
                        action_obj["arguments"]["coordinate"] = [
                            int(x * 1000),
                            int(y * 1000)
                        ]

                # 从 scroll 获取滚动参数
                scroll_params = action2.get('params', {})
                amount = scroll_params.get('amount', -3)
                # 转换为 pixels (向下滚动是负数)
                pixels = amount * -100
                action_obj["arguments"]["pixels"] = pixels

                return f"<tool_call>\n{json.dumps(action_obj)}\n</tool_call>"

            # moveTo + dragTo 复合动作
            elif action1.get('type') == 'moveTo' and action2.get('type') == 'dragTo':
                action_obj["arguments"]["action"] = "left_click_drag"

                # 使用 dragTo 的目标坐标
                if 'params' in action2 and 'position' in action2['params']:
                    pos = action2['params']['position']
                    if isinstance(pos, dict):
                        x = pos.get('x', 0)
                        y = pos.get('y', 0)
                        action_obj["arguments"]["coordinate"] = [
                            int(x * 1000),
                            int(y * 1000)
                        ]

                return f"<tool_call>\n{json.dumps(action_obj)}\n</tool_call>"

        # 处理单个动作
        action = ground_truth_actions[0]
        action_type = action.get("type", "")
        params = action.get("params", {})

        # 根据 action type 设置对应的参数
        if action_type == 'click':
            action_obj["arguments"]["action"] = "left_click"
            if 'position' in params and isinstance(params['position'], dict):
                x = params['position'].get('x', 0)
                y = params['position'].get('y', 0)
                action_obj["arguments"]["coordinate"] = [
                    int(x * 1000),
                    int(y * 1000)
                ]

        elif action_type == 'doubleClick' or action_type == 'doubleclick':
            action_obj["arguments"]["action"] = "double_click"
            if 'position' in params and isinstance(params['position'], dict):
                x = params['position'].get('x', 0)
                y = params['position'].get('y', 0)
                action_obj["arguments"]["coordinate"] = [
                    int(x * 1000),
                    int(y * 1000)
                ]

        elif action_type == 'rightClick':
            action_obj["arguments"]["action"] = "right_click"
            if 'position' in params and isinstance(params['position'], dict):
                x = params['position'].get('x', 0)
                y = params['position'].get('y', 0)
                action_obj["arguments"]["coordinate"] = [
                    int(x * 1000),
                    int(y * 1000)
                ]

        elif action_type == 'write':
            action_obj["arguments"]["action"] = "type"
            action_obj["arguments"]["text"] = params.get('text', '')

        elif action_type in ['press', 'hotkey']:
            action_obj["arguments"]["action"] = "key"
            keys = params.get('keys', [])
            if isinstance(keys, str):
                keys = [keys]
            action_obj["arguments"]["keys"] = keys

        elif action_type == 'scroll':
            action_obj["arguments"]["action"] = "scroll"
            amount = params.get('amount', -3)
            # 转换为 pixels (向下滚动是负数)
            pixels = amount * -100
            action_obj["arguments"]["pixels"] = pixels
            # scroll 可能有 position
            if 'position' in params and isinstance(params['position'], dict):
                x = params['position'].get('x', 0)
                y = params['position'].get('y', 0)
                action_obj["arguments"]["coordinate"] = [
                    int(x * 1000),
                    int(y * 1000)
                ]

        elif action_type == 'moveTo':
            action_obj["arguments"]["action"] = "mouse_move"
            if 'position' in params and isinstance(params['position'], dict):
                x = params['position'].get('x', 0)
                y = params['position'].get('y', 0)
                action_obj["arguments"]["coordinate"] = [
                    int(x * 1000),
                    int(y * 1000)
                ]

        elif action_type == 'dragTo':
            action_obj["arguments"]["action"] = "left_click_drag"
            if 'position' in params and isinstance(params['position'], dict):
                x = params['position'].get('x', 0)
                y = params['position'].get('y', 0)
                action_obj["arguments"]["coordinate"] = [
                    int(x * 1000),
                    int(y * 1000)
                ]

        elif action_type == 'terminate':
            action_obj["arguments"]["action"] = "terminate"
            action_obj["arguments"]["status"] = params.get('status', 'success')

        elif action_type == 'wait' or action_type == 'sleep':
            action_obj["arguments"]["action"] = "wait"
            action_obj["arguments"]["time"] = params.get('time', 1)

        else:
            # 未知动作类型，使用原始类型
            action_obj["arguments"]["action"] = action_type.lower()

        return f"<tool_call>\n{json.dumps(action_obj)}\n</tool_call>"

    def extract_thought_action_toolcall(self, response: str) -> Tuple[str, str, str]:
        """提取 Thought, Action, tool_call"""
        tool_call_match = re.search(r'<tool_call>.*?</tool_call>', response, re.DOTALL)

        if tool_call_match:
            tool_call_part = tool_call_match.group(0)
            before_tool_call = response[:tool_call_match.start()].strip()

            thought_pattern = r'(?:Thought|THOUGHT|thought):\s*(.*?)(?=(?:Action|ACTION|action):|$)'
            action_pattern = r'(?:Action|ACTION|action):\s*(.*?)$'

            thought_match = re.search(thought_pattern, before_tool_call, re.DOTALL | re.IGNORECASE)
            action_match = re.search(action_pattern, before_tool_call, re.DOTALL | re.IGNORECASE)

            thought_part = ""
            if thought_match:
                thought_part = "Thought: " + thought_match.group(1).strip()

            action_part = ""
            if action_match:
                action_part = "Action: " + action_match.group(1).strip()

            return thought_part, action_part, tool_call_part
        else:
            return response, "", ""

    def prepare_all_samples(self, sampling_file: Path):
        """
        预处理所有数据，返回需要计算的样本列表

        Returns:
            List[Dict]: 每个元素包含：
                - sample_id: 全局样本ID
                - query_id: query ID
                - task_id: task ID
                - step_num: 步骤编号
                - prompt: 输入 prompt
                - targets: [baseline_target, with_action_target, with_thought_action_target]
                - image_path: 图像路径（用于后续加载）
                - sample_idx: 原始 sample 索引
        """
        print("="*70)
        print("第一阶段：数据预处理")
        print("="*70)

        # 读取采样结果
        with open(sampling_file, "r") as f:
            sampling_results = [json.loads(line) for line in f]

        all_samples = []
        sample_id = 0

        for query_result in tqdm(sampling_results, desc="预处理数据"):
            query_id = query_result["query_id"]

            # 解析 query_id
            if "-" not in query_id:
                continue

            task_id, step_num_str = query_id.rsplit("-", 1)
            try:
                step_num = int(step_num_str)
            except:
                continue

            # 加载 trajectory
            trajectory = self.load_trajectory(task_id)
            if not trajectory:
                continue

            # 获取 ground truth
            if step_num >= len(trajectory["steps"]):
                continue

            step_data = trajectory["steps"][step_num]
            ground_truth_actions = step_data.get("ground_truth_actions", [])

            # 构造 ground truth tool_call
            gt_tool_call = self.construct_tool_call_from_gt(ground_truth_actions)
            if not gt_tool_call:
                continue

            # 构造 prompt
            instruction = query_result.get("instruction", "")
            history = query_result.get("history", "None")
            prompt = (
                f"<image>\nThe user query: {instruction}\n"
                f"Task progress (You have done the following operation on the current device): {history}\n"
            )

            # 图像路径
            # image_path = str(self.data_dir / "images" / f"{task_id}_{step_num}.png")
            image_path = str(self.data_dir / "images" / step_data['image'])

            # 从 ground truth 获取所有需要的部分
            gt_observation = step_data.get("inner_monologue", {}).get("observation", "")
            gt_thought = step_data.get("inner_monologue", {}).get("thought", "")
            gt_low_level_instruction = step_data.get("inner_monologue", {}).get("low_level_instruction", "")

            # 格式化 ground truth 部分
            gt_observation_text = f"Observation: {gt_observation}" if gt_observation else ""
            gt_thought_text = f"Thought: {gt_thought}" if gt_thought else ""
            gt_action_text = f"Action: {gt_low_level_instruction}" if gt_low_level_instruction else ""

            # 为每个 sample 构造对照组
            # GT 相关的对照组只在第一个 sample 时创建（所有 sample 共享）
            for sample_idx, sample in enumerate(query_result["samples"]):
                # 从模型 sampling 结果中提取 Thought 和 Action
                response = "Thought:" + sample["response"]
                model_thought, model_action, _ = self.extract_thought_action_toolcall(response)

                # ============ 每个 sample 都需要的对照组（涉及模型生成）============

                # 1. baseline_toolcall: 只有 <tool_call>（作为基准，每个sample都需要）
                all_samples.append({
                    "sample_id": sample_id,
                    "query_id": query_id,
                    "task_id": task_id,
                    "step_num": step_num,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "target": gt_tool_call,
                    "image_path": image_path,
                    "perplexity_type": "baseline_toolcall",
                    "measure_part": "tool_call",
                })
                sample_id += 1

                # 2. model_action_toolcall: 模型 Action + <tool_call>
                target = ""
                if model_action:
                    target += model_action + "\n"
                target += gt_tool_call
                all_samples.append({
                    "sample_id": sample_id,
                    "query_id": query_id,
                    "task_id": task_id,
                    "step_num": step_num,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "target": target,
                    "image_path": image_path,
                    "perplexity_type": "model_action_toolcall",
                    "measure_part": "tool_call",
                })
                sample_id += 1

                # 3. model_thought_action_toolcall: 模型 Thought + 模型 Action + <tool_call>
                target = ""
                if model_thought:
                    target += model_thought + "\n"
                if model_action:
                    target += model_action + "\n"
                target += gt_tool_call
                all_samples.append({
                    "sample_id": sample_id,
                    "query_id": query_id,
                    "task_id": task_id,
                    "step_num": step_num,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "target": target,
                    "image_path": image_path,
                    "perplexity_type": "model_thought_action_toolcall",
                    "measure_part": "tool_call",
                })
                sample_id += 1

                # 4. model_thought_action: 模型 Thought + GT low_level_instruction（用于计算Thought对Action的影响）
                if gt_action_text:
                    target = ""
                    if model_thought:
                        target += model_thought + "\n"
                    target += gt_action_text
                    all_samples.append({
                        "sample_id": sample_id,
                        "query_id": query_id,
                        "task_id": task_id,
                        "step_num": step_num,
                        "sample_idx": sample_idx,
                        "prompt": prompt,
                        "target": target,
                        "image_path": image_path,
                        "perplexity_type": "model_thought_action",
                        "measure_part": "action",
                    })
                    sample_id += 1

                # ============ GT 相关的对照组（只在第一个 sample 时创建）============
                if sample_idx == 0:
                    # 5. gt_action_toolcall: GT low_level_instruction + <tool_call>
                    target = ""
                    if gt_action_text:
                        target += gt_action_text + "\n"
                    target += gt_tool_call
                    all_samples.append({
                        "sample_id": sample_id,
                        "query_id": query_id,
                        "task_id": task_id,
                        "step_num": step_num,
                        "sample_idx": 0,  # 标记为第一个sample
                        "prompt": prompt,
                        "target": target,
                        "image_path": image_path,
                        "perplexity_type": "gt_action_toolcall",
                        "measure_part": "tool_call",
                    })
                    sample_id += 1

                    # 6. gt_thought_action_toolcall: GT thought + GT low_level_instruction + <tool_call>
                    target = ""
                    if gt_thought_text:
                        target += gt_thought_text + "\n"
                    if gt_action_text:
                        target += gt_action_text + "\n"
                    target += gt_tool_call
                    all_samples.append({
                        "sample_id": sample_id,
                        "query_id": query_id,
                        "task_id": task_id,
                        "step_num": step_num,
                        "sample_idx": 0,
                        "prompt": prompt,
                        "target": target,
                        "image_path": image_path,
                        "perplexity_type": "gt_thought_action_toolcall",
                        "measure_part": "tool_call",
                    })
                    sample_id += 1

                    # 7. gt_full_toolcall: GT observation + GT thought + GT low_level_instruction + <tool_call>
                    target = ""
                    if gt_observation_text:
                        target += gt_observation_text + "\n"
                    if gt_thought_text:
                        target += gt_thought_text + "\n"
                    if gt_action_text:
                        target += gt_action_text + "\n"
                    target += gt_tool_call
                    all_samples.append({
                        "sample_id": sample_id,
                        "query_id": query_id,
                        "task_id": task_id,
                        "step_num": step_num,
                        "sample_idx": 0,
                        "prompt": prompt,
                        "target": target,
                        "image_path": image_path,
                        "perplexity_type": "gt_full_toolcall",
                        "measure_part": "tool_call",
                    })
                    sample_id += 1

                    # ============ 对于预测 Action 的 GT 对照组 ============
                    if gt_action_text:
                        # 8. baseline_action: 只有 GT low_level_instruction
                        all_samples.append({
                            "sample_id": sample_id,
                            "query_id": query_id,
                            "task_id": task_id,
                            "step_num": step_num,
                            "sample_idx": 0,
                            "prompt": prompt,
                            "target": gt_action_text,
                            "image_path": image_path,
                            "perplexity_type": "baseline_action",
                            "measure_part": "action",
                        })
                        sample_id += 1

                        # 9. gt_thought_action: GT thought + GT low_level_instruction
                        target = ""
                        if gt_thought_text:
                            target += gt_thought_text + "\n"
                        target += gt_action_text
                        all_samples.append({
                            "sample_id": sample_id,
                            "query_id": query_id,
                            "task_id": task_id,
                            "step_num": step_num,
                            "sample_idx": 0,
                            "prompt": prompt,
                            "target": target,
                            "image_path": image_path,
                            "perplexity_type": "gt_thought_action",
                            "measure_part": "action",
                        })
                        sample_id += 1

                        # 10. gt_full_action: GT observation + GT thought + GT low_level_instruction
                        target = ""
                        if gt_observation_text:
                            target += gt_observation_text + "\n"
                        if gt_thought_text:
                            target += gt_thought_text + "\n"
                        target += gt_action_text
                        all_samples.append({
                            "sample_id": sample_id,
                            "query_id": query_id,
                            "task_id": task_id,
                            "step_num": step_num,
                            "sample_idx": 0,
                            "prompt": prompt,
                            "target": target,
                            "image_path": image_path,
                            "perplexity_type": "gt_full_action",
                            "measure_part": "action",
                        })
                        sample_id += 1

        print(f"\n✓ 预处理完成")
        print(f"  总样本数: {len(all_samples)}")
        print(f"  总查询数: {len(sampling_results)}")

        return all_samples


class InfoGainDataset(Dataset):
    """
    PyTorch Dataset for InfoGain calculation
    支持多线程数据加载
    """

    def __init__(self, samples: List[Dict]):
        """
        Args:
            samples: 预处理好的样本列表，每个样本包含：
                - sample_id
                - query_id
                - sample_idx
                - prompt
                - target
                - image_path
                - perplexity_type
                - thought_part
                - action_part
                - measure_part: "tool_call" | "action"
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        加载单个样本
        在 DataLoader 的 worker 线程中并行执行
        """
        sample = self.samples[idx]

        # 加载图像（多线程并行）
        image = None
        if sample["image_path"]:
            try:
                image = Image.open(sample["image_path"]).convert("RGB")
            except Exception as e:
                print(f"⚠️  警告: 无法加载图像 {sample['image_path']}: {e}")

        return {
            "sample_id": sample["sample_id"],
            "query_id": sample["query_id"],
            "sample_idx": sample["sample_idx"],
            "prompt": sample["prompt"],
            "target": sample["target"],
            "image": image,
            "perplexity_type": sample["perplexity_type"],
            "thought_part": sample.get("thought_part", ""),  # 可选字段
            "action_part": sample.get("action_part", ""),  # 可选字段
            "measure_part": sample.get("measure_part", "tool_call"),  # 默认测量tool_call
        }


def collate_fn(batch):
    """
    自定义 collate 函数，将 batch 中的样本组织成列表
    不进行 tensor 拼接，因为图像大小可能不同
    """
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "query_ids": [item["query_id"] for item in batch],
        "sample_idxs": [item["sample_idx"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "targets": [item["target"] for item in batch],
        "images": [item["image"] for item in batch],
        "perplexity_types": [item["perplexity_type"] for item in batch],
        "thought_parts": [item["thought_part"] for item in batch],
        "action_parts": [item["action_part"] for item in batch],
        "measure_parts": [item["measure_part"] for item in batch],  # 添加measure_part
    }


def worker_process(gpu_id: int, samples: List[Dict], model_path: str,
                   mini_batch_size: int, output_dir: Path, num_workers: int = 4):
    """
    Worker 进程：在指定 GPU 上加载模型并进行推理

    Args:
        gpu_id: GPU ID
        samples: 分配给该进程的样本列表
        model_path: 模型路径
        mini_batch_size: Mini-batch 大小
        output_dir: 输出目录（用于保存中间结果）
        num_workers: DataLoader 的工作线程数（用于并行加载数据）
    """
    # 设置该进程使用的 GPU
    # 计算逻辑 GPU ID（考虑 CUDA_VISIBLE_DEVICES 设置）
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        visible_gpus = [int(g.strip()) for g in cuda_visible.split(",")]
        logical_gpu_id = visible_gpus.index(gpu_id)  # 找到物理 GPU 在可见列表中的位置
    else:
        logical_gpu_id = gpu_id

    device = f"cuda:{logical_gpu_id}"

    print(f"[GPU {gpu_id}] 进程启动，处理 {len(samples)} 个样本")
    print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES={cuda_visible}")
    print(f"[GPU {gpu_id}] 使用逻辑设备: {device}")

    # 加载模型
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 尝试使用 flash-attention
    attn_impl = "flash_attention_2"
    try:
        # 使用具体的 device 而不是 "auto"，避免模型并行导致显存不均
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,  # 指定具体设备，确保模型完全加载到单个GPU
            attn_implementation=attn_impl
        )
        print(f"[GPU {gpu_id}] ✓ Model loaded on {device} with {attn_impl}")
    except Exception as e:
        print(f"[GPU {gpu_id}] Flash-attention not available, using SDPA")
        attn_impl = "sdpa"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,  # 指定具体设备，确保模型完全加载到单个GPU
            attn_implementation=attn_impl
        )
        print(f"[GPU {gpu_id}] ✓ Model loaded on {device} with {attn_impl}")

    model.eval()

    # 创建 Dataset 和 DataLoader（支持多线程数据加载）
    dataset = InfoGainDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=mini_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # 加速 CPU 到 GPU 的数据传输
        prefetch_factor=2,  # 每个 worker 预取的 batch 数
    )

    print(f"[GPU {gpu_id}] DataLoader 配置: batch_size={mini_batch_size}, num_workers={num_workers}")

    # Batch 推理
    results = []

    for batch in tqdm(dataloader, desc=f"GPU {gpu_id}", position=gpu_id):
        # batch 已经通过 collate_fn 组织好了
        prompts = batch["prompts"]
        targets = batch["targets"]
        images = batch["images"]
        measure_parts = batch["measure_parts"]

        # Batch inference
        batch_perplexities = calculate_perplexity_batch_internal(
            model, processor, device, prompts, targets, images, measure_parts
        )

        # 保存结果（包含 target 信息）
        for i in range(len(prompts)):
            results.append({
                "sample_id": batch["sample_ids"][i],
                "query_id": batch["query_ids"][i],
                "sample_idx": batch["sample_idxs"][i],
                "perplexity_type": batch["perplexity_types"][i],
                "perplexity": batch_perplexities[i],
                "thought_part": batch["thought_parts"][i],
                "action_part": batch["action_parts"][i],
                # 添加 target 信息以便后续分析
                "target": targets[i],
                "measure_part": measure_parts[i],
            })

    print(f"[GPU {gpu_id}] 完成，计算了 {len(results)} 个困惑度")

    # 将结果写入JSON文件（避免队列死锁）
    output_file = output_dir / f"gpu_{gpu_id}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[GPU {gpu_id}] 结果已保存到: {output_file}")


def calculate_perplexity_batch_internal(model, processor, device,
                                       prompts: List[str], targets: List[str],
                                       images: List[Image],
                                       measure_parts: List[str]) -> List[float]:
    """
    内部批处理困惑度计算

    修复：
    1. 添加 system message (WEB_SYS_PROMPT)
    2. target 作为 assistant 回复，而不是放在 user 中
    3. 支持测量不同部分的 perplexity：
       - "tool_call": 只计算 <tool_call>...</tool_call> 部分
       - "action": 只计算 Action: ... 部分

    Args:
        measure_parts: 列表，每个样本指定测量哪个部分（"tool_call" 或 "action"）
    """
    try:
        batch_size = len(prompts)

        # 为每个样本构造 messages
        batch_inputs_list = []
        batch_prompt_inputs_list = []
        measure_only_lengths = []  # 用于记录测量部分的长度

        for i in range(batch_size):
            measure_part = measure_parts[i]

            # 根据 measure_part 提取要测量的部分
            if measure_part == "tool_call":
                # 提取 <tool_call> 部分
                target_match = re.search(r'<tool_call>.*?</tool_call>', targets[i], re.DOTALL)
            elif measure_part == "action":
                # 提取 Action: 部分（从 "Action:" 开始，到 <tool_call> 或字符串结尾）
                target_match = re.search(r'Action:.*?(?=<tool_call>|$)', targets[i], re.DOTALL)
            else:
                print(f"⚠️  警告: 未知的 measure_part '{measure_part}'，跳过")
                target_match = None

            if not target_match:
                # 如果没有找到要测量的部分，跳过
                measure_only_lengths.append(0)
                batch_inputs_list.append(None)
                batch_prompt_inputs_list.append(None)
                continue

            measure_only = target_match.group(0)

            # 构造完整对话：system + user + assistant
            # 完整输入（包含 target）
            user_content = []
            if images[i] is not None:
                user_content.append({"type": "image", "image": images[i]})
            else:
                print(f"⚠️  警告: 样本 {i} 缺少图像，继续处理文本部分")
            user_content.append({"type": "text", "text": prompts[i]})

            messages_full = [
                {"role": "system", "content": [{"type": "text", "text": WEB_SYS_PROMPT}]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": targets[i]}]}
            ]

            inputs_full = processor.apply_chat_template(
                messages_full,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            inputs_full.pop("token_type_ids", None)
            batch_inputs_list.append(inputs_full)

            # Prompt 部分（system + user + assistant 前缀，到测量部分之前）
            target_before_measure = targets[i][:target_match.start()]

            messages_before_measure = [
                {"role": "system", "content": [{"type": "text", "text": WEB_SYS_PROMPT}]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": target_before_measure}]}
            ]

            inputs_before_measure = processor.apply_chat_template(
                messages_before_measure,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            inputs_before_measure.pop("token_type_ids", None)
            batch_prompt_inputs_list.append(inputs_before_measure)

            # 计算测量部分的 token 长度
            measure_only_length = inputs_full["input_ids"].shape[1] - inputs_before_measure["input_ids"].shape[1]
            measure_only_lengths.append(measure_only_length)

        # 过滤掉 None 的样本
        valid_indices = [i for i, inp in enumerate(batch_inputs_list) if inp is not None]

        if not valid_indices:
            # 所有样本都无效
            return [float('inf')] * batch_size

        # 只处理有效样本
        valid_batch_inputs_list = [batch_inputs_list[i] for i in valid_indices]
        valid_batch_prompt_inputs_list = [batch_prompt_inputs_list[i] for i in valid_indices]
        valid_measure_only_lengths = [measure_only_lengths[i] for i in valid_indices]

        # Pad 到相同长度
        prompt_lengths = [inp["input_ids"].shape[1] for inp in valid_batch_prompt_inputs_list]
        max_length = max(inp["input_ids"].shape[1] for inp in valid_batch_inputs_list)

        padded_input_ids = []
        padded_attention_mask = []
        all_pixel_values = []
        all_image_grid_thw = []

        for i, inp in enumerate(valid_batch_inputs_list):
            input_ids = inp["input_ids"][0]
            attention_mask = inp.get("attention_mask", torch.ones_like(input_ids))[0]

            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                input_ids = torch.cat([
                    torch.full((pad_length,), processor.tokenizer.pad_token_id, dtype=input_ids.dtype),
                    input_ids
                ])
                attention_mask = torch.cat([
                    torch.zeros(pad_length, dtype=attention_mask.dtype),
                    attention_mask
                ])

            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)

            if "pixel_values" in inp:
                all_pixel_values.append(inp["pixel_values"])
            if "image_grid_thw" in inp:
                all_image_grid_thw.append(inp["image_grid_thw"])

        batch_inputs = {
            "input_ids": torch.stack(padded_input_ids).to(device),
            "attention_mask": torch.stack(padded_attention_mask).to(device),
        }

        if all_pixel_values:
            batch_inputs["pixel_values"] = torch.cat(all_pixel_values, dim=0).to(device)
        if all_image_grid_thw:
            batch_inputs["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0).to(device)

        adjusted_prompt_lengths = [
            pl + (max_length - valid_batch_inputs_list[i]["input_ids"].shape[1])
            for i, pl in enumerate(prompt_lengths)
        ]

        # Forward pass
        with torch.no_grad():
            outputs = model(**batch_inputs)
            logits = outputs.logits

            valid_perplexities = []
            for i in range(len(valid_indices)):
                # prompt_length 是到测量部分之前的长度
                prompt_length = adjusted_prompt_lengths[i]
                measure_length = valid_measure_only_lengths[i]

                if measure_length <= 0:
                    valid_perplexities.append(float('inf'))
                    continue

                # 只计算测量部分的 loss
                # shift_logits: [prompt_length-1 : prompt_length + measure_length - 1]
                # shift_labels: [prompt_length : prompt_length + measure_length]
                start_pos = prompt_length - 1
                end_pos = prompt_length + measure_length - 1

                shift_logits = logits[i:i+1, start_pos:end_pos, :].contiguous()
                shift_labels = batch_inputs["input_ids"][i:i+1, prompt_length:prompt_length+measure_length].contiguous()
                mask = batch_inputs["attention_mask"][i:i+1, prompt_length:prompt_length+measure_length].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                loss = loss.view(shift_labels.shape)
                masked_loss = (loss * mask.float()).sum() / mask.float().sum()

                perplexity = torch.exp(masked_loss).item()
                valid_perplexities.append(perplexity)

        # 将结果映射回原始 batch
        perplexities = []
        valid_idx = 0
        for i in range(batch_size):
            if i in valid_indices:
                perplexities.append(valid_perplexities[valid_idx])
                valid_idx += 1
            else:
                perplexities.append(float('inf'))

        return perplexities

    except Exception as e:
        print(f"Error in batch inference: {e}")
        import traceback
        traceback.print_exc()
        return [float('inf')] * len(prompts)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU InfoGain Calculation")
    parser.add_argument("--sampling_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=4,
                       help="Number of GPUs to use")
    parser.add_argument("--mini_batch_size", type=int, default=8,
                       help="Mini-batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of DataLoader workers per GPU for parallel data loading")

    args = parser.parse_args()

    # 获取实际的物理 GPU IDs（考虑 CUDA_VISIBLE_DEVICES）
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        physical_gpu_ids = [int(gpu.strip()) for gpu in visible_gpus]
        print(f"检测到 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"将使用物理 GPU: {physical_gpu_ids[:args.num_gpus]}")
    else:
        # 如果没有设置，使用 0, 1, 2, ...
        physical_gpu_ids = list(range(args.num_gpus))
        print(f"未设置 CUDA_VISIBLE_DEVICES，将使用 GPU: {physical_gpu_ids}")

    # 第一阶段：数据预处理
    preprocessor = DataPreprocessor(args.data_dir)
    all_samples = preprocessor.prepare_all_samples(Path(args.sampling_file))

    if len(all_samples) == 0:
        print("❌ 没有样本需要处理")
        return

    # 创建临时输出目录
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="infogain_"))
    print(f"临时结果目录: {temp_dir}")

    # 第二阶段：多 GPU 并行推理
    print(f"\n{'='*70}")
    print(f"第二阶段：多 GPU 并行推理")
    print(f"{'='*70}")
    print(f"使用 {args.num_gpus} 个 GPU")
    print(f"每个 GPU 的 mini-batch size: {args.mini_batch_size}")

    # 分配样本到各个 GPU
    samples_per_gpu = len(all_samples) // args.num_gpus
    sample_chunks = []

    for i in range(args.num_gpus):
        start_idx = i * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if i < args.num_gpus - 1 else len(all_samples)
        sample_chunks.append(all_samples[start_idx:end_idx])
        print(f"  GPU {physical_gpu_ids[i]}: {len(sample_chunks[i])} 个样本")

    # 启动多个进程（传递output_dir而不是queue，避免死锁）
    processes = []
    for i in range(args.num_gpus):
        physical_gpu_id = physical_gpu_ids[i]
        p = mp.Process(
            target=worker_process,
            args=(physical_gpu_id, sample_chunks[i], args.model_path,
                  args.mini_batch_size, temp_dir, args.num_workers)
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 从JSON文件收集结果
    print(f"\n{'='*70}")
    print("收集结果...")
    print(f"{'='*70}")

    all_results = []
    for i in range(args.num_gpus):
        gpu_id = physical_gpu_ids[i]
        result_file = temp_dir / f"gpu_{gpu_id}_results.json"
        if result_file.exists():
            with open(result_file, "r") as f:
                gpu_results = json.load(f)
                all_results.extend(gpu_results)
            print(f"✓ 从 GPU {gpu_id} 收集了 {len(gpu_results)} 个结果")
        else:
            print(f"⚠️  警告: GPU {gpu_id} 的结果文件不存在: {result_file}")

    print(f"✓ 总共收集了 {len(all_results)} 个结果")

    # 组织结果（按 query 组织）
    query_results = {}
    query_targets = {}  # 保存 targets 信息

    for result in all_results:
        query_id = result["query_id"]

        # 初始化 query
        if query_id not in query_results:
            query_results[query_id] = {}
            query_targets[query_id] = {
                "per_query_targets": {},  # GT-only targets (sample_idx=0 时记录)
                "per_sample_targets": {}  # 每个 sample 不同的 targets
            }

        sample_idx = result["sample_idx"]
        if sample_idx not in query_results[query_id]:
            query_results[query_id][sample_idx] = {}
        if sample_idx not in query_targets[query_id]["per_sample_targets"]:
            query_targets[query_id]["per_sample_targets"][sample_idx] = {}

        # 保存 perplexity
        query_results[query_id][sample_idx][result["perplexity_type"]] = result["perplexity"]

        # 保存 target 信息
        perplexity_type = result["perplexity_type"]
        target_info = {
            "target": result["target"],
            "measure_part": result["measure_part"],
        }

        # GT-related targets (只在 sample_idx=0 时记录，所有 samples 共享)
        gt_types = ["gt_action_toolcall", "gt_thought_action_toolcall", "gt_full_toolcall",
                   "baseline_action", "gt_thought_action", "gt_full_action"]

        if perplexity_type in gt_types:
            if sample_idx == 0:  # 只在第一个 sample 时记录 GT targets
                query_targets[query_id]["per_query_targets"][perplexity_type] = target_info
        else:
            # Model-related targets (每个 sample 都不同)
            query_targets[query_id]["per_sample_targets"][sample_idx][perplexity_type] = target_info

    # 计算 InfoGain
    print(f"\n{'='*70}")
    print("计算 InfoGain...")
    print(f"{'='*70}")

    final_results = []
    for query_id, samples in query_results.items():
        sample_infogains = []

        for sample_idx, perplexities in samples.items():
            # ============ 对于预测 tool_call 的困惑度 ============
            baseline_toolcall = perplexities.get("baseline_toolcall", float('inf'))
            model_action_toolcall = perplexities.get("model_action_toolcall", float('inf'))
            model_thought_action_toolcall = perplexities.get("model_thought_action_toolcall", float('inf'))
            gt_action_toolcall = perplexities.get("gt_action_toolcall", float('inf'))
            gt_thought_action_toolcall = perplexities.get("gt_thought_action_toolcall", float('inf'))
            gt_full_toolcall = perplexities.get("gt_full_toolcall", float('inf'))

            # ============ 对于预测 Action 的困惑度 ============
            baseline_action = perplexities.get("baseline_action", None)
            model_thought_action = perplexities.get("model_thought_action", None)
            gt_thought_action = perplexities.get("gt_thought_action", None)
            gt_full_action = perplexities.get("gt_full_action", None)

            # ============ 计算对于 tool_call 的 InfoGain ============
            model_action_infogain = baseline_toolcall - model_action_toolcall
            model_thought_action_infogain = baseline_toolcall - model_thought_action_toolcall
            gt_action_infogain = baseline_toolcall - gt_action_toolcall
            gt_thought_action_infogain = baseline_toolcall - gt_thought_action_toolcall
            gt_full_infogain = baseline_toolcall - gt_full_toolcall

            # ============ 计算对于 Action 的 InfoGain ============
            model_thought_to_action_infogain = None
            gt_thought_to_action_infogain = None
            gt_full_to_action_infogain = None

            if baseline_action is not None:
                if model_thought_action is not None:
                    model_thought_to_action_infogain = baseline_action - model_thought_action
                if gt_thought_action is not None:
                    gt_thought_to_action_infogain = baseline_action - gt_thought_action
                if gt_full_action is not None:
                    gt_full_to_action_infogain = baseline_action - gt_full_action

            sample_infogain = {
                "sample_id": sample_idx,
                # ============ 对于 tool_call 的 Perplexity ============
                "perplexity_baseline_toolcall": baseline_toolcall,
                "perplexity_model_action_toolcall": model_action_toolcall,
                "perplexity_model_thought_action_toolcall": model_thought_action_toolcall,
                "perplexity_gt_action_toolcall": gt_action_toolcall,
                "perplexity_gt_thought_action_toolcall": gt_thought_action_toolcall,
                "perplexity_gt_full_toolcall": gt_full_toolcall,
                # ============ 对于 tool_call 的 InfoGain ============
                "model_action_infogain": model_action_infogain,
                "model_thought_action_infogain": model_thought_action_infogain,
                "gt_action_infogain": gt_action_infogain,
                "gt_thought_action_infogain": gt_thought_action_infogain,
                "gt_full_infogain": gt_full_infogain,
                # ============ 对于 Action 的 Perplexity ============
                "perplexity_baseline_action": baseline_action,
                "perplexity_model_thought_action": model_thought_action,
                "perplexity_gt_thought_action": gt_thought_action,
                "perplexity_gt_full_action": gt_full_action,
                # ============ 对于 Action 的 InfoGain ============
                "model_thought_to_action_infogain": model_thought_to_action_infogain,
                "gt_thought_to_action_infogain": gt_thought_to_action_infogain,
                "gt_full_to_action_infogain": gt_full_to_action_infogain,
            }
            sample_infogains.append(sample_infogain)

        # 计算平均值
        valid_samples = [s for s in sample_infogains if s["gt_full_infogain"] != float('inf')]
        valid_samples_with_action = [s for s in sample_infogains if s["model_thought_to_action_infogain"] is not None]

        avg_result = {
            "query_id": query_id,
            "sample_infogains": sample_infogains,
            "inference_targets": query_targets[query_id],  # 保存推理时使用的 target 信息
        }

        # 添加对于 tool_call 的平均 InfoGain
        if valid_samples:
            avg_result["avg_model_action_infogain"] = np.mean([s["model_action_infogain"] for s in valid_samples])
            avg_result["avg_model_thought_action_infogain"] = np.mean([s["model_thought_action_infogain"] for s in valid_samples])
            avg_result["avg_gt_action_infogain"] = np.mean([s["gt_action_infogain"] for s in valid_samples])
            avg_result["avg_gt_thought_action_infogain"] = np.mean([s["gt_thought_action_infogain"] for s in valid_samples])
            avg_result["avg_gt_full_infogain"] = np.mean([s["gt_full_infogain"] for s in valid_samples])

        # 添加对于 Action 的平均 InfoGain
        if valid_samples_with_action:
            avg_result["avg_model_thought_to_action_infogain"] = np.mean([s["model_thought_to_action_infogain"] for s in valid_samples_with_action if s["model_thought_to_action_infogain"] is not None])
            avg_result["avg_gt_thought_to_action_infogain"] = np.mean([s["gt_thought_to_action_infogain"] for s in valid_samples_with_action if s["gt_thought_to_action_infogain"] is not None])
            avg_result["avg_gt_full_to_action_infogain"] = np.mean([s["gt_full_to_action_infogain"] for s in valid_samples_with_action if s["gt_full_to_action_infogain"] is not None])

        final_results.append(avg_result)

    # 保存结果
    output_file = args.output_file or Path(args.sampling_file).parent / "analysis" / f"{Path(args.sampling_file).stem}_infogain_multiGPU.json"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "summary": {
            "total_queries": len(final_results),
            "total_samples": sum(len(query_results[qr["query_id"]]) for qr in final_results),  # 实际样本数
        },
        "detailed_results": final_results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存到: {output_file}")

    # 清理临时目录
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"✓ 临时目录已清理: {temp_dir}")
    except Exception as e:
        print(f"⚠️  清理临时目录失败: {e}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
