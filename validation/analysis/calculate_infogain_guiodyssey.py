#!/usr/bin/env python3
"""
GUI-Odyssey InfoGain 计算 - 多 GPU 数据并行

使用 1000 个随机 episode 计算:
- tool_call baseline perplexity
- GT Action → tool_call 的 InfoGain
- GT Thought+Action → tool_call 的 InfoGain

Thought 定义: description + intention
Action 定义: low_level_instruction
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
import random


# 导入原始的辅助函数
import sys
sys.path.insert(0, str(Path(__file__).parent))

# ODYSSEY System Prompt
ODYSSEY_SYS_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"mobile_use\", \"description\": \"Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", \"parameters\": {\"properties\": {\"action\": {\"description\": \"The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\", \"enum\": [\"click\", \"long_press\", \"swipe\", \"type\", \"answer\", \"system_button\", \"wait\", \"terminate\"], \"type\": \"string\"}, \"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.\", \"type\": \"array\"}, \"coordinate2\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.\", \"type\": \"array\"}, \"text\": {\"description\": \"Required only by `action=type` and `action=answer`.\", \"type\": \"string\"}, \"time\": {\"description\": \"The seconds to wait. Required only by `action=long_press` and `action=wait`.\", \"type\": \"number\"}, \"button\": {\"description\": \"Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`\", \"enum\": [\"Back\", \"Home\", \"Menu\", \"Enter\"], \"type\": \"string\"}, \"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, \"required\": [\"action\"], \"type\": \"object\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n"
    "# Response format\n\n"
    "Response format for every step:\n"
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

# Action type mapping from GUI_odyssey to reward function format
ACTION_MAPPING = {
    'CLICK': 'click',
    'LONG_PRESS': 'long_press',
    'SCROLL': 'swipe',
    'TEXT': 'type',
    'COMPLETE': 'terminate',
    'INCOMPLETE': 'terminate',
}


class GuiOdysseyDataPreprocessor:
    """GUI-Odyssey 数据预处理器"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir / "annotations"
        self.screenshots_dir = self.data_dir / "screenshots" / "screenshots"
        self.split_file = self.data_dir / "splits" / "random_split.json"

    def load_split(self):
        """加载数据集划分"""
        with open(self.split_file, "r") as f:
            splits = json.load(f)
        return splits

    def load_trajectory(self, episode_id: str):
        """加载 trajectory JSON"""
        trajectory_path = self.annotations_dir / f"{episode_id}.json"
        if not trajectory_path.exists():
            return None
        with open(trajectory_path, "r") as f:
            return json.load(f)

    def load_image(self, screenshot_filename: str):
        """加载图像"""
        image_path = self.screenshots_dir / screenshot_filename
        if not image_path.exists():
            return None
        return Image.open(image_path)

    def construct_tool_call_from_step(self, step: Dict) -> str:
        """
        从 GUI-Odyssey step 构造 tool_call
        """
        action = step['action']
        info = step['info']

        # 初始化 action 对象
        action_obj = {
            "name": "mobile_use",
            "arguments": {}
        }

        # 映射 action 类型
        mapped_action = ACTION_MAPPING.get(action, action.lower())
        action_obj["arguments"]["action"] = mapped_action

        # 根据 action 类型设置参数
        if action == 'CLICK':
            if isinstance(info, list) and len(info) > 0:
                coord = info[0] if isinstance(info[0], list) else []
                if coord:
                    action_obj["arguments"]["coordinate"] = coord

        elif action == 'LONG_PRESS':
            if isinstance(info, list) and len(info) > 0:
                coord = info[0] if isinstance(info[0], list) else []
                if coord:
                    action_obj["arguments"]["coordinate"] = coord

        elif action == 'SCROLL':
            # For swipe, use start and end coordinates
            if isinstance(info, list) and len(info) >= 2:
                start = info[0]
                end = info[1]
                if isinstance(start, list) and isinstance(end, list):
                    action_obj["arguments"]["coordinate"] = start
                    action_obj["arguments"]["coordinate2"] = end

        elif action == 'TEXT':
            # Extract text from info
            text = info if isinstance(info, str) else ""
            action_obj["arguments"]["text"] = text

        elif action in ['COMPLETE', 'INCOMPLETE']:
            # Terminate action with status
            status = "success" if action == 'COMPLETE' else "failure"
            action_obj["arguments"]["status"] = status

        # Handle special system buttons (KEY_HOME, KEY_BACK, etc.)
        if isinstance(info, str) and info.startswith('KEY_'):
            action_obj["arguments"]["action"] = 'system_button'
            # Extract button name: KEY_HOME -> Home, KEY_BACK -> Back
            button_name = info.replace('KEY_', '').capitalize()
            # Map to expected button names
            button_mapping = {
                'Home': 'Home',
                'Back': 'Back',
                'Menu': 'Menu',
                'Enter': 'Enter',
            }
            action_obj["arguments"]["button"] = button_mapping.get(button_name, button_name)

        return f"<tool_call>\n{json.dumps(action_obj)}\n</tool_call>"

    def prepare_all_samples(self, num_episodes: int = 1000, random_seed: int = 42):
        """
        预处理数据，从 random 训练集中随机选择 num_episodes 个 episode

        Returns:
            List[Dict]: 每个元素包含：
                - sample_id: 全局样本ID
                - query_id: episode_id_step
                - episode_id: episode ID
                - step_num: 步骤编号
                - prompt: 输入 prompt
                - targets: [baseline_target, gt_action_target, gt_thought_action_target]
                - image_path: 图像路径
                - device_resolution: (width, height)
        """
        print("="*70)
        print("第一阶段：数据预处理")
        print("="*70)

        # 加载划分
        splits = self.load_split()
        train_episodes = splits.get("train", [])

        print(f"训练集共有 {len(train_episodes)} 个 episode")

        # 随机采样 num_episodes
        random.seed(random_seed)
        sampled_episodes = random.sample(train_episodes, min(num_episodes, len(train_episodes)))

        print(f"采样了 {len(sampled_episodes)} 个 episode")

        all_samples = []
        sample_id = 0

        for episode_filename in tqdm(sampled_episodes, desc="预处理数据"):
            episode_id = episode_filename.replace(".json", "")

            # 加载 trajectory
            trajectory = self.load_trajectory(episode_id)
            if not trajectory:
                continue

            # 获取设备分辨率
            device_info = trajectory.get("device_info", {})
            device_width = device_info.get("w", 1440)
            device_height = device_info.get("h", 3120)

            # 获取任务指令
            task_info = trajectory.get("task_info", {})
            instruction = task_info.get("instruction", "")

            # 处理每个 step
            steps = trajectory.get("steps", [])
            history = "None"

            for step_idx, step in enumerate(steps):
                screenshot = step.get("screenshot", "")
                if not screenshot:
                    continue

                # 构造 ground truth tool_call
                gt_tool_call = self.construct_tool_call_from_step(step)

                # 获取 GT Thought 和 Action
                gt_description = step.get("description", "")
                gt_intention = step.get("intention", "")
                gt_low_level_instruction = step.get("low_level_instruction", "")

                # 组合 Thought: description + intention
                gt_thought_text = f"{gt_description} {gt_intention}".strip()
                if gt_thought_text:
                    gt_thought_text = f"Thought: {gt_thought_text}"

                # Action: low_level_instruction
                gt_action_text = f"Action: {gt_low_level_instruction}" if gt_low_level_instruction else ""

                # 构造 prompt
                prompt = (
                    f"<image>\nThe user query: {instruction}\n"
                    f"Task progress (You have done the following operation on the current device): {history}\n"
                )

                # 图像路径
                image_path = str(self.screenshots_dir / screenshot)

                query_id = f"{episode_id}_{step_idx}"

                # 创建 3 个对照组（只计算 GT，不涉及模型生成）

                # 1. baseline_toolcall: 只有 <tool_call>
                all_samples.append({
                    "sample_id": sample_id,
                    "query_id": query_id,
                    "episode_id": episode_id,
                    "step_num": step_idx,
                    "prompt": prompt,
                    "target": gt_tool_call,
                    "image_path": image_path,
                    "perplexity_type": "baseline_toolcall",
                    "measure_part": "tool_call",
                    "device_resolution": (device_width, device_height),
                })
                sample_id += 1

                # 2. gt_action_toolcall: GT Action + <tool_call>
                target = ""
                if gt_action_text:
                    target += gt_action_text + "\n"
                target += gt_tool_call
                all_samples.append({
                    "sample_id": sample_id,
                    "query_id": query_id,
                    "episode_id": episode_id,
                    "step_num": step_idx,
                    "prompt": prompt,
                    "target": target,
                    "image_path": image_path,
                    "perplexity_type": "gt_action_toolcall",
                    "measure_part": "tool_call",
                    "device_resolution": (device_width, device_height),
                })
                sample_id += 1

                # 3. gt_thought_action_toolcall: GT Thought + GT Action + <tool_call>
                target = ""
                if gt_thought_text:
                    target += gt_thought_text + "\n"
                if gt_action_text:
                    target += gt_action_text + "\n"
                target += gt_tool_call
                all_samples.append({
                    "sample_id": sample_id,
                    "query_id": query_id,
                    "episode_id": episode_id,
                    "step_num": step_idx,
                    "prompt": prompt,
                    "target": target,
                    "image_path": image_path,
                    "perplexity_type": "gt_thought_action_toolcall",
                    "measure_part": "tool_call",
                    "device_resolution": (device_width, device_height),
                })
                sample_id += 1

                # 更新 history
                if history == "None":
                    history = f"Step {step_idx + 1}: {gt_low_level_instruction}"
                else:
                    history += f"; Step {step_idx + 1}: {gt_low_level_instruction}"

        print(f"\n✓ 预处理完成")
        print(f"  总样本数: {len(all_samples)}")
        print(f"  每个 step 生成 3 个对照组")

        return all_samples


class InfoGainDataset(Dataset):
    """
    PyTorch Dataset for InfoGain calculation
    支持多线程数据加载
    """

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """加载单个样本"""
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
            "prompt": sample["prompt"],
            "target": sample["target"],
            "image": image,
            "perplexity_type": sample["perplexity_type"],
            "measure_part": sample.get("measure_part", "tool_call"),
            "device_resolution": sample.get("device_resolution", (1440, 3120)),
        }


def collate_fn(batch):
    """自定义 collate 函数"""
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "query_ids": [item["query_id"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "targets": [item["target"] for item in batch],
        "images": [item["image"] for item in batch],
        "perplexity_types": [item["perplexity_type"] for item in batch],
        "measure_parts": [item["measure_part"] for item in batch],
        "device_resolutions": [item["device_resolution"] for item in batch],
    }


def worker_process(gpu_id: int, samples: List[Dict], model_path: str,
                   mini_batch_size: int, output_dir: Path, num_workers: int = 4):
    """
    Worker 进程：在指定 GPU 上加载模型并进行推理
    """
    # 设置该进程使用的 GPU
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        visible_gpus = [int(g.strip()) for g in cuda_visible.split(",")]
        logical_gpu_id = visible_gpus.index(gpu_id)
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
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_impl
        )
        print(f"[GPU {gpu_id}] ✓ Model loaded on {device} with {attn_impl}")
    except Exception as e:
        print(f"[GPU {gpu_id}] Flash-attention not available, using SDPA")
        attn_impl = "sdpa"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_impl
        )
        print(f"[GPU {gpu_id}] ✓ Model loaded on {device} with {attn_impl}")

    model.eval()

    # 创建 Dataset 和 DataLoader
    dataset = InfoGainDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=mini_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )

    print(f"[GPU {gpu_id}] DataLoader 配置: batch_size={mini_batch_size}, num_workers={num_workers}")

    # Batch 推理
    results = []

    for batch in tqdm(dataloader, desc=f"GPU {gpu_id}", position=gpu_id):
        prompts = batch["prompts"]
        targets = batch["targets"]
        images = batch["images"]
        measure_parts = batch["measure_parts"]
        device_resolutions = batch["device_resolutions"]

        # Batch inference
        batch_perplexities = calculate_perplexity_batch_internal(
            model, processor, device, prompts, targets, images, measure_parts, device_resolutions
        )

        # 保存结果
        for i in range(len(prompts)):
            results.append({
                "sample_id": batch["sample_ids"][i],
                "query_id": batch["query_ids"][i],
                "perplexity_type": batch["perplexity_types"][i],
                "perplexity": batch_perplexities[i],
                "target": targets[i],
                "measure_part": measure_parts[i],
            })

    print(f"[GPU {gpu_id}] 完成，计算了 {len(results)} 个困惑度")

    # 将结果写入JSON文件
    output_file = output_dir / f"gpu_{gpu_id}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[GPU {gpu_id}] 结果已保存到: {output_file}")


def calculate_perplexity_batch_internal(model, processor, device,
                                       prompts: List[str], targets: List[str],
                                       images: List[Image],
                                       measure_parts: List[str],
                                       device_resolutions: List[Tuple[int, int]]) -> List[float]:
    """
    内部批处理困惑度计算

    Args:
        device_resolutions: 每个样本的设备分辨率 (width, height)
    """
    try:
        batch_size = len(prompts)

        # 为每个样本构造 messages
        batch_inputs_list = []
        batch_prompt_inputs_list = []
        measure_only_lengths = []

        for i in range(batch_size):
            measure_part = measure_parts[i]
            device_width, device_height = device_resolutions[i]
            system_prompt = ODYSSEY_SYS_PROMPT
            # 动态构造 System Prompt (注入设备分辨率)
            # .format(
            #     display_width_px=device_width,
            #     display_height_px=device_height
            # )

            # 根据 measure_part 提取要测量的部分
            if measure_part == "tool_call":
                target_match = re.search(r'<tool_call>.*?</tool_call>', targets[i], re.DOTALL)
            elif measure_part == "action":
                target_match = re.search(r'Action:.*?(?=<tool_call>|$)', targets[i], re.DOTALL)
            else:
                print(f"⚠️  警告: 未知的 measure_part '{measure_part}'，跳过")
                target_match = None

            if not target_match:
                measure_only_lengths.append(0)
                batch_inputs_list.append(None)
                batch_prompt_inputs_list.append(None)
                continue

            measure_only = target_match.group(0)

            # 构造完整对话
            user_content = []
            if images[i] is not None:
                user_content.append({"type": "image", "image": images[i]})
            else:
                print(f"⚠️  警告: 样本 {i} 缺少图像，继续处理文本部分")
            user_content.append({"type": "text", "text": prompts[i]})

            messages_full = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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

            # Prompt 部分
            target_before_measure = targets[i][:target_match.start()]

            messages_before_measure = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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
                prompt_length = adjusted_prompt_lengths[i]
                measure_length = valid_measure_only_lengths[i]

                if measure_length <= 0:
                    valid_perplexities.append(float('inf'))
                    continue

                # 只计算测量部分的 loss
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
    parser = argparse.ArgumentParser(description="GUI-Odyssey InfoGain Calculation")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to GUI-Odyssey data directory")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Qwen3-VL model")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--num_episodes", type=int, default=1000,
                       help="Number of episodes to sample from training set")
    parser.add_argument("--num_gpus", type=int, default=4,
                       help="Number of GPUs to use")
    parser.add_argument("--mini_batch_size", type=int, default=8,
                       help="Mini-batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of DataLoader workers per GPU")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for episode sampling")

    args = parser.parse_args()

    # 获取实际的物理 GPU IDs
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        physical_gpu_ids = [int(gpu.strip()) for gpu in visible_gpus]
        print(f"检测到 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"将使用物理 GPU: {physical_gpu_ids[:args.num_gpus]}")
    else:
        physical_gpu_ids = list(range(args.num_gpus))
        print(f"未设置 CUDA_VISIBLE_DEVICES，将使用 GPU: {physical_gpu_ids}")

    # 第一阶段：数据预处理
    preprocessor = GuiOdysseyDataPreprocessor(args.data_dir)
    all_samples = preprocessor.prepare_all_samples(
        num_episodes=args.num_episodes,
        random_seed=args.random_seed
    )

    if len(all_samples) == 0:
        print("❌ 没有样本需要处理")
        return

    # 创建临时输出目录
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="guiodyssey_infogain_"))
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

    # 启动多个进程
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
    query_targets = {}  # 保存每个 query 的 targets

    for result in all_results:
        query_id = result["query_id"]

        if query_id not in query_results:
            query_results[query_id] = {}
            query_targets[query_id] = {}  # 初始化 targets 字典

        query_results[query_id][result["perplexity_type"]] = result["perplexity"]

        # 保存 target 信息
        query_targets[query_id][result["perplexity_type"]] = {
            "target": result.get("target", ""),
            "measure_part": result.get("measure_part", "tool_call"),
        }

    # 计算 InfoGain
    print(f"\n{'='*70}")
    print("计算 InfoGain...")
    print(f"{'='*70}")

    final_results = []
    for query_id, perplexities in query_results.items():
        baseline_toolcall = perplexities.get("baseline_toolcall", float('inf'))
        gt_action_toolcall = perplexities.get("gt_action_toolcall", float('inf'))
        gt_thought_action_toolcall = perplexities.get("gt_thought_action_toolcall", float('inf'))

        # 计算 InfoGain
        gt_action_infogain = baseline_toolcall - gt_action_toolcall
        gt_thought_action_infogain = baseline_toolcall - gt_thought_action_toolcall

        final_results.append({
            "query_id": query_id,
            "perplexity_baseline_toolcall": baseline_toolcall,
            "perplexity_gt_action_toolcall": gt_action_toolcall,
            "perplexity_gt_thought_action_toolcall": gt_thought_action_toolcall,
            "gt_action_infogain": gt_action_infogain,
            "gt_thought_action_infogain": gt_thought_action_infogain,
            # 添加 target 信息
            "inference_targets": query_targets.get(query_id, {}),
        })

    # 计算汇总统计
    valid_results = [r for r in final_results if r["gt_action_infogain"] != float('inf')]

    summary = {
        "total_queries": len(final_results),
        "valid_queries": len(valid_results),
    }

    if valid_results:
        summary["mean_baseline_ppl"] = np.mean([r["perplexity_baseline_toolcall"] for r in valid_results])
        summary["mean_gt_action_ppl"] = np.mean([r["perplexity_gt_action_toolcall"] for r in valid_results])
        summary["mean_gt_thought_action_ppl"] = np.mean([r["perplexity_gt_thought_action_toolcall"] for r in valid_results])
        summary["mean_gt_action_infogain"] = np.mean([r["gt_action_infogain"] for r in valid_results])
        summary["mean_gt_thought_action_infogain"] = np.mean([r["gt_thought_action_infogain"] for r in valid_results])
        summary["median_gt_action_infogain"] = np.median([r["gt_action_infogain"] for r in valid_results])
        summary["median_gt_thought_action_infogain"] = np.median([r["gt_thought_action_infogain"] for r in valid_results])
        summary["std_gt_action_infogain"] = np.std([r["gt_action_infogain"] for r in valid_results])
        summary["std_gt_thought_action_infogain"] = np.std([r["gt_thought_action_infogain"] for r in valid_results])

    # 保存结果
    output_dir = Path(args.output_dir) if args.output_dir else Path("./guiodyssey_infogain_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "guiodyssey_infogain_results.json"
    output_data = {
        "summary": summary,
        "detailed_results": final_results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存到: {output_file}")

    # 打印汇总统计
    print(f"\n{'='*70}")
    print("汇总统计")
    print(f"{'='*70}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # 清理临时目录
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"\n✓ 临时目录已清理: {temp_dir}")
    except Exception as e:
        print(f"\n⚠️  清理临时目录失败: {e}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
