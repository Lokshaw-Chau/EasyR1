#!/usr/bin/env python3
"""
计算 InfoGain 的脚本

功能：
1. 用 ground_truth 构造 <tool_call> 格式
2. 用模型进行两次 forward：
   - 只用 <tool_call> 部分（baseline）
   - 用完整内容（包括思考部分）
3. 计算 InfoGain = perplexity_baseline - perplexity_with_thinking
"""

import os
import json
import argparse
import torch
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Tuple
import base64
from io import BytesIO
from PIL import Image


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

class InfoGainCalculator:
    """计算 thinking 内容的 InfoGain"""

    def __init__(self, model_path: str, data_dir: str, device: str = "cuda", mini_batch_size: int = 8):
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.device = device
        self.mini_batch_size = mini_batch_size

        print(f"Loading model from {model_path}...")

        # 加载 processor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # 加载模型 - 使用 Qwen3VLForConditionalGeneration
        from transformers import Qwen3VLForConditionalGeneration

        # 尝试使用 flash-attention，失败则回退到 sdpa
        attn_impl = "flash_attention_2"
        use_compile = False  # torch.compile 与 flash-attention 不兼容

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=attn_impl
            )
            print(f"✓ Model loaded with {attn_impl}")
            print("  Note: torch.compile disabled (incompatible with flash-attention)")
        except Exception as e:
            print(f"⚠️  Flash-attention not available, falling back to SDPA")
            print(f"  Reason: {str(e)[:100]}...")
            attn_impl = "sdpa"
            use_compile = True  # 可以使用 torch.compile
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=attn_impl
            )
            print(f"✓ Model loaded with {attn_impl}")

        self.model.eval()

        # 启用 torch.compile 加速（仅在使用 SDPA 时）
        if use_compile:
            try:
                if hasattr(torch, 'compile') and torch.cuda.is_available():
                    print("Attempting to compile model with torch.compile for acceleration...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("✓ Model compiled successfully!")
            except Exception as e:
                print(f"⚠️  Could not compile model (will use without compilation): {e}")

        print("Model loaded successfully as Qwen3VLForConditionalGeneration")
        print("Model initialization complete!")

    def load_trajectory(self, task_id: str):
        """加载某个task的trajectory"""
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
        """从 ground_truth_actions 构造 <tool_call> 格式的字符串

        参考数据格式（来自 AgentNetBench test_data）：
        {
            "type": "click",  # 或 "write", "terminate", "hotkey" 等
            "params": {
                "position": {"x": 0.4421, "y": 0.2482},  # click 类型
                # 或 "text": "SHANGHAI",  # write 类型
                # 或 "status": "success",  # terminate 类型
                # 或 "keys": ["ctrl", "c"],  # hotkey 类型
            },
            "metadata": {...}
        }
        """
        if not ground_truth_actions:
            return ""

        # 通常只有一个 action
        action = ground_truth_actions[0]
        action_type = action.get("type", "")
        params = action.get("params", {})

        # 映射 action type（AgentNetBench 使用的命名可能与 tool_call 不同）
        # click -> left_click, write -> type, hotkey -> key 等
        type_mapping = {
            "click": "left_click",
            "doubleclick": "double_click",
            "write": "type",
            "hotkey": "key",
            "press": "key",
            "moveto": "mouse_move",
            "dragto": "left_click_drag",
            "rightclick": "right_click",
            "scroll": "scroll",
            "terminate": "terminate"
        }

        mapped_action = type_mapping.get(action_type, action_type)

        # 构造 tool_call JSON
        tool_call_dict = {
            "name": "computer_use",
            "arguments": {
                "action": mapped_action
            }
        }

        # 添加参数（注意：AgentNetBench 用 position，tool_call 用 coordinate）
        if "position" in params:
            position = params["position"]
            # position 格式: {"x": 0.123, "y": 0.456} (相对坐标，范围 0-1)
            # tool_call 需要: [x, y] 列表，像素坐标
            # AgentNetBench 屏幕分辨率是 1000x1000（见 WEB_SYS_PROMPT）
            if isinstance(position, dict):
                x = position.get("x", 0)
                y = position.get("y", 0)
                # 转换为像素坐标（1000x1000 分辨率）
                tool_call_dict["arguments"]["coordinate"] = [
                    int(x * 1000),
                    int(y * 1000)
                ]
            elif isinstance(position, list) and len(position) >= 2:
                tool_call_dict["arguments"]["coordinate"] = position[:2]

        # write -> type 的 text 参数
        if "text" in params:
            tool_call_dict["arguments"]["text"] = params["text"]

        # hotkey/press -> key 的 keys 参数
        if "keys" in params:
            tool_call_dict["arguments"]["keys"] = params["keys"]

        # scroll 的 pixels 参数
        if "pixels" in params:
            tool_call_dict["arguments"]["pixels"] = params["pixels"]

        # terminate 的 status 参数
        if "status" in params:
            tool_call_dict["arguments"]["status"] = params["status"]

        # 构造最终字符串
        tool_call_str = f"<tool_call>\n{json.dumps(tool_call_dict, ensure_ascii=False)}\n</tool_call>"
        return tool_call_str

    def extract_thought_action_toolcall(self, response: str) -> Tuple[str, str, str]:
        """
        从响应中提取 Thought, Action 和 tool_call 三部分

        典型格式：
        Thought: I need to click on the search button.
        Action: Click on the search button.
        <tool_call>
        {...}
        </tool_call>

        Returns:
            (thought_part, action_part, tool_call_part)
        """
        # 提取 <tool_call> 部分
        tool_call_match = re.search(r'<tool_call>.*?</tool_call>', response, re.DOTALL)

        if tool_call_match:
            tool_call_part = tool_call_match.group(0)
            before_tool_call = response[:tool_call_match.start()].strip()

            # 尝试分离 Thought 和 Action
            # 查找 "Thought:" 和 "Action:" 标记
            thought_pattern = r'(?:Thought|THOUGHT|thought):\s*(.*?)(?=(?:Action|ACTION|action):|$)'
            action_pattern = r'(?:Action|ACTION|action):\s*(.*?)$'

            thought_match = re.search(thought_pattern, before_tool_call, re.DOTALL | re.IGNORECASE)
            action_match = re.search(action_pattern, before_tool_call, re.DOTALL | re.IGNORECASE)

            if thought_match and action_match:
                # 同时有 Thought 和 Action
                thought_part = "Thought: " + thought_match.group(1).strip()
                action_part = "Action: " + action_match.group(1).strip()
            elif thought_match:
                # 只有 Thought
                thought_part = "Thought: " + thought_match.group(1).strip()
                action_part = ""
            elif action_match:
                # 只有 Action
                thought_part = ""
                action_part = "Action: " + action_match.group(1).strip()
            else:
                # 没有明确标记，整个算作 Thought
                thought_part = before_tool_call
                action_part = ""

            return thought_part, action_part, tool_call_part
        else:
            # 如果没有 <tool_call>，整个都算 thinking
            return response, "", ""

    def calculate_perplexity_batch(self, prompts: List[str], targets: List[str],
                                   images: List[Image] = None) -> List[float]:
        """
        批量计算困惑度（真正的批处理实现）

        每个样本复制一份图片，实现真正的批处理推理

        Args:
            prompts: 输入 prompt 列表
            targets: 目标文本列表
            images: 可选的图像列表（可以是同一张图）

        Returns:
            困惑度列表
        """
        try:
            batch_size = len(prompts)
            if batch_size == 0:
                return []

            # 如果没有提供 images，创建 None 列表
            if images is None:
                images = [None] * batch_size
            elif isinstance(images, Image.Image):
                # 如果只提供了一张图，复制给所有样本（每个样本独立的副本）
                images = [images] * batch_size

            # 分成小批量处理
            all_perplexities = []

            for start_idx in range(0, batch_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, batch_size)

                # 处理这个 mini-batch
                mini_prompts = prompts[start_idx:end_idx]
                mini_targets = targets[start_idx:end_idx]
                mini_images = images[start_idx:end_idx]

                # 真正的批处理
                mini_perplexities = self._calculate_perplexity_batch_internal(
                    mini_prompts, mini_targets, mini_images
                )
                all_perplexities.extend(mini_perplexities)

            return all_perplexities

        except Exception as e:
            print(f"Error in calculate_perplexity_batch: {e}")
            import traceback
            traceback.print_exc()
            return [float('inf')] * len(prompts)

    def _calculate_perplexity_batch_internal(self, prompts: List[str], targets: List[str],
                                            images: List[Image]) -> List[float]:
        """
        内部批处理实现（处理一个 mini-batch）

        Args:
            prompts: prompt 列表
            targets: target 列表
            images: 图像列表（每个样本一张）

        Returns:
            困惑度列表
        """
        try:
            batch_size = len(prompts)

            # 为每个样本构造 messages 并 tokenize
            batch_inputs_list = []
            batch_prompt_inputs_list = []

            for i in range(batch_size):
                full_text = prompts[i] + targets[i]

                # 完整输入的 messages（每个样本独立的图片副本）
                content = []
                if images[i] is not None:
                    content.append({"type": "image", "image": images[i]})
                content.append({"type": "text", "text": full_text})
                messages = [{"role": "user", "content": content}]

                # Tokenize 完整输入
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs.pop("token_type_ids", None)
                batch_inputs_list.append(inputs)

                # Tokenize prompt 部分（用于计算长度）
                prompt_content = []
                if images[i] is not None:
                    prompt_content.append({"type": "image", "image": images[i]})
                prompt_content.append({"type": "text", "text": prompts[i]})
                prompt_messages = [{"role": "user", "content": prompt_content}]

                prompt_inputs = self.processor.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )
                prompt_inputs.pop("token_type_ids", None)
                batch_prompt_inputs_list.append(prompt_inputs)

            # 计算每个样本的 prompt 长度
            prompt_lengths = [inp["input_ids"].shape[1] for inp in batch_prompt_inputs_list]

            # 找到最大长度
            max_length = max(inp["input_ids"].shape[1] for inp in batch_inputs_list)

            # Pad 并组合成 batch
            padded_input_ids = []
            padded_attention_mask = []
            all_pixel_values = []
            all_image_grid_thw = []

            for i, inp in enumerate(batch_inputs_list):
                input_ids = inp["input_ids"][0]
                attention_mask = inp.get("attention_mask", torch.ones_like(input_ids))[0]

                # 左侧 padding
                pad_length = max_length - len(input_ids)
                if pad_length > 0:
                    input_ids = torch.cat([
                        torch.full((pad_length,), self.processor.tokenizer.pad_token_id, dtype=input_ids.dtype),
                        input_ids
                    ])
                    attention_mask = torch.cat([
                        torch.zeros(pad_length, dtype=attention_mask.dtype),
                        attention_mask
                    ])

                padded_input_ids.append(input_ids)
                padded_attention_mask.append(attention_mask)

                # 收集图像张量（每个样本一份）
                if "pixel_values" in inp:
                    all_pixel_values.append(inp["pixel_values"])
                if "image_grid_thw" in inp:
                    all_image_grid_thw.append(inp["image_grid_thw"])

            # 堆叠文本输入
            batch_inputs = {
                "input_ids": torch.stack(padded_input_ids).to(self.device),
                "attention_mask": torch.stack(padded_attention_mask).to(self.device),
            }

            # 处理图像：concatenate 所有图像（因为每个样本都有自己的图）
            if all_pixel_values:
                # pixel_values: 每个是 [num_patches, channels, height, width]
                # 需要 concatenate 成 [total_patches, channels, height, width]
                batch_inputs["pixel_values"] = torch.cat(all_pixel_values, dim=0).to(self.device)

            if all_image_grid_thw:
                # image_grid_thw: 每个是 [num_images, 3]
                # concatenate 成 [total_images, 3]
                batch_inputs["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0).to(self.device)

            # 调整 prompt_lengths（考虑 padding）
            adjusted_prompt_lengths = [
                pl + (max_length - batch_inputs_list[i]["input_ids"].shape[1])
                for i, pl in enumerate(prompt_lengths)
            ]

            # Batch forward pass
            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                # 对每个样本计算困惑度
                perplexities = []
                for i in range(batch_size):
                    prompt_length = adjusted_prompt_lengths[i]
                    total_length = logits.shape[1]
                    target_length = total_length - prompt_length

                    if target_length <= 0:
                        print(f"Warning: sample {i} target_length <= 0 ({target_length})")
                        perplexities.append(float('inf'))
                        continue

                    # 计算该样本的 loss
                    shift_logits = logits[i:i+1, prompt_length-1:-1, :].contiguous()
                    shift_labels = batch_inputs["input_ids"][i:i+1, prompt_length:].contiguous()

                    # 只计算非 padding 部分
                    mask = batch_inputs["attention_mask"][i:i+1, prompt_length:].contiguous()

                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                    # 应用 mask 并计算平均
                    loss = loss.view(shift_labels.shape)
                    masked_loss = (loss * mask.float()).sum() / mask.float().sum()

                    perplexity = torch.exp(masked_loss).item()
                    perplexities.append(perplexity)

            return perplexities

        except Exception as e:
            print(f"Error in _calculate_perplexity_batch_internal: {e}")
            import traceback
            traceback.print_exc()
            # 失败时回退到逐个计算
            print(f"Falling back to sequential processing for this mini-batch...")
            perplexities = []
            for i in range(len(prompts)):
                perplexity = self.calculate_perplexity(
                    prompt=prompts[i],
                    target=targets[i],
                    image=images[i]
                )
                perplexities.append(perplexity)
            return perplexities

    def calculate_perplexity(self, prompt: str, target: str, image: Image = None) -> float:
        """
        计算给定 target 的困惑度

        Args:
            prompt: 输入 prompt
            target: 需要计算困惑度的目标文本
            image: 可选的图像输入

        Returns:
            困惑度值
        """
        try:
            # 构建完整的输入文本（prompt + target）
            full_text = prompt + target

            # 构造 messages 格式（用于完整输入）
            content = []
            if image is not None:
                # 添加图像（PIL Image对象）
                content.append({
                    "type": "image",
                    "image": image  # 直接传PIL对象
                })

            # 添加文本
            content.append({
                "type": "text",
                "text": full_text
            })

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            # 使用 apply_chat_template 处理
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,  # 不添加生成提示，因为我们要计算困惑度
                return_dict=True,
                return_tensors="pt",
            )

            # 移除 token_type_ids（如果存在）
            inputs.pop("token_type_ids", None)

            # 移动到设备
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

            # 同样处理 prompt 部分来计算长度
            prompt_content = []
            if image is not None:
                prompt_content.append({
                    "type": "image",
                    "image": image
                })
            prompt_content.append({
                "type": "text",
                "text": prompt
            })

            prompt_messages = [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]

            prompt_inputs = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_inputs.pop("token_type_ids", None)

            # 计算 target 部分的长度
            prompt_length = prompt_inputs["input_ids"].shape[1]
            total_length = inputs["input_ids"].shape[1]
            target_length = total_length - prompt_length

            if target_length <= 0:
                print(f"Warning: target_length <= 0 ({target_length}), returning inf")
                return float('inf')

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # 只计算 target 部分的 loss
                # shift: logits[:-1] 预测 labels[1:]
                shift_logits = logits[:, prompt_length-1:-1, :].contiguous()
                shift_labels = inputs["input_ids"][:, prompt_length:].contiguous()

                # 计算 cross entropy loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # 困惑度 = exp(loss)
                perplexity = torch.exp(loss).item()

            return perplexity

        except Exception as e:
            print(f"Error in calculate_perplexity: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')

    def calculate_infogain_for_query_batch(self, query_result: Dict, trajectory: Dict, step_num: int) -> List[Dict]:
        """
        批量计算一个 query 的所有 samples 的 InfoGain（加速版本）

        Args:
            query_result: query 的完整信息
            trajectory: trajectory 数据
            step_num: 步骤编号

        Returns:
            所有 samples 的 InfoGain 结果列表
        """
        # 获取 ground truth
        step_data = trajectory["steps"][step_num]
        ground_truth_actions = step_data.get("ground_truth_actions", [])

        # 构造 ground truth 的 <tool_call>
        gt_tool_call = self.construct_tool_call_from_gt(ground_truth_actions)

        if not gt_tool_call:
            # 返回所有 samples 的错误结果
            return [{
                "sample_id": sample["sample_id"],
                "action_infogain": 0.0,
                "thought_infogain": 0.0,
                "total_infogain": 0.0,
                "perplexity_baseline": 0.0,
                "perplexity_with_action": 0.0,
                "perplexity_with_thought_action": 0.0,
                "error": "No ground truth tool call"
            } for sample in query_result["samples"]]

        # 加载图像（所有 samples 共享同一张图）
        task_id = query_result.get("task_id", query_result["query_id"].rsplit("-", 1)[0])
        image = self.load_image(task_id, step_num)

        # 构造 prompt（所有 samples 共享）
        instruction = query_result.get("instruction", "")
        history = query_result.get("history", "None")
        prompt = (
            f"<image>\nThe user query: {instruction}\n"
            f"Task progress (You have done the following operation on the current device): {history}\n"
        )

        try:
            # 为所有 samples 提取 Thought, Action 和 tool_call
            samples_info = []
            for sample in query_result["samples"]:
                response = sample["response"]
                thought_part, action_part, tool_call_part = self.extract_thought_action_toolcall(response)
                samples_info.append({
                    "sample_id": sample["sample_id"],
                    "thought_part": thought_part,
                    "action_part": action_part,
                    "tool_call_part": tool_call_part
                })

            # 收集所有需要计算的 (prompt, target) 对
            # 每个 sample 需要计算 3 个困惑度：baseline, with_action, with_thought_action
            num_samples = len(samples_info)

            # 构建 batch
            batch_prompts = []
            batch_targets = []
            batch_types = []  # 记录每个是 baseline/with_action/with_thought_action

            for sample_info in samples_info:
                thought_part = sample_info["thought_part"]
                action_part = sample_info["action_part"]

                # 1. Baseline
                batch_prompts.append(prompt)
                batch_targets.append(gt_tool_call)
                batch_types.append("baseline")

                # 2. With Action
                action_and_tool_call = (action_part + "\n" + gt_tool_call) if action_part else gt_tool_call
                batch_prompts.append(prompt)
                batch_targets.append(action_and_tool_call)
                batch_types.append("with_action")

                # 3. With Thought+Action
                full_target = ""
                if thought_part:
                    full_target += thought_part + "\n"
                if action_part:
                    full_target += action_part + "\n"
                full_target += gt_tool_call

                batch_prompts.append(prompt)
                batch_targets.append(full_target)
                batch_types.append("with_thought_action")

            # 批量计算所有困惑度（一次 forward pass）
            batch_perplexities = self.calculate_perplexity_batch(
                batch_prompts,
                batch_targets,
                images=image  # 所有样本共享同一张图
            )

            # 将结果分配回每个 sample
            results = []
            for i, sample_info in enumerate(samples_info):
                # 每个 sample 对应 3 个困惑度值
                idx_base = i * 3
                perplexity_baseline = batch_perplexities[idx_base]
                perplexity_with_action = batch_perplexities[idx_base + 1]
                perplexity_with_thought_action = batch_perplexities[idx_base + 2]

                # 计算 InfoGain
                action_infogain = perplexity_baseline - perplexity_with_action
                thought_infogain = perplexity_with_action - perplexity_with_thought_action
                total_infogain = perplexity_baseline - perplexity_with_thought_action

                results.append({
                    "sample_id": sample_info["sample_id"],
                    "action_infogain": action_infogain,
                    "thought_infogain": thought_infogain,
                    "total_infogain": total_infogain,
                    "perplexity_baseline": perplexity_baseline,
                    "perplexity_with_action": perplexity_with_action,
                    "perplexity_with_thought_action": perplexity_with_thought_action,
                    "thought_length": len(sample_info["thought_part"]),
                    "action_length": len(sample_info["action_part"]),
                    "thought_content": sample_info["thought_part"][:200] if sample_info["thought_part"] else "",
                    "action_content": sample_info["action_part"][:200] if sample_info["action_part"] else ""
                })

            return results

        except Exception as e:
            print(f"Error in calculate_infogain_for_query_batch: {e}")
            import traceback
            traceback.print_exc()
            # 返回所有 samples 的错误结果
            return [{
                "sample_id": sample["sample_id"],
                "action_infogain": 0.0,
                "thought_infogain": 0.0,
                "total_infogain": 0.0,
                "perplexity_baseline": 0.0,
                "perplexity_with_action": 0.0,
                "perplexity_with_thought_action": 0.0,
                "error": str(e)
            } for sample in query_result["samples"]]

    def calculate_infogain_for_sample(self, query_result: Dict, sample: Dict,
                                      trajectory: Dict, step_num: int) -> Dict:
        """
        计算单个 sample 的 InfoGain

        分别计算：
        1. Baseline: 只用 <tool_call>
        2. With Action: Action + <tool_call>
        3. With Thought+Action: Thought + Action + <tool_call>

        InfoGain 指标：
        - action_infogain: 从 baseline 到 with_action 的困惑度降低
        - thought_infogain: 从 with_action 到 with_thought_action 的困惑度降低
        - total_infogain: 从 baseline 到 with_thought_action 的总困惑度降低

        Args:
            query_result: query 的完整信息
            sample: 单个采样结果
            trajectory: trajectory 数据
            step_num: 步骤编号

        Returns:
            InfoGain 相关指标
        """
        # 获取 ground truth
        step_data = trajectory["steps"][step_num]
        ground_truth_actions = step_data.get("ground_truth_actions", [])

        # 构造 ground truth 的 <tool_call>
        gt_tool_call = self.construct_tool_call_from_gt(ground_truth_actions)

        if not gt_tool_call:
            return {
                "action_infogain": 0.0,
                "thought_infogain": 0.0,
                "total_infogain": 0.0,
                "perplexity_baseline": 0.0,
                "perplexity_with_action": 0.0,
                "perplexity_with_thought_action": 0.0,
                "error": "No ground truth tool call"
            }

        # 从 sample 的响应中提取 Thought, Action 和 tool_call 三部分
        response = sample["response"]
        thought_part, action_part, tool_call_part = self.extract_thought_action_toolcall(response)

        # 加载图像
        task_id = query_result.get("task_id", query_result["query_id"].rsplit("-", 1)[0])
        image = self.load_image(task_id, step_num)

        # 构造 prompt
        instruction = query_result.get("instruction", "")
        history = query_result.get("history", "None")

        prompt = (
                f"<image>\nThe user query: {instruction}\n"
                f"Task progress (You have done the following operation on the current device): {history}\n"
            )

        try:
            # 计算三种情况的困惑度
            # 1. Baseline: 只用 <tool_call>（没有 thinking 和 action）
            perplexity_baseline = self.calculate_perplexity(
                prompt=prompt,
                target=gt_tool_call,
                image=image
            )

            # 2. With Action: Action + <tool_call>
            action_and_tool_call = (action_part + "\n" + gt_tool_call) if action_part else gt_tool_call
            perplexity_with_action = self.calculate_perplexity(
                prompt=prompt,
                target=action_and_tool_call,
                image=image
            )

            # 3. With Thought+Action: Thought + Action + <tool_call>
            full_target = ""
            if thought_part:
                full_target += thought_part + "\n"
            if action_part:
                full_target += action_part + "\n"
            full_target += gt_tool_call

            perplexity_with_thought_action = self.calculate_perplexity(
                prompt=prompt,
                target=full_target,
                image=image
            )

            # 计算 InfoGain
            # Action 的 InfoGain: 添加 Action 描述降低了多少困惑度
            action_infogain = perplexity_baseline - perplexity_with_action

            # Thought 的额外 InfoGain: 在有 Action 的基础上，添加 Thought 又降低了多少困惑度
            thought_infogain = perplexity_with_action - perplexity_with_thought_action

            # 总 InfoGain: 从 baseline 到完整内容的总降低量
            total_infogain = perplexity_baseline - perplexity_with_thought_action

            return {
                "action_infogain": action_infogain,
                "thought_infogain": thought_infogain,
                "total_infogain": total_infogain,
                "perplexity_baseline": perplexity_baseline,
                "perplexity_with_action": perplexity_with_action,
                "perplexity_with_thought_action": perplexity_with_thought_action,
                "thought_length": len(thought_part),
                "action_length": len(action_part),
                "thought_content": thought_part[:200] if thought_part else "",
                "action_content": action_part[:200] if action_part else ""
            }

        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            import traceback
            traceback.print_exc()
            return {
                "action_infogain": 0.0,
                "thought_infogain": 0.0,
                "total_infogain": 0.0,
                "perplexity_baseline": 0.0,
                "perplexity_with_action": 0.0,
                "perplexity_with_thought_action": 0.0,
                "error": str(e)
            }

    def analyze_sampling_file(self, sampling_file: Path, output_file: Path = None, use_batch: bool = True):
        """分析采样文件并计算 InfoGain

        Args:
            sampling_file: 采样结果文件路径
            output_file: 输出文件路径（可选）
            use_batch: 是否使用批量推理加速（默认 True）
        """
        print(f"\n{'='*70}")
        print(f"Calculating InfoGain for: {sampling_file.name}")
        if use_batch:
            print(f"Mode: Batch processing (mini-batch size: {self.mini_batch_size})")
        else:
            print("Mode: Sequential processing")
        print(f"{'='*70}\n")

        # 读取采样结果
        with open(sampling_file, "r") as f:
            sampling_results = [json.loads(line) for line in f]

        all_infogain_results = []

        for query_result in tqdm(sampling_results, desc="Calculating InfoGain"):
            query_id = query_result["query_id"]

            # 解析 query_id
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

            # 批量或顺序计算所有 samples 的 InfoGain
            if use_batch:
                # 批量计算（加速版本）
                sample_infogains = self.calculate_infogain_for_query_batch(
                    query_result, trajectory, step_num
                )
            else:
                # 顺序计算（兼容版本）
                sample_infogains = []
                for sample in query_result["samples"]:
                    infogain_result = self.calculate_infogain_for_sample(
                        query_result, sample, trajectory, step_num
                    )
                    sample_infogains.append({
                        "sample_id": sample["sample_id"],
                        **infogain_result
                    })

            # 计算平均 InfoGain（三种）
            valid_samples = [s for s in sample_infogains if "error" not in s]
            avg_action_infogain = np.mean([s["action_infogain"] for s in valid_samples]) if valid_samples else 0.0
            avg_thought_infogain = np.mean([s["thought_infogain"] for s in valid_samples]) if valid_samples else 0.0
            avg_total_infogain = np.mean([s["total_infogain"] for s in valid_samples]) if valid_samples else 0.0

            all_infogain_results.append({
                "query_id": query_id,
                "task_id": task_id,
                "step_num": step_num,
                "sample_infogains": sample_infogains,
                "avg_action_infogain": avg_action_infogain,
                "avg_thought_infogain": avg_thought_infogain,
                "avg_total_infogain": avg_total_infogain
            })

        # 计算总体统计
        all_action_infogains = []
        all_thought_infogains = []
        all_total_infogains = []
        all_baseline_perplexities = []
        all_action_perplexities = []
        all_thought_action_perplexities = []

        for result in all_infogain_results:
            for sample_ig in result["sample_infogains"]:
                if "error" not in sample_ig:
                    all_action_infogains.append(sample_ig["action_infogain"])
                    all_thought_infogains.append(sample_ig["thought_infogain"])
                    all_total_infogains.append(sample_ig["total_infogain"])
                    all_baseline_perplexities.append(sample_ig["perplexity_baseline"])
                    all_action_perplexities.append(sample_ig["perplexity_with_action"])
                    all_thought_action_perplexities.append(sample_ig["perplexity_with_thought_action"])

        summary_stats = {
            "total_queries": len(all_infogain_results),
            "total_samples": len(all_action_infogains),
            # Action InfoGain 统计
            "avg_action_infogain": float(np.mean(all_action_infogains)) if all_action_infogains else 0.0,
            "median_action_infogain": float(np.median(all_action_infogains)) if all_action_infogains else 0.0,
            "std_action_infogain": float(np.std(all_action_infogains)) if all_action_infogains else 0.0,
            # Thought InfoGain 统计
            "avg_thought_infogain": float(np.mean(all_thought_infogains)) if all_thought_infogains else 0.0,
            "median_thought_infogain": float(np.median(all_thought_infogains)) if all_thought_infogains else 0.0,
            "std_thought_infogain": float(np.std(all_thought_infogains)) if all_thought_infogains else 0.0,
            # Total InfoGain 统计
            "avg_total_infogain": float(np.mean(all_total_infogains)) if all_total_infogains else 0.0,
            "median_total_infogain": float(np.median(all_total_infogains)) if all_total_infogains else 0.0,
            "std_total_infogain": float(np.std(all_total_infogains)) if all_total_infogains else 0.0,
            # 困惑度统计
            "avg_perplexity_baseline": float(np.mean(all_baseline_perplexities)) if all_baseline_perplexities else 0.0,
            "avg_perplexity_with_action": float(np.mean(all_action_perplexities)) if all_action_perplexities else 0.0,
            "avg_perplexity_with_thought_action": float(np.mean(all_thought_action_perplexities)) if all_thought_action_perplexities else 0.0,
            # 正向比例
            "percentage_positive_action_ig": float(sum(1 for ig in all_action_infogains if ig > 0) / len(all_action_infogains) * 100) if all_action_infogains else 0.0,
            "percentage_positive_thought_ig": float(sum(1 for ig in all_thought_infogains if ig > 0) / len(all_thought_infogains) * 100) if all_thought_infogains else 0.0,
            "percentage_positive_total_ig": float(sum(1 for ig in all_total_infogains if ig > 0) / len(all_total_infogains) * 100) if all_total_infogains else 0.0
        }

        # 打印结果
        print(f"\n{'='*70}")
        print("INFOGAIN ANALYSIS RESULTS")
        print(f"{'='*70}")
        print(f"Total queries: {summary_stats['total_queries']}")
        print(f"Total samples: {summary_stats['total_samples']}")
        print(f"\n📊 Action InfoGain Statistics:")
        print(f"  Average:  {summary_stats['avg_action_infogain']:.4f}")
        print(f"  Median:   {summary_stats['median_action_infogain']:.4f}")
        print(f"  Std Dev:  {summary_stats['std_action_infogain']:.4f}")
        print(f"  Positive: {summary_stats['percentage_positive_action_ig']:.2f}% of samples")
        print(f"\n💭 Thought InfoGain Statistics:")
        print(f"  Average:  {summary_stats['avg_thought_infogain']:.4f}")
        print(f"  Median:   {summary_stats['median_thought_infogain']:.4f}")
        print(f"  Std Dev:  {summary_stats['std_thought_infogain']:.4f}")
        print(f"  Positive: {summary_stats['percentage_positive_thought_ig']:.2f}% of samples")
        print(f"\n🎯 Total InfoGain Statistics:")
        print(f"  Average:  {summary_stats['avg_total_infogain']:.4f}")
        print(f"  Median:   {summary_stats['median_total_infogain']:.4f}")
        print(f"  Std Dev:  {summary_stats['std_total_infogain']:.4f}")
        print(f"  Positive: {summary_stats['percentage_positive_total_ig']:.2f}% of samples")
        print(f"\n📉 Perplexity:")
        print(f"  Baseline (no thinking/action):      {summary_stats['avg_perplexity_baseline']:.4f}")
        print(f"  With Action only:                   {summary_stats['avg_perplexity_with_action']:.4f}")
        print(f"  With Thought + Action:              {summary_stats['avg_perplexity_with_thought_action']:.4f}")
        print(f"\n💡 Interpretation:")
        print(f"  Action contribution: {summary_stats['avg_perplexity_baseline'] - summary_stats['avg_perplexity_with_action']:.4f} reduction")
        print(f"  Thought contribution: {summary_stats['avg_perplexity_with_action'] - summary_stats['avg_perplexity_with_thought_action']:.4f} reduction")
        print(f"  Total reduction: {summary_stats['avg_perplexity_baseline'] - summary_stats['avg_perplexity_with_thought_action']:.4f}")
        print(f"{'='*70}\n")

        # 保存结果
        if output_file is None:
            output_file = sampling_file.parent / "analysis" / f"{sampling_file.stem}_infogain.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "summary": summary_stats,
            "detailed_results": all_infogain_results
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"InfoGain results saved to: {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Calculate InfoGain for multi-sampling results")
    parser.add_argument("--sampling_file", type=str, required=True,
                       help="Path to the sampling results file (.jsonl)")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to the test data directory")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model for calculating perplexity")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for InfoGain results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--use_batch", action="store_true", default=True,
                       help="Use batch processing for acceleration (default: True)")
    parser.add_argument("--no_batch", action="store_false", dest="use_batch",
                       help="Disable batch processing (use sequential mode)")
    parser.add_argument("--mini_batch_size", type=int, default=8,
                       help="Mini-batch size for batch processing (default: 8)")

    args = parser.parse_args()

    # 创建计算器
    calculator = InfoGainCalculator(args.model_path, args.data_dir, args.device,
                                   mini_batch_size=args.mini_batch_size)

    # 计算 InfoGain
    sampling_file = Path(args.sampling_file)
    output_file = Path(args.output_file) if args.output_file else None

    calculator.analyze_sampling_file(sampling_file, output_file, use_batch=args.use_batch)


if __name__ == "__main__":
    main()
