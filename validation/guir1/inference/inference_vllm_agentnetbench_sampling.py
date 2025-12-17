import os
import json
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import ray
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import re
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from datasets import Dataset as hf_dataset
from qwen_vl_utils import smart_resize
import copy
import base64
import time

# 初始化 Ray
def initialize_ray():
    """Initialize Ray with proper error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # First try to connect to existing cluster
            ray.init(address='auto')
            print("Connected to existing Ray cluster")
            return True
        except Exception as e1:
            print(f"Failed to connect to existing cluster (attempt {attempt + 1}): {e1}")
            try:
                # If that fails, shutdown any existing Ray and start new
                ray.shutdown()
                time.sleep(1)
                ray.init()
                print("Started new Ray cluster")
                return True
            except Exception as e2:
                print(f"Failed to start new Ray cluster (attempt {attempt + 1}): {e2}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"Warning: Could not initialize Ray after {max_retries} attempts")
                    print("You may need to manually clean up Ray processes and restart")
                    return False

# Initialize Ray
if not initialize_ray():
    print("Continuing without Ray - this may cause issues with distributed processing")
    exit(1)

# 模型路径
MODEL_PATH = ""

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


# 数据路径
DATA_PATH = ""

# 微批大小
MICRO_BATCH = 4


class MultiModalDataset(Dataset):
    def __init__(self, data, processor, prefix=None):
        self.data = data
        self.processor = processor
        self.processor.max_pixels=6400*32*32
        self.prefix = prefix if prefix else ""

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"]
        dummy_image = Image.open(BytesIO(image["bytes"]))
        text = sample["instruction"]
        history="None" if 'history' not in sample else sample['history']

        user_query = (
                f"<image>\nThe user query: {text}\n"
                f"Task progress (You have done the following operation on the current device): {history}\n"
            )
        resized_height, resized_width  = smart_resize(dummy_image.height,
            dummy_image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=self.processor.image_processor.max_pixels,)

        img_bytes = image["bytes"]
        b64_str = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64_str}"

        message=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": WEB_SYS_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": data_uri,
                    },
                    {"type": "text", "text": user_query},
                ],
            }
        ]

        # 生成推理所需的 prompt 和多模态输入
        prompt = self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt += self.prefix

        image_inputs, video_inputs, video_kwargs = process_vision_info(message, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

        inputs = self.processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    do_resize=False
                )

        resized_height = inputs['image_grid_thw'][0][1] * self.processor.image_processor.patch_size
        resized_width = inputs['image_grid_thw'][0][2] * self.processor.image_processor.patch_size

        origin_height = image_inputs[0].size[1]
        origin_width = image_inputs[0].size[0]
        scale_x = origin_width / resized_width
        scale_y = origin_height / resized_height

        del inputs

        sample["scale"]=[scale_x.item(),scale_y.item()]
        sample["image_size"]=[origin_width,origin_height]

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
            "original_sample": sample,
        }


def custom_collate_fn(batch):
    collated_batch = {
        "prompts": [],
        "multi_modal_data": [],
        "mm_processor_kwargs": [],
        "original_samples": [],
    }
    for item in batch:
        collated_batch["prompts"].append(item["prompt"])
        collated_batch["multi_modal_data"].append(item["multi_modal_data"])
        collated_batch["mm_processor_kwargs"].append(item["mm_processor_kwargs"])
        collated_batch["original_samples"].append(item["original_sample"])
    return collated_batch


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, model_path, sampling_params, num_samples, output_path=None):
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 1, "video": 1},
            max_model_len=120000,
            gpu_memory_utilization=0.9,
        )
        self.sampling_params = sampling_params
        self.num_samples = num_samples
        self.output_path = output_path

    def process_data(self, dataloader):
        all_results = []

        for batch in tqdm(dataloader):
            prompts = batch["prompts"]
            multi_modal_data = batch["multi_modal_data"]
            mm_processor_kwargs = batch["mm_processor_kwargs"]
            original_samples = batch["original_samples"]

            # 对每个样本进行处理
            for idx, (prompt, mm_data, mm_kwargs, original_sample) in enumerate(
                zip(prompts, multi_modal_data, mm_processor_kwargs, original_samples)
            ):
                # 为当前query准备结果字典
                query_result = {
                    "query_id": original_sample.get("uid", f"query_{idx}"),
                    "instruction": original_sample.get("instruction", ""),
                    "history": original_sample.get("history", "None"),
                    "image_size": original_sample["image_size"],
                    "scale": original_sample["scale"],
                    "samples": []  # 存储n次采样结果
                }

                print(f"\n{'='*60}")
                print(f"Sampling {self.num_samples} times for query: {query_result['instruction'][:50]}...")
                print(f"{'='*60}")

                # 准备输入
                llm_input = {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": mm_kwargs,
                }

                # 一次性生成 n 个样本（通过 SamplingParams 的 n 参数）
                outputs = self.llm.generate([llm_input], sampling_params=self.sampling_params)

                # outputs[0] 是一个 RequestOutput，包含多个 CompletionOutput（n个样本）
                request_output = outputs[0]

                # 遍历所有采样结果
                for sample_idx, completion_output in enumerate(request_output.outputs):
                    generated_text = completion_output.text

                    # 记录单次采样结果
                    sample_result = {
                        "sample_id": sample_idx,
                        "response": generated_text,
                        "finish_reason": completion_output.finish_reason,
                    }

                    query_result["samples"].append(sample_result)

                    print(f"\n--- Sample {sample_idx + 1}/{self.num_samples} ---")
                    print(f"Response: {generated_text[:200]}...")
                    print(f"Finish reason: {completion_output.finish_reason}")

                all_results.append(query_result)

                # 实时保存结果（追加模式）
                if self.output_path:
                    print(f"\nWriting result to {self.output_path}")
                    with open(self.output_path, "a") as f:
                        f.write(json.dumps(query_result, ensure_ascii=False) + "\n")

        return all_results


def main(args):
    # 将数据分成多份
    MODEL_PATH = args.model_path
    DATA_PATH = args.data_path

    if DATA_PATH.endswith('.parquet'):
        data = load_dataset("parquet", data_files=DATA_PATH, split="train")
    else:
        data = [json.loads(s) for s in open(DATA_PATH, "r")] if DATA_PATH.endswith(".jsonl") else json.load(open(DATA_PATH,"r"))

    # 输出路径
    OUTPUT_DIR = args.output_path
    num_actors = args.num_actor
    num_samples = args.num_samples

    OUTPUT_DIR = os.path.join(OUTPUT_DIR, MODEL_PATH.split('/')[-1])

    # 文件名包含采样次数信息
    prefix_str = args.prefix if args.prefix else "None"
    NEW_FILE = os.path.join(
        OUTPUT_DIR,
        DATA_PATH.split("/")[-1]
        .replace(".jsonl", f"_sampling{num_samples}_{prefix_str}.jsonl")
        .replace('.parquet', f'_sampling{num_samples}_{prefix_str}.jsonl')
    )

    print(f"\n{'='*60}")
    print(f"Output file: {NEW_FILE}")
    print(f"Number of samples per query: {num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 清空输出文件（如果存在）
    if os.path.exists(NEW_FILE):
        print(f"Removing existing file: {NEW_FILE}")
        os.remove(NEW_FILE)

    # 数据分片
    data_chunks = [hf_dataset.from_dict(data[i::num_actors]) for i in range(num_actors)]

    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 创建采样参数（使用非零temperature以支持多样性采样）
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        n=args.num_samples,
        stop_token_ids=[],
    )

    # 创建多个 Worker
    workers = [
        Worker.remote(MODEL_PATH, sampling_params, num_samples, NEW_FILE)
        for _ in range(num_actors)
    ]

    # 使用 PyTorch Dataset 和 DataLoader
    futures = []
    for i, chunk in enumerate(data_chunks):
        dataset = MultiModalDataset(chunk, processor, args.prefix)
        dataloader = DataLoader(
            dataset,
            batch_size=MICRO_BATCH,
            shuffle=False,
            num_workers=16,
            collate_fn=custom_collate_fn
        )
        futures.append(workers[i].process_data.remote(dataloader))

    # 收集所有结果
    all_results = ray.get(futures)

    print(f"\n{'='*60}")
    print(f"All sampling completed!")
    print(f"Total queries processed: {sum(len(worker_results) for worker_results in all_results)}")
    print(f"Results saved to: {NEW_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-sampling inference for AgentNetBench")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--output_path', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--num_actor', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of samples per query')
    parser.add_argument('--prefix', type=str, default=None, help='Prefix to add to prompts')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (0.0 = greedy)')
    parser.add_argument('--top_p', type=float, default=0.99, help='Top-p sampling')
    parser.add_argument('--repetition_penalty', type=float, default=1.05, help='Repetition penalty')
    parser.add_argument('--max_tokens', type=int, default=1024, help='Maximum tokens to generate')

    args = parser.parse_args()
    main(args)
