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

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from qwen_vl_utils import smart_resize
from utils.agent_function_call import GUIAction
import copy
import base64
# 初始化 Ray
ray.init()

# 模型路径
MODEL_PATH = ""

# 推理参数
# SAMPLING_PARAMS = SamplingParams(
#     temperature=0.0,
#     top_p=0.001,
#     repetition_penalty=1.05,
#     max_tokens=1024,  # 根据需要调整最大生成长度
#     stop_token_ids=[],  # 停止标志
# )

SAMPLING_PARAMS = SamplingParams(
    temperature=1.0,
    top_p=0.7,
    repetition_penalty=1.05,
    max_tokens=1024,  # 根据需要调整最大生成长度
    n=256,
    stop_token_ids=[],  # 停止标志
)

# 数据路径
DATA_PATH = ""

# 微批大小
MICRO_BATCH = 2

def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    # answer_tag_pattern = r'<tool_call>(.*?)</tool_call>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    # content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        # if content_answer_match:
        #     content_answer = content_answer_match.group(1).strip()
        coord_match = re.search(bbox_pattern, content)
        if coord_match:
            coord = [int(coord_match.group(1)), int(coord_match.group(2))]
            return coord, True
        else:
            coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False
    
class MultiModalDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
        self.processor.max_pixels=1258291

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """返回单个样本，包含预处理后的数据"""
        sample = self.data[idx]
        image = sample["image"]
        # image = Image.open(BytesIO(image["bytes"]))
        dummy_image = Image.open(BytesIO(image["bytes"]))
        # image = copy.deepcopy(dummy_image)
        text = sample["instruction"]

        user_query = (
            f"<image>\nThe user query: {text}\n"
            "Output the thinking process in <think></think> tags, and the function call in <tool_call></tool_call> tags as follows:\n"
            "<think> ... </think> <tool_call>{\"name\": \"gui_action\", \"arguments\": {\"action\": \"click\", \"coordinate\": [x, y]}}</tool_call>\n"
            "or directly output the function call in <tool_call></tool_call> tags as follows:\n"
            "<tool_call>{\"name\": \"gui_action\", \"arguments\": {\"action\": \"click\", \"coordinate\": [x, y]}}</tool_call>\n"
        )

        resized_height, resized_width  = smart_resize(dummy_image.height,
            dummy_image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=self.processor.image_processor.max_pixels,)
        screenspot = GUIAction(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )
        img_bytes = image["bytes"]
        b64_str = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:image/png;base64,{b64_str}"
        nousFnCallPrompt = NousFnCallPrompt()
        system_message = nousFnCallPrompt.preprocess_fncall_messages(
            messages = [
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                # Message(role="user", content=[
                #     ContentItem(text=user_query),
                #     ContentItem(image=data_uri)
                # ]),
            ],
            functions=[screenspot.function],
            lang=None,
        )
        system_message = system_message[0].model_dump()
        message=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": msg["text"]} for msg in system_message["content"]
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                        # PNG image:  f"data:image/png;base64,{base64_image}"
                        # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                        # WEBP image: f"data:image/webp;base64,{base64_image}"
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

        # prompt.replace("<|vision_start|><|image_pad|><|vision_end|>","")
        # prompt.replace("<image>","<|vision_start|><|image_pad|><|vision_end|>")

        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)


        inputs = self.processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
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
    def __init__(self, model_path, sampling_params, output_path=None):
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 1, "video": 1},
        )
        self.sampling_params = sampling_params
        self.output_path = output_path

    def process_data(self, dataloader):
        results = []

        for batch in tqdm(dataloader):
            batch_results = []
            prompts = batch["prompts"]
            multi_modal_data = batch["multi_modal_data"]
            mm_processor_kwargs = batch["mm_processor_kwargs"]
            original_samples = batch["original_samples"]

            for prefix in ['<think>', '<tool_call>']:

                llm_inputs = [
                    {
                        "prompt": prompt+prefix,
                        "multi_modal_data": mm_data,
                        "mm_processor_kwargs": mm_kwargs,
                    }
                    for prompt, mm_data, mm_kwargs in zip(prompts, multi_modal_data, mm_processor_kwargs)
                ]

                # 执行推理
                outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params, use_tqdm=True)

                # 保存结果
                
                for original_sample, output in zip(original_samples, outputs):
                    # one_pass = False
                    # one_fail = False
                    flags = []
                    # texts = [output.outputs[i].text for i in range(len(output.outputs))]
                    # print(texts)
                    gt_bbox = original_sample["gt_bbox"]

                    for i in range(len(output.outputs)):

                        generated_text = output.outputs[i].text
                        # original_sample["pred"] = generated_text
                        pred_coord, _ = extract_coord(generated_text)
                        pred_coord = [pred_coord[0]*original_sample["scale"][0],pred_coord[1]*original_sample["scale"][1]]
                        print(pred_coord, gt_bbox)
                        if gt_bbox[0] < pred_coord[0] < gt_bbox[2] and gt_bbox[1] < pred_coord[1] < gt_bbox[3]:
                            # print(generated_text, "pass")
                            flags.append(True)
                            # one_pass = True
                        else:
                            flags.append(False)
                        # original_sample["pred_coord"] = pred_coord
                    original_sample[f"{prefix}_pass"] = flags
                    original_sample["image"]=''
                    if prefix == '<tool_call>':
                        results.append(original_sample)
                        batch_results.append(original_sample)
                    # results.append(original_sample)
                    # print(original_sample)
            
            with open(self.output_path, "a") as ans_file:
                for sample in batch_results:
                    ans_file.write(json.dumps(sample) + "\n")

        return results


def main(args):
    # 将数据分成 8 份
    MODEL_PATH=args.model_path
    DATA_PATH=args.data_path
    if DATA_PATH.endswith('parquet'):
        data=load_dataset("parquet", data_files=DATA_PATH, split="train")
    else:
        data = [json.loads(s) for s in open(DATA_PATH, "r")] if DATA_PATH.endswith(".jsonl") else json.load(open(DATA_PATH,"r"))
    # 输出路径
    OUTPUT_DIR = args.output_path
    num_actors = args.num_actor
    OUTPUT_DIR = os.path.join(OUTPUT_DIR,MODEL_PATH.split('/')[-1])
    NEW_FILE = os.path.join(OUTPUT_DIR, DATA_PATH.split("/")[-1].replace(".jsonl", "_pred.jsonl").replace('.parquet','.json'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_chunks = [hf_dataset.from_dict(data[i::num_actors]) for i in range(num_actors)]


    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    # processor.max_pixels=1048576
    # processor.min_pixels=

    # 创建 8 个 Actor，每个 Actor 分配到一个 GPU
    workers = [Worker.remote(MODEL_PATH, SAMPLING_PARAMS, NEW_FILE) for _ in range(num_actors)]

    # 使用 PyTorch Dataset 和 DataLoader
    futures = []
    for i, chunk in enumerate(data_chunks):
        dataset = MultiModalDataset(chunk, processor)
        dataloader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)
        futures.append(workers[i].process_data.remote(dataloader))

    # 收集所有结果
    all_results = ray.get(futures)

    # 将结果写入文件
    # with open(NEW_FILE, "w") as ans_file:
    #     for worker_results in all_results:
    #         for sample in worker_results:
    #             ans_file.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='<model_path>')
    parser.add_argument('--data_path', type=str, default="<data_path>")
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--num_actor', type=int, default=8)
    args = parser.parse_args()
    main(args)
