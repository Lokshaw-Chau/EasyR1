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
# 初始化 Ray
ray.init()
from datasets import load_dataset
from datasets import Dataset as hf_dataset

from qwen_vl_utils import smart_resize
import copy
import base64
import time



# 模型路径
MODEL_PATH = ""
WEB_SYS_PROMPT = (
    "You are a helpful assistant.\n\n"
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n" 
    "<tools>\n{{" 
        "\"type\": \"function\", " 
        "\"function\": {{" 
            "\"name_for_human\": \"computer_use\", " 
            "\"name\": \"computer_use\", " 
            "\"description\": \"Use a mouse and keyboard to interact with a computer, and take screenshots.\\n" 
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n" 
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\\n" 
            "* The screen's resolution is {display_width_px}x{display_height_px}.\\n" 
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n" 
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n" 
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", " 
        "\"parameters\": {{" 
            "\"properties\": {{" 
                "\"action\": {{" 
                    "\"description\": \"The action to perform. The available actions are:\\n" 
                    "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n" 
                    "* `type`: Type a string of text on the keyboard.\\n" 
                    "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n" 
                    "* `left_click`: Click the left mouse button.\\n"
                    "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n" 
                    "* `right_click`: Click the right mouse button.\\n" 
                    "* `middle_click`: Click the middle mouse button.\\n" 
                    "* `double_click`: Double-click the left mouse button.\\n" 
                    "* `scroll`: Performs a scroll of the mouse scroll wheel.\\n" 
                    "* `terminate`: Terminate the current task and report its completion status.\", " 
                    "\"enum\": [\"key\", \"type\", \"mouse_move\", \"left_click\", \"left_click_drag\", \"right_click\", \"middle_click\", \"double_click\", \"scroll\", \"wait\", \"terminate\"], \"type\": \"string\"}}, " 
                "\"keys\": {{\"description\": \"Required only by `action=key`.\", \"type\": \"array\"}}, " 
                "\"text\": {{\"description\": \"Required only by `action=type`.\", \"type\": \"string\"}}, " 
                "\"coordinate\": {{\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.\", \"type\": \"array\"}}, " 
                "\"pixels\": {{\"description\": \"The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.\", \"type\": \"number\"}}, " 
                "\"status\": {{\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}}}, " 
            "\"required\": [\"action\"], " 
            "\"type\": \"object\"}}, " 
        "\"args_format\": \"Format the arguments as a JSON object.\"}}" 
    "}}\n</tools>\n\n" 
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
)

# 推理参数
# SAMPLING_PARAMS = SamplingParams(
#     temperature=0.0,
#     top_p=0.001,
#     repetition_penalty=1.05,
#     max_tokens=1024,  # 根据需要调整最大生成长度
#     stop_token_ids=[],  # 停止标志
# )
# 数据路径
DATA_PATH = ""

# 微批大小
MICRO_BATCH = 4

def extract_action(content):
    # answer_tag_pattern = r'<tool_call>(.*?)</tool_call>'
    action_pattern = r"\"action\":\s*\"(\w+)\""
    # content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    # if content_answer_match:
        # content_answer = content_answer_match.group(1).strip()
    action_match = re.search(action_pattern, content)
    if action_match:
        return action_match.group(1)
    return None

def extract_input_text(content):
    # answer_tag_pattern = r'<tool_call>(.*?)</tool_call>'
    action_pattern = r"\"text\":\s*\"(.*?)\""
    # content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    # if content_answer_match:
    #     content_answer = content_answer_match.group(1).strip()
    action_match = re.search(action_pattern, content)
    if action_match:
        return action_match.group(1)
    return "no input text"

def extract_keys(content):
    # answer_tag_pattern = r'<tool_call>(.*?)</tool_call>'
    action_pattern = r"\"keys\":\s*(.*?)"
    # content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    # if content_answer_match:
    #     content_answer = content_answer_match.group(1).strip()
    action_match = re.search(action_pattern, content)
    if action_match:
        return action_match.group(1)
    return "no input text"

def extract_status(content):
    # answer_tag_pattern = r'<tool_call>(.*?)</tool_call>'
    action_pattern = r"\"button\":\s*\"(.*?)\""
    # content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    # if content_answer_match:
    #     content_answer = content_answer_match.group(1).strip()
    action_match = re.search(action_pattern, content)
    if action_match:
        return action_match.group(1)
    return "no input text"

def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    # answer_tag_pattern = r'<tool_call>(.*?)</tool_call>'
    bbox_pattern = r'\"coordinate\": \[(\d+),\s*(\d+)\]'
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
    
def extract_coord2(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    # answer_tag_pattern = r'<tool_call>(.*?)</tool_call>'
    bbox_pattern = r'\"coordinate2\": \[(\d+),\s*(\d+)\]'
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

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_str=predicted_str.replace("[","").replace("]","")
    ground_truth_str=ground_truth_str.replace("[","").replace("]","")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens)==1 and len(ground_truth_tokens)==1:
        predicted_token=list(predicted_tokens)[0]
        ground_truth_token=list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

class MultiModalDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
        self.processor.max_pixels=1258291

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"]
        dummy_image = Image.open(BytesIO(image["bytes"]))
        text = sample["instruction"]
        history="None" if 'history' not in sample else sample['history']

        # sys_prompt='''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> nd <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>'''
        user_query = (
                "You may conduct step-by-step reasoning to help you better solve the problem before the <tool_call></tool_call> XML tags."
                "The thinking process MUST be surrounded <thinking></thinking> tags as follows:\n"
                "<thinking> ... </thinking> <tool_call>{\"name\": \"computer_use\", \"arguments\": {\"action\": \"...\", ...}}</tool_call>\n"
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
                    {"type": "text", "text": WEB_SYS_PROMPT.format(display_width_px=resized_width,display_height_px=resized_height)} 
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

            for prefix in ['<thinking>', '<tool_call>']:

                llm_inputs = [
                    {
                        "prompt": prompt + prefix,
                        "multi_modal_data": mm_data,
                        "mm_processor_kwargs": mm_kwargs,
                    }
                    for prompt, mm_data, mm_kwargs in zip(prompts, multi_modal_data, mm_processor_kwargs)
                ]

                # 执行推理
                outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params, use_tqdm=True)

                # 保存结果
                for original_sample, output in zip(original_samples, outputs):

                    flags = []
                    preds = []
                    gt_bbox = original_sample["gt_bbox"]

                    for i in range(len(output.outputs)):

                        generated_text = output.outputs[i].text
                        preds.append(generated_text)
                        pred_action = extract_action(generated_text)
                        
                        if pred_action != original_sample["gt_action"]:
                            flags.append(False)
                            continue
                        if pred_action in ['left_click', 'right_click', 'double_click', 'middle_click', 'move_mouse', 'left_click_drag']:
                            pred_coord, _ = extract_coord(generated_text)
                            pred_coord = [pred_coord[0]*original_sample["scale"][0],pred_coord[1]*original_sample["scale"][1]]
                            print(pred_coord, gt_bbox)
                            pred_x, pred_y = pred_coord[:2]
                            gt_bbox = original_sample['gt_bbox']
                            if ((gt_bbox[0]-pred_x)/original_sample['image_size'][0])**2 + ((gt_bbox[1]-pred_y)/original_sample['image_size'][1])**2 < 0.14**2:
                                flags.append(True)
                            else:
                                flags.append(False)
                        elif pred_action in ['type']:
                            pred_input_text = extract_input_text(generated_text)
                            if calculate_f1_score(pred_input_text, original_sample['gt_input_text'])>=0.5:
                                flags.append(True)
                            else:
                                flags.append(False)
                        elif pred_action in ['key']:
                            pred_input_text = extract_keys(generated_text)
                            if calculate_f1_score(pred_input_text, original_sample['gt_input_text'])>=0.5:
                                flags.append(True)
                            else:
                                flags.append(False)
                        elif pred_action in ['scroll']:
                                flags.append(True)
                        elif pred_action in ['terminate']:
                            flags.append(True)
                        else:
                            print(f"Unrecognized action: {pred_action}")
                            
                        
                    original_sample[f"{prefix}_pass"] = flags
                    original_sample["image"]=''
                    original_sample[f"{prefix}_pred"] = preds
                    if prefix == '<tool_call>':
                        # results.append(original_sample)
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
    if DATA_PATH.endswith('.parquet'):
        data=load_dataset("parquet", data_files=DATA_PATH, split="train")
    else:
        data = [json.loads(s) for s in open(DATA_PATH, "r")] if DATA_PATH.endswith(".jsonl") else json.load(open(DATA_PATH,"r"))
    # 输出路径
    OUTPUT_DIR = args.output_path
    num_actors = args.num_actor
    OUTPUT_DIR = os.path.join(OUTPUT_DIR,MODEL_PATH.split('/')[-1])
    NEW_FILE = os.path.join(OUTPUT_DIR, DATA_PATH.split("/")[-1].replace(".jsonl", "_pred.jsonl").replace('.parquet',f'_{args.n}.json'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # filter data
    # if args.n != 1:
    #     old_file = NEW_FILE.replace(f'_{args.n}.json', f'_{args.n//2}.json')
    #     if os.path.exists(old_file):
    #         print(f"Filtering data based on {old_file}")
    #         with open(old_file, "r") as f:
    #             old_file_data = [json.loads(line) for line in f.readlines()]
    #         new_data_id_list = []
    #         for item in old_file_data:
    #             think_pass_list = item['<thinking>_pass']
    #             tool_call_pass_list = item['<tool_call>_pass']
    #             if any(think_pass_list) and any(tool_call_pass_list):
    #                 continue
    #             new_data_id_list.append(item['instruction']+str(item['gt_bbox'])+item['gt_input_text']+item['gt_action']+item['history'])

    #     # if id not in new_data_id_list, drop the sample
    #         data = [item for item in data if (item['instruction']+str(item['gt_bbox'])+item['gt_input_text']+item['gt_action']+item['history']) in new_data_id_list]
    #         data = hf_dataset.from_list(data)
    #         print(f"Filtered data size: {len(data)}")
    
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
        dataloader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False, num_workers=64, collate_fn=custom_collate_fn)
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
    parser.add_argument('--n', type=int, default=8, help='Number of trials to run')
    args = parser.parse_args()
    SAMPLING_PARAMS = SamplingParams(
    temperature=1.0,
    top_p=0.999,
    repetition_penalty=1.05,
    max_tokens=1024,  # 根据需要调整最大生成长度
    n=args.n,
    stop_token_ids=[],  # 停止标志
)
    main(args)