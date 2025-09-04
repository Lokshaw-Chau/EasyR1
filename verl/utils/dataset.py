# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
import json
from .agent_function_call import MobileUse

# from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
#     NousFnCallPrompt,
#     Message,
#     ContentItem,
# )

ORIGINAL_SYS_PROMPT = (
    "You are a helpful assistant.\n\n"
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{{"
        "\"type\": \"function\", "
        "\"function\": {{"
            "\"name_for_human\": \"mobile_use\", "
            "\"name\": \"mobile_use\", "
            "\"description\": \"Use a touchscreen to interact with a mobile device, and take screenshots.\\n" 
                "* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n" 
                "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n" 
                "* The screen's resolution is {display_width_px}x{display_height_px}.\\n" 
                "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", "
            "\"parameters\": {{"
                "\"properties\": {{"
                    "\"action\": {{"
                        "\"description\": \"The action to perform. The available actions are:\\n" 
                        "* `key`: Perform a key event on the mobile device.\\n    - This supports adb's `keyevent` syntax.\\n    - Examples: \\\"volume_up\\\", \\\"volume_down\\\", \\\"power\\\", \\\"camera\\\", \\\"clear\\\".\\n" 
                        "* `click`: Click the point on the screen with coordinate (x, y).\\n"
                        "* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n"
                        "* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n"
                        "* `type`: Input the specified text into the activated input box.\\n"
                        "* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n"
                        "* `open`: Open an app on the device.\\n"
                        "* `wait`: Wait specified seconds for the change to happen.\\n"
                        "* `terminate`: Terminate the current task and report its completion status.\", "
                        "\"enum\": [\"key\", \"click\", \"long_press\", \"swipe\", \"type\", \"answer\", \"system_button\", \"open\", \"wait\", \"terminate\"], \"type\": \"string\"}}, "
                    "\"coordinate\": {{\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.\", \"type\": \"array\"}}, "
                    "\"coordinate2\": {{\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.\", \"type\": \"array\"}}, "
                    "\"text\": {{\"description\": \"Required only by `action=key`, `action=type`, `action=answer`, and `action=open`.\", \"type\": \"string\"}}, "
                    "\"time\": {{\"description\": \"The seconds to wait. Required only by `action=long_press` and `action=wait`.\", \"type\": \"number\"}}, "
                    "\"button\": {{\"description\": \"Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`\", \"enum\": [\"Back\", \"Home\", \"Menu\", \"Enter\"], \"type\": \"string\"}}, "
                    "\"status\": {{\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}}}, "
                "\"required\": [\"action\"], """
                "\"type\": \"object\"}}, "
            "\"args_format\": \"Format the arguments as a JSON object.\"}}"
    "}}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
)

AC_SYS_PROMPT = (
    "You are a helpful assistant.\n\n"
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{{"
        "\"type\": \"function\", "
        "\"function\": {{"
            "\"name_for_human\": \"mobile_use\", "
            "\"name\": \"mobile_use\", "
            "\"description\": \"Use a touchscreen to interact with a mobile device, and take screenshots.\\n" 
                "* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n" 
                "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n" 
                "* The screen's resolution is {display_width_px}x{display_height_px}.\\n" 
                "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", "
            "\"parameters\": {{"
                "\"properties\": {{"
                    "\"action\": {{"
                        "\"description\": \"The action to perform. The available actions are:\\n" 
                        "* `click`: Click the point on the screen with coordinate (x, y).\\n"
                        "* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n"
                        "* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n"
                        "* `type`: Input the specified text into the activated input box.\\n"
                        "* `system_button`: Press the system button.\\n"
                        "* `open`: Open an app on the device.\\n"
                        "* `wait`: Wait specified seconds for the change to happen.\","
                        "\"enum\": [\"click\", \"long_press\", \"swipe\", \"type\", \"system_button\", \"open\", \"wait\"], \"type\": \"string\"}}, "
                    "\"coordinate\": {{\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.\", \"type\": \"array\"}}, "
                    "\"coordinate2\": {{\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.\", \"type\": \"array\"}}, "
                    "\"text\": {{\"description\": \"Required only by `action=type` and `action=open`.\", \"type\": \"string\"}}, "
                    "\"time\": {{\"description\": \"The seconds to wait. Required only by `action=long_press` and `action=wait`.\", \"type\": \"number\"}}, "
                    "\"button\": {{\"description\": \"Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`.\", \"enum\": [\"Back\"], \"type\": \"string\"}}}}, "
                "\"required\": [\"action\"], """
                "\"type\": \"object\"}}, "
            "\"args_format\": \"Format the arguments as a JSON object.\"}}"
    "}}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
)

ODYSSEY_SYS_PROMPT = (
    "You are a helpful assistant.\n\n"
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{{"
        "\"type\": \"function\", "
        "\"function\": {{"
            "\"name_for_human\": \"mobile_use\", "
            "\"name\": \"mobile_use\", "
            "\"description\": \"Use a touchscreen to interact with a mobile device, and take screenshots.\\n" 
                "* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n" 
                "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n" 
                "* The screen's resolution is {display_width_px}x{display_height_px}.\\n" 
                "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", "
            "\"parameters\": {{"
                "\"properties\": {{"
                    "\"action\": {{"
                        "\"description\": \"The action to perform. The available actions are:\\n" 
                        "* `click`: Click the point on the screen with coordinate (x, y).\\n"
                        "* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n"
                        "* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n"
                        "* `type`: Input the specified text into the activated input box.\\n"
                        "* `system_button`: Press the system button.\\n"
                        "* `terminate`: Terminate the current task and report its completion status.\", "
                        "\"enum\": [\"click\", \"long_press\", \"swipe\", \"type\", \"system_button\", \"terminate\"], \"type\": \"string\"}}, "
                    "\"coordinate\": {{\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.\", \"type\": \"array\"}}, "
                    "\"coordinate2\": {{\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.\", \"type\": \"array\"}}, "
                    "\"text\": {{\"description\": \"Required only by `action=type`.\", \"type\": \"string\"}}, "
                    "\"time\": {{\"description\": \"The seconds to wait. Required only by `action=long_press`.\", \"type\": \"number\"}}, "
                    "\"button\": {{\"description\": \"Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`\", \"enum\": [\"Back\", \"Home\", \"Menu\"], \"type\": \"string\"}}, "
                    "\"status\": {{\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}}}, "
                "\"required\": [\"action\"], """
                "\"type\": \"object\"}}, "
            "\"args_format\": \"Format the arguments as a JSON object.\"}}"
    "}}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
)

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    if isinstance(image, str):
        image = Image.open(image)
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.format_prompt = format_prompt
        # self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"
            # print(data_path)

        if os.path.isdir(data_path):
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]

        # prompt_str: str = row_dict[self.prompt_key]
        text=row_dict['instruction']
        ui_type = row_dict['ui_type']
        history=row_dict['history']
        # task_type=row_dict['task_type']
        row_dict.pop('verify_bbox', None)
        row_dict.pop('success_rate', None)
        row_dict.pop('scale', None)
        images=[row_dict['image']]
        
      
        # if task_type=='high':
        if self.format_prompt == "no_think":
            prompt_str=  (
                f"<image>\nThe user query: {text}\n"
                "Directly output the function call in <tool_call></tool_call> tags as follows:\n"
                "<tool_call>{\"name\": \"gui_action\", \"arguments\": {\"action\": \"click\", \"coordinate\": [x, y]}}</tool_call>\n"
            )
        elif self.format_prompt == "think":
            prompt_str=  (
                f"<image>\nThe user query: {text}\n"
                "Output the thinking process in <think></think> tags, and the function call in <tool_call></tool_call> tags as follows:\n"
                "<think> ... </think> <tool_call>{\"name\": \"gui_action\", \"arguments\": {\"action\": \"click\", \"coordinate\": [x, y]}}</tool_call>\n"
            )
        elif self.format_prompt == "adaptive":
            # prompt_str=  (
            #     f"<image>\nThe user query: {text}\n"
            #     "Output the thinking process in <think></think> tags, and the function call in <tool_call></tool_call> tags as follows:\n"
            #     "<think> ... </think> <tool_call>{\"name\": \"gui_action\", \"arguments\": {\"action\": \"click\", \"coordinate\": [x, y]}}</tool_call>\n"
            #     "or directly output the function call in <tool_call></tool_call> tags as follows:\n"
            #     "<tool_call>{\"name\": \"gui_action\", \"arguments\": {\"action\": \"click\", \"coordinate\": [x, y]}}</tool_call>\n"
            # )
            prompt_str = (
                "You may conduct step-by-step reasoning to help you better solve the problem before the <tool_call></tool_call> XML tags."
                "The thinking process MUST be surrounded <thinking></thinking> tags as follows:\n"
                "<thinking> ... </thinking> <tool_call>{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"...\", ...}}</tool_call>\n"
                f"<image>\nThe user query: {text}\n"
                f"Task progress (You have done the following operation on the current device): {history}\n"
            )
        else:
            raise ValueError(f"Unknown format_prompt {self.format_prompt}.")
        
 # w/ think prompt
            #  prompt_str=  (
            #     f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being '{history}'.\n"
            #     "Please provide the action to perform (enumerate from ['complete', 'close/delete', 'press_home', 'click', 'press_back', 'type', 'select', 'scroll', 'enter']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
            #     "Output the final answer in <answer> </answer> tags as follows:\n"
            #     "<answer>[{'action': enum['complete', 'close/delete', 'press_home', 'click', 'press_back', 'type', 'select', 'scroll', 'enter'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
            #     "Note:\n specific input text (no default) is necessary for actions enum['type', 'select', 'scroll'] \n Example:\n"
            #     "[{'action': enum['complete', 'close/delete', 'press_home', 'press_back', 'enter'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
            #     "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
            #     "[{'action': enum['type', 'select'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
            #     "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
            # ) # w/o think prompt
            # else:
            #     prompt_str=(
            #         f"In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being '{history}'.\n"
            #         "Please provide the action to perform (enumerate from ['click']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
            #         "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
            #         "<think> ... </think> <answer>[{'action': enum[ 'click'], 'point': [x, y], 'input_text': 'no input text'}]</answer>\n" \
            #         "Note:\n thinking process can be omitted with ...\n"
            #         "Example:\n"
            #         "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
            #     ) # w/ think prompt
            # prompt_str=(
            #     f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being '{history}'.\n"
            #     "Please provide the action to perform (enumerate from ['click']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
            #     "Output the final answer in <answer> </answer> tags as follows:\n"
            #     "<answer>[{'action': enum[ 'click'], 'point': [x, y], 'input_text': 'no input text'}]</answer>\n"
            #     "Example:\n"
            #     "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
            # ) # w/o think prompt
        
        images=[process_image(image, self.max_pixels, self.min_pixels) for image in images]

        # screenspot = MobileUse(
        #     cfg={"display_width_px": images[0].width, "display_height_px": images[0].height},
        # )
        # nousFnCallPrompt = NousFnCallPrompt()
        # system_message = nousFnCallPrompt.preprocess_fncall_messages(
        #     messages = [
        #         Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
        #         # Message(role="user", content=[
        #         #     ContentItem(text=user_query),
        #         #     ContentItem(image=data_uri)
        #         # ]),
        #     ],
        #     functions=[screenspot.function],
        #     lang=None,
        # )
        # system_message = system_message[0].model_dump()
        # syttem_message = [text['text'] for text in system_message['content']]
        # system_message = ' '.join(syttem_message)
        if ui_type == 'gui_odyssey':
            system_message = ODYSSEY_SYS_PROMPT.format(
                display_width_px=images[0].width, display_height_px=images[0].height
            )
        elif ui_type == 'android_control':
            system_message = AC_SYS_PROMPT.format(
                display_width_px=images[0].width, display_height_px=images[0].height
            )
        else:
            print(f"[Warning] Unknown ui_type: {ui_type}, use the original system prompt as default.")
            system_message = ORIGINAL_SYS_PROMPT.format(
                display_width_px=images[0].width, display_height_px=images[0].height
            )
        
        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user", 
                "content": prompt_str
            },
        ]
        scalex,scaley=images[0].size
        gt_bbox=row_dict['gt_bbox']
        gt_bbox[0]*=scalex
        gt_bbox[1]*=scaley
        if len(gt_bbox)>2:
            gt_bbox[2]*=scalex
            gt_bbox[3]*=scaley

        gt={'action': row_dict['gt_action'],'gt_bbox': gt_bbox,'input_text': row_dict['gt_input_text'], 'image_size': [scalex,scaley], 'ui_type': ui_type}
        # gt={'gt_bbox': gt_bbox}
        # if self.system_prompt:
        #     messages.insert(0, {"role": "system", "content": self.system_prompt})

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # if self.image_key in row_dict:
        prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        row_dict["multi_modal_data"] = {
            "image": images
        }
        model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        row_dict["multi_modal_inputs"] = dict(model_inputs)
        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs["image_grid_thw"],
            attention_mask=attention_mask,
        )  # (3, seq_length)
        # else:
        #     model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        #     input_ids = model_inputs.pop("input_ids")[0]
        #     attention_mask = model_inputs.pop("attention_mask")[0]
        #     position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = json.dumps(gt)
        return row_dict
