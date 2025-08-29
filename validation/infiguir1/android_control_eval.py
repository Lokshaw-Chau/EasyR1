import json
import os
import random
import gc
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from typing import List, Dict, Any
import multiprocessing as mp
import signal
from functools import partial

from evaluate_android_control import *
from qwen2vl import Qwen2VL
from qwen_vl_utils import smart_resize
import time
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from agent_function_call import MobileUse

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def make_history(step_pam):
    action = step_pam['action']
    if action in ['click', 'long_press']:
        bbox = step_pam['coordinate']
        step_info = f'{{\"name\": \"mobile_use\", \"arguments\": {{\"action\": \"{action}\", \"coordinate\": {bbox}}}}}'
    elif action in ['type', 'open']:
        input_txt = step_pam['text']
        step_info = f'{{\"name\": \"mobile_use\", \"arguments\": {{\"action\": \"{action}\", \"text\": \"{input_txt}\"}}}}'
    elif action in ['system_button']:
        input_txt = step_pam['button']
        step_info = f'{{\"name\": \"mobile_use\", \"arguments\": {{\"action\": \"{action}\", \"button\": \"{input_txt}\"}}}}'
    elif action in ['wait']:
        step_info = f'{{\"name\": \"mobile_use\", \"arguments\": {{\"action\": \"{action}\", \"time\": 3}}}}'
    elif action in ['swipe']:
        coord1 = step_pam['coordinate']
        coord2 = step_pam['coordinate2']
        step_info = f'{{\"name\": \"mobile_use\", \"arguments\": {{\"action\": \"{action}\", \"coordinate\": {coord1}, \"coordinate2\": {coord2}}}}}'
    else:
        print('Unknown action type:', action)
        return ""
    return step_info

def _process_image_worker(args):
    try:
        image_root, image_path, resized_width, resized_height = args
        image = Image.open(os.path.join(image_root, image_path))
        if image.size != (resized_width, resized_height):
            image = image.resize((resized_width, resized_height))
        return image
    except Exception as e:
        print(f"Error processing image {args[1] if len(args) > 1 else 'unknown'}: {e}")
        return None

class AndroidControl:
    def __init__(
        self, 
        model_path,
        eval_file, 
        image_root,
        output_dir,
        eval_type,
        # thinking=False,
        prefix='',
        max_pixels=6400*28*28,
        tensor_parallel_size=1,
        enforce_eager=False,
        max_num_seqs=2,
        seed=42,
        # debug=False,
        num_processes=16,
        ):
        self.model_path = model_path
        self.eval_file = eval_file
        self.eval_type = eval_type
        self.max_pixels = max_pixels
        self.image_root = image_root
        # self.thinking = thinking
        self.prefix = prefix
        # self.debug = debug
        self.num_processes = num_processes
        self.output_dir = output_dir
        # if thinking:
        #     self.system_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags.\nDuring the reasoning process, identify and state the sub-goal of the current step by enclosing it within <step> </step> tags."
        # else:
        #     self.system_prompt = "You are a helpful assistant."
        self.seed = seed

        # self.llm = Qwen2VL(
        #     model_path=model_path,
        #     tensor_parallel_size=tensor_parallel_size,
        #     enforce_eager=enforce_eager,
        #     max_num_seqs=max_num_seqs,
        #     max_pixels=max_pixels
        # )

    def generate_jobs(self, total_split, split) -> List[Dict]:

        with open(self.eval_file, 'r', encoding='utf-8') as f:
            lines = json.load(f)
        split_len = len(lines) // total_split
        start_idx = split * split_len
        end_idx = (split + 1) * split_len
        end_idx = min(end_idx, len(lines))
        lines = lines[start_idx: end_idx]
        jobs = []
        for line in tqdm(lines, desc='Generating jobs'):
            # image
            width, height = line['width'], line['height']
            h_bar, w_bar = smart_resize(height, width, max_pixels=self.max_pixels)

            # system message
            # tool_system_prompt = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"mobile_use\", \"description\": \"Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is " + str(w_bar) + "x" + str(h_bar) + ".\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", \"parameters\": {\"properties\": {\"action\": {\"description\": \"The action to perform. The available actions are:\\n* `key`: Perform a key event on the mobile device.\\n    - This supports adb's `keyevent` syntax.\\n    - Examples: \\\"volume_up\\\", \\\"volume_down\\\", \\\"power\\\", \\\"camera\\\", \\\"clear\\\".\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `system_button`: Press the system button.\\n* `open`: Open an app on the device.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\", \"enum\": [\"key\", \"click\", \"long_press\", \"swipe\", \"type\", \"system_button\", \"open\", \"wait\", \"terminate\"], \"type\": \"string\"}, \"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.\", \"type\": \"array\"}, \"coordinate2\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.\", \"type\": \"array\"}, \"text\": {\"description\": \"Required only by `action=key`, `action=type`, and `action=open`.\", \"type\": \"string\"}, \"time\": {\"description\": \"The seconds to wait. Required only by `action=long_press` and `action=wait`.\", \"type\": \"number\"}, \"button\": {\"description\": \"Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`\", \"enum\": [\"Back\", \"Home\", \"Menu\", \"Enter\"], \"type\": \"string\"}, \"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, \"required\": [\"action\"], \"type\": \"object\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
            # system_message = self.system_prompt + tool_system_prompt

            screenspot = MobileUse(
                cfg={"display_width_px": w_bar, "display_height_px": h_bar},
                )
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
            system_message = [text['text'] for text in system_message['content']]
            system_message = ' '.join(system_message)

            # user message
            task_progress = []
            assert len(line["step_check_pams"]) == len(line["step_pams"]) == len(line["episode"]["step_instructions"]) == len(line["episode"]["screenshot_paths"]) - 1
            for i in range(len(line["step_check_pams"])):
                step_check_pam = line["step_check_pams"][i]
                step_pam = line["step_pams"][i]
                step_instruction = line["episode"]["step_instructions"][i]
                screenshot_path = line["episode"]["screenshot_paths"][i]
                if self.eval_type == 'low':
                    user_message_template = (
                        "You may conduct step-by-step reasoning to help you better solve the problem before the <tool_call></tool_call> XML tags."
                        "The thinking process MUST be surrounded <thinking></thinking> tags as follows:\n"
                        "<thinking> ... </thinking> <tool_call>{{\"name\": \"mobile_use\", \"arguments\": {{\"action\": \"...\", ...}}}}</tool_call>\n"
                        "<image>\nThe user query: {goal}\nCurrent step query: {step_instruction}\n"
                        "Task progress (You have done the following operation on the current device): {task_progress}\n"
                    )
                    # user_message_template = "The user query:  {goal}\nCurrent step query: {step_instruction}\nTask progress (You have done the following operation on the current device): {task_progress}"
                    user_message = user_message_template.format(
                        goal=line["episode"]["goal"], 
                        step_instruction=step_instruction, 
                        task_progress=''.join([f'Step {n+1}: {tp}; ' for n, tp in enumerate(task_progress)])
                    )
                elif self.eval_type == 'high':
                    user_message_template = (
                        "You may conduct step-by-step reasoning to help you better solve the problem before the <tool_call></tool_call> XML tags."
                        "The thinking process MUST be surrounded <thinking></thinking> tags as follows:\n"
                        "<thinking> ... </thinking> <tool_call>{{\"name\": \"mobile_use\", \"arguments\": {{\"action\": \"...\", ...}}}}</tool_call>\n"
                        "<image>\nThe user query: {goal}\n"
                        "Task progress (You have done the following operation on the current device): {task_progress}\n"
                    )
                    
                    # user_message_template = "The user query:  {goal}\nTask progress (You have done the following operation on the current device): {task_progress}"
                    user_message = user_message_template.format(
                        goal=line["episode"]["goal"], 
                        task_progress=''.join([f'Step {n+1}: {tp}; ' for n, tp in enumerate(task_progress)])
                    )
                else:
                    raise ValueError(f"Invalid eval type: {self.eval_type}")
                # task_progress.append(json.dumps(step_pam, ensure_ascii=False))
                task_progress.append(make_history(step_pam))
                jobs.append({
                    'question_id': len(jobs),
                    'step_id': i,
                    'episode_id': line['episode']['episode_id'],
                    'messages': [
                        {
                            'role': 'system',
                            'content': system_message
                        },
                        {
                            'role': 'user',
                            'content': user_message # + '<image>'
                        }
                    ],
                    'image_path': screenshot_path,
                    'width': width,
                    'height': height,
                    'resized_width': w_bar,
                    'resized_height': h_bar,
                    'check_pams': step_check_pam,
                })
        
        print(jobs[0]['messages'])

        if self.debug:
            random.seed(self.seed)
            jobs = random.sample(jobs, len(jobs)//10)
        return jobs 

    def inference(self, jobs, temperature=0.0, max_tokens=4096, seed=42) -> List[Dict]:
        input_messages = [job['messages'] for job in jobs]
        
        images = []
        
        if self.num_processes > 1:
            process_args = [
                (self.image_root, job['image_path'], job['resized_width'], job['resized_height']) 
                for job in jobs
            ]
            
            mp_context = mp.get_context('spawn')
            try:
                with mp_context.Pool(processes=self.num_processes, initializer=init_worker) as pool:
                    processed_results = []
                    
                    pbar = tqdm(total=len(jobs), desc='Preprocessing images (multiprocessing)')
                    
                    for i, result in enumerate(pool.map(_process_image_worker, process_args)):
                        processed_results.append((i, result))
                        pbar.update(1)
                    
                    pbar.close()
                    
                    processed_results.sort(key=lambda x: x[0])
                    images = [result for _, result in processed_results]
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                if 'pool' in locals():
                    pool.terminate()
                    pool.join()
                raise
            except Exception as e:
                print(f"Error in multiprocessing: {e}")
                print("Falling back to single process mode")
                images = []
                for job in tqdm(jobs, desc='Preprocessing images (single process fallback)'):
                    try:
                        image = Image.open(os.path.join(self.image_root, job['image_path']))
                        if image.size != (job['resized_width'], job['resized_height']):
                            image = image.resize((job['resized_width'], job['resized_height']))
                        images.append(image)
                    except Exception as e:
                        print(f"Error processing image {job['image_path']}: {e}")
                        images.append(None)
        else:
            for job in tqdm(jobs, desc='Preprocessing images (single process)'):
                try:
                    image = Image.open(os.path.join(self.image_root, job['image_path']))
                    if image.size != (job['resized_width'], job['resized_height']):
                        image = image.resize((job['resized_width'], job['resized_height']))
                    images.append(image)
                except Exception as e:
                    print(f"Error processing image {job['image_path']}: {e}")
                    images.append(None)
        
        valid_jobs = []
        valid_images = []
        valid_input_messages = []
        for i, (job, image, msg) in enumerate(zip(jobs, images, input_messages)):
            if image is not None:
                valid_jobs.append(job)
                valid_images.append(image)
                valid_input_messages.append(msg)
            else:
                print(f"Skipping job {i} due to image processing failure")
                # continue
        
        batch_size = 256
        outputs = []
        
        with tqdm(total=len(valid_input_messages), desc='Processing batches') as pbar:
            for i in range(0, len(valid_input_messages), batch_size):
                batch_messages = valid_input_messages[i:i+batch_size]
                batch_images = valid_images[i:i+batch_size]
                batch_outputs = self.llm.chat(batch_messages, batch_images, 
                                            temperature=temperature, 
                                            max_tokens=max_tokens, 
                                            seed=seed)
                outputs.extend(batch_outputs)
                pbar.update(len(batch_messages))
        
        for job, output in zip(valid_jobs, outputs):
            job['llm_output'] = output
        
        return valid_jobs
    
    def compute_scores(self, jobs) -> Dict[str, Any]:
        Type_match_num = 0
        Extact_match_num = 0
        # click_match_num = 0
        # input_match_num = 0
        # long_press_match_num = 0
        # swipe_match_num = 0
        # sb_match_num = 0
        # open_match_
        match_num_dict = {}
        history_len_dict = {}
        # all_action_dict = {}
        # all_click_num = 0
        error_num = 0
        think_cnt = 0
        for job in tqdm(jobs, desc='Computing scores'):
            output = job['llm_output']
            current_check_pam = job['check_pams']
            history_len = job['step_id'] + 1
            if '</thinking>' in output:
                think_cnt += 1

            if str(history_len) not in history_len_dict.keys():
                history_len_dict[f"{history_len}_all"] = 1
                history_len_dict[f"{history_len}"] = 0
            else:
                history_len_dict[f"{history_len}_all"] += 1


            if current_check_pam['action'] not in match_num_dict.keys():
                # all_click_num += 1
                match_num_dict[f"{current_check_pam['action']}_all"] = 1
                match_num_dict[f"{current_check_pam['action']}"] = 0
            else:
                match_num_dict[f"{current_check_pam['action']}_all"] += 1
            
            try:
                pred = output
                if '</thinking>' in pred:
                    pred = pred.split('</thinking>')[-1]
                if '<tool_call>' in pred:
                    pred = pred.split('<tool_call>')[1]
                else:
                    pred = '{"name": "mobile_use", "arguments":'+pred.split('{"name": "mobile_use", "arguments":',1)[1]
                if '</tool_call>' in pred:
                    pred = pred.split('</tool_call>')[0]
                else:
                    pred = pred.split("<conclusion>")[0]
                    pred = pred.rsplit('}}',1)[0]+'}}'

                pred_action = json.loads(pred.strip())['arguments']
                job['llm_prediction'] = pred_action

                type_match, extact_match = evaluate_android_control_action(pred_action, current_check_pam, job['width'], job['height'], job['resized_width'], job['resized_height'], pred_type ='abs_resized', gt_type='abs_resized')
                if type_match:
                    Type_match_num += 1
                    job['type_match'] = True
                
                if extact_match:
                    Extact_match_num += 1
                    job['extact_match'] = True
                    
                # if extact_match and pred_action['action'] == 'click':
                #     click_match_num += 1
                if extact_match:
                    match_num_dict[pred_action['action']] += 1
                    history_len_dict[f"{history_len}"] += 1
                    
                
            except:
                import traceback
                traceback.print_exc()
                print(output)
                print(job)
                error_num += 1
                continue
        
        print('Type_match_num and Extact_match_num: ', Type_match_num, Extact_match_num, '/ all =', len(jobs))
        # print('click_match_num:', click_match_num, '/ all =', all_click_num)
        print('error num', error_num)

        res = {
            'type_match_acc': Type_match_num/len(jobs)*100,
            'extact_match_acc': Extact_match_num/len(jobs)*100,
            # 'click_match_acc': click_match_num/all_click_num*100,
            'think_ratio': think_cnt/len(jobs)*100,
            'error_num': error_num,
        }
        for key in match_num_dict.keys():
            if key.endswith('_all'):
                continue
            res[f'{key}_acc'] = match_num_dict[key]/match_num_dict[f'{key}_all']*100
        for key in history_len_dict.keys():
            if key.endswith('_all'):
                continue
            res[f'history_len_{key}_acc'] = history_len_dict[key]/history_len_dict[f'{key}_all']*100
        # print(history_len_dict)
        print(json.dumps(res, indent=' '))

        model_name = self.model_path.split('/')[-1]
        prefix = self.prefix
        # output_dir = os.path.join(self.output_dir, model_name, 'android_control', self.eval_type, prefix)
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # timestamp = self.eval_file.split('/').replace(".json", "")
        with open(os.path.join(output_dir, f'scores_{timestamp}.json'), 'w') as f:
            json.dump(res, f, indent=2)
        

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Android Control Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--eval_type', type=str, required=True, choices=['high', 'low'], help='Evaluation type')
    parser.add_argument('--eval_file', type=str, required=True, default='./android_control_test.json', help='Path to the evaluation file')
    parser.add_argument('--image_root', type=str, required=True, default='./', help='Path to the image root')
    parser.add_argument('--output_dir', type=str, required=True, default='./', help='Path to the output directory')
    # parser.add_argument('--prefix', type=str, default="None", help='Prefix to add to the input')
    # parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    # parser.add_argument('--total_split', type=int, default=4)
    # parser.add_argument('--split', type=int, required=True)
    
    args = parser.parse_args()
    
    android_control = AndroidControl(
        model_path=args.model_path, 
        eval_file=args.eval_file,
        image_root=args.image_root,
        output_dir=args.output_dir,
        eval_type=args.eval_type, 
        # thinking=args.thinking, 
        # prefix=args.prefix,
        # debug=args.debug
    )
    # jobs = android_control.generate_jobs(total_split=args.total_split, split=args.split)
    # jobs = android_control.inference(jobs)
    
    model_name = args.model_path.split('/')[-1]
    # prefix = args.prefix
    output_dir = args.output_dir
    # output_dir = os.path.join(args.output_dir, model_name, 'android_control', args.eval_type, prefix)
    os.makedirs(output_dir, exist_ok=True)
    # with open(os.path.join(output_dir, f'jobs_{args.split}_{args.total_split}{"_debug" if args.debug else ""}.json'), 'w') as f:
    #     json.dump(jobs, f, indent=2)
    jobs = []
    for file in os.listdir(output_dir):
        if file.startswith('jobs_') and file.endswith('.json'):
            with open(os.path.join(output_dir, file), 'r') as f:
                part_data = json.load(f)
            
            jobs.extend(part_data)
    print(len(jobs))
    android_control.compute_scores(jobs)
    
