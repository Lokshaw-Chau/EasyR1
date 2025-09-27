import re
import json
import math

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

def extract_button(content):
    # answer_tag_pattern = r'<tool_call>(.*?)</tool_call>'
    action_pattern = r"\"button\":\s*\"(.*?)\""
    # content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    # if content_answer_match:
    #     content_answer = content_answer_match.group(1).strip()
    action_match = re.search(action_pattern, content)
    if action_match:
        return action_match.group(1)
    return "no input text"

def extract_status(content):
    # answer_tag_pattern = r'<tool_call>(.*?)</tool_call>'
    action_pattern = r"\"status\":\s*\"(.*?)\""
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
    
def r1gui_format_reward(predict_str: str, ground_truth: str) -> float:
    """
    检查 predict_str 是否符合 <thinking></thinking><answer></answer> 的格式，
    并验证 <answer> 中的内容是否符合 [{'action': 'action', 'point': '[x,y]', 'input_text': 'no input text'}] 的格式要求。
    """
    # 检查 <thinking> 和 <answer> 的外部结构
    outer_pattern_1 = re.compile(r"<thinking>.*?</thinking>\s*<tool_call>.*?</tool_call>", re.DOTALL)
    outer_pattern_2 = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
    if not re.fullmatch(outer_pattern_1, predict_str) and not re.fullmatch(outer_pattern_2, predict_str):
        return 0.0
    
    if '<thinking>' in predict_str and not re.fullmatch(outer_pattern_1, predict_str):
        return 0.0
    
    # 提取 <answer> 中的内容
    answer_match = re.search(r"<tool_call>(.*?)</tool_call>", predict_str, re.DOTALL)
    if not answer_match:
        return 0.0

    # 提取 <answer> 内的内容并解析为 JSON 格式
    answer_content = answer_match.group(1).strip()
    ui_type = json.loads(ground_truth).get("ui_type", "android_control")
    try:
        pred_action = extract_action(answer_content)
        if pred_action is None:
            return 0.0
        if ui_type == "android_control":
            if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'open', 'wait']:
                print(f"Invalid action: {pred_action} for ui_type: {ui_type}")
                return 0.0
        if ui_type == "gui_odyssey":
            if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'terminate']:
                print(f"Invalid action: {pred_action} for ui_type: {ui_type}")
                return 0.0

        if pred_action in ['click', 'long_press']:
            coord, valid = extract_coord(predict_str)
            if not valid:
                return 0.0
        elif pred_action in ['open','type']:
            pred_input_text = extract_input_text(answer_content)
            if pred_input_text == "no input text":
                return 0.0
        elif pred_action in ['system_button']:
            button = extract_button(answer_content)
            if button == "no input text":
                return 0.0
        elif pred_action in ['terminate']:
            status = extract_status(answer_content)
            if status == "no input text":
                return 0.0
        elif pred_action in ['swipe']:
            pred_coord, valid1 = extract_coord(answer_content)
            pred_coord2, valid2 = extract_coord2(answer_content)
            if not (valid1 and valid2):
                return 0.0
        elif pred_action in ['wait']:
            return 1.0
        
        else:
            print(f"Unknown action: {pred_action}")
            return 0.0
        # 如果所有检查均通过，返回 1.0
        return 1.0
    except:
        return 0.0

def r1gui_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    比较 predict_str 和 ground_truth 中的动作和参数是否一致。
    """
    try:
        # 提取 ground_truth 的动作和参数
        ground_truth=json.loads(ground_truth)
        gt_action=ground_truth['action'].lower()
        gt_bbox=ground_truth['gt_bbox']
        gt_input_text=ground_truth['input_text']
        pred_action=extract_action(predict_str).lower()
        ui_type = ground_truth["ui_type"]
        # pred_input_text=extract_input_text(predict_str)
        # pred_bbox , _ =extract_coord(predict_str)
        
        if pred_action!=gt_action:
            return 0.0
        
        if ui_type == "android_control":
            if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'open', 'wait']:
                return 0.0
        if ui_type == "gui_odyssey":
            if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'terminate']:
                return 0.0

        if gt_action in ["click", "long_press"]:
            pred_bbox , _ =extract_coord(predict_str)
            if len(gt_bbox)==2:
                if ((pred_bbox[0]-gt_bbox[0])/ground_truth['image_size'][0])**2+((pred_bbox[1]-gt_bbox[1])/ground_truth['image_size'][1])**2 < 0.14**2:
                    return 1.0
                else:
                    return 0.0
            elif len(gt_bbox)==4:
                if (gt_bbox[0]<pred_bbox[0]<gt_bbox[2]) and (gt_bbox[1]<pred_bbox[1]<gt_bbox[3]):
                    return 1.0
                else:
                    return 0.0
        elif pred_action in ['open','type']:
            pred_input_text = extract_input_text(predict_str)
            if calculate_f1_score(pred_input_text, gt_input_text)>=0.5:
                return 1.0
            else:
                return 0.0
        elif pred_action in ['system_button']:
            pred_button = extract_button(predict_str)
            if calculate_f1_score(pred_button, gt_input_text)>=0.5:
                return 1.0
            else:
                return 0.0
        elif pred_action in ['swipe']:
            pred_coord, valid1 = extract_coord(predict_str)
            pred_coord2, valid2 = extract_coord2(predict_str)
            if not (valid1 and valid2):
                return 0.0
            x1, y1 = pred_coord
            x2, y2 = pred_coord2
            delta_x = x2 - x1
            delta_y = y2 - y1
            if abs(delta_x) > abs(delta_y):
                if delta_x > 0:
                    pred_direction = 'right'
                else:
                    pred_direction = 'left'
            else:
                if delta_y > 0:
                    pred_direction = 'down'
                else:
                    pred_direction = 'up'

            if pred_direction == gt_input_text:
                return 1.0
            else:
                return 0.0
        elif pred_action in ['terminate']:
            pred_status = extract_status(predict_str)
            if calculate_f1_score(pred_status, gt_input_text)>=0.5:
                return 1.0
            else:
                return 0.0
        
        elif pred_action in ['wait']:
            return 1.0
        else:
            print(f"Unknown action: {pred_action}")
            return 0.0
        
    except Exception as e:
        return 0.0
    
def _compute_score(predict_str: str, ground_truth: str, think_ratio: float = 1.0, training_progress: float = None):
    format = r1gui_format_reward(predict_str, ground_truth)
    accuracy = r1gui_accuracy_reward(predict_str, ground_truth)
    
    # Calculate base score
    base_score = accuracy + format
    mode_ratio = think_ratio if "<thinking>" in predict_str else 1 - think_ratio
    scale_factor = 1 / mode_ratio
    # Apply progressive scaling based on training progress
    overall_score = base_score

    return {
        "overall": overall_score,
        "format": format,
        "accuracy": accuracy,
        "think_ratio": 1.0 if "<thinking>" in predict_str else 0.0,
        "training_progress": training_progress if training_progress is not None else 0.0,
        "think_acc": accuracy*scale_factor if "<thinking>" in predict_str else 0.0,
        "no_think_acc": accuracy*scale_factor if "<thinking>" not in predict_str else 0.0,
    }

def think_ratio(predict_strs: list[str]):
    """
    计算 predict_strs 中 <thinking> ... </thinking> 的比例。
    """
    think_count = sum(1 for s in predict_strs if "<thinking>" in s)
    total_count = len(predict_strs)
    
    if total_count == 0:
        return 0.0
    
    return think_count / total_count

def _batch_wise_penalty_reward(scores, ground_truths, gamma):
    think_ratio = sum(score["think_ratio"] for score in scores) / len(scores)
    if think_ratio < gamma:
        for score in scores:
            if score["think_ratio"] < 0.5:
                score["overall"] -= 2
    elif think_ratio > 1 - gamma:
        for score in scores:
            if score["think_ratio"] >= 0.5:
                score["overall"] -= 2
    else:
        print("No batch-wise penalty applied, think_ratio:", think_ratio)
    return scores

def _group_wise_bias(scores, ground_truths):
    gt2tr_list = {}
    for i, score in enumerate(scores):
        ground_truth = ground_truths[i]
        if ground_truth not in gt2tr_list:
            gt2tr_list[ground_truth] = []
        gt2tr_list[ground_truth].append(score["think_ratio"])

    gt2tr = {k: sum(v) / len(v) for k, v in gt2tr_list.items()}
    # print("gt2tr:", gt2tr)

    for i, score in enumerate(scores):
        tr = gt2tr[ground_truths[i]]
        score["tnt_bias"] = abs(tr - 0.5)
        
    return scores

def compute_score(predict_strs: list[str], ground_truths: list[str], training_progress: float = None):
    scores = []
    current_think_ratio = think_ratio(predict_strs)
    for predict_str, ground_truth in zip(predict_strs, ground_truths):
        scores.append(_compute_score(predict_str, ground_truth, current_think_ratio, None))
    
    scores = _batch_wise_penalty_reward(scores, ground_truths, gamma=0.1)

    scores = _group_wise_bias(scores, ground_truths)

    return scores

if __name__ == "__main__":
    pr=["<thinking> I need to go back to see the brand option. </thinking>  \n<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"system_button\", \"button\": \"Back\"}}</tool_call>"]
    gt=[json.dumps({"action": "system_button", "gt_bbox": [-1.0, -1.0], "input_text": "Back", "image_size": [1080, 1920]})]
    # print(r1gui_accuracy_reward(pr,gt))
    print(compute_score(pr, gt))