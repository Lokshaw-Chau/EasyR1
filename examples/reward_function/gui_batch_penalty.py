import re
import json
import math

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

def extract_action(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'action':\s*'(\w+)'"
    action_pattern2 = r"action':\s*(\w+)"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        action_match_2 = re.search(action_pattern2, content_answer)
        if action_match:
            return action_match.group(1)
        if action_match_2:
            return action_match_2.group(1) 
    return "no action"

def extract_input_text(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'input_text':\s*'(.*?)'"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no input text"

def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<tool_call>(.*?)</tool_call>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
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
    
def r1gui_format_reward(predict_str: str) -> float:
    """
    检查 predict_str 是否符合 <think></think><answer></answer> 的格式，
    并验证 <answer> 中的内容是否符合 [{'action': 'action', 'point': '[x,y]', 'input_text': 'no input text'}] 的格式要求。
    """
    # 检查 <think> 和 <answer> 的外部结构
    outer_pattern_1 = re.compile(r"<think>.*?</think>\s*<tool_call>.*?\"action\": \"click\",.*?</tool_call>", re.DOTALL)
    outer_pattern_2 = re.compile(r"<tool_call>.*?\"action\": \"click\",.*?</tool_call>", re.DOTALL)
    if not re.fullmatch(outer_pattern_1, predict_str) and not re.fullmatch(outer_pattern_2, predict_str):
        return 0.0

    # # 提取 <answer> 中的内容
    # answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    # if not answer_match:
    #     return 0.0

    # # 提取 <answer> 内的内容并解析为 JSON 格式
    # answer_content = answer_match.group(1).strip()
    # try:
    #     actions = eval(answer_content)  # 尝试将 <answer> 的内容解析为 JSON

    #     # 验证 actions 是否为列表
    #     if not isinstance(actions, list):
    #         return 0.0

    #     # 验证每个 action 的格式
    #     for action in actions:
    #         if not isinstance(action, dict):
    #             return 0.0
    #         # 检查 action 字典是否包含所需的键
    #         if "action" not in action or "point" not in action or "input_text" not in action:
    #             return 0.0
    #         # 验证 action 的值是否符合要求
    #         if not isinstance(action["action"], str):
    #             return 0.0
    #         if not (isinstance(action["point"][0],int) and isinstance(action["point"][1],int)):  # 匹配形如 [x,y] 的点
    #             return 0.0
    #         if not isinstance(action["input_text"], str):
    #             return 0.0
    #         if action["action"] in ['type', 'select','open_app'] and action["input_text"] in ['no input text']:
    #             return 0.0
    #         if action["action"] in ['scroll'] and action["input_text"] not in ['left','right','up','down']:
    #             return 0.0

        # 如果所有检查均通过，返回 1.0
    return 1.0
    # except:
    #     return 0.0

def r1gui_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    比较 predict_str 和 ground_truth 中的动作和参数是否一致。
    """
    try:
        # 提取 ground_truth 的动作和参数
        ground_truth=json.loads(ground_truth)
        # gt_action=ground_truth['action'].lower()
        gt_bbox=ground_truth['gt_bbox']
        # gt_input_text=ground_truth['input_text']
        # pred_action=extract_action(predict_str).lower()
        # pred_input_text=extract_input_text(predict_str)
        pred_bbox , _ =extract_coord(predict_str)
        
        # if pred_action!=gt_action:
        #     return 0.0
        
        # if gt_action in ["click"]:
        # if len(gt_bbox)==2:
        #     if (pred_bbox[0]-gt_bbox[0])**2+(pred_bbox[1]-gt_bbox[1])**2<140**2:
        #         return 1.0
        #     else:
        #         return 0.0
        # elif len(gt_bbox)==4:
        if (gt_bbox[0]<pred_bbox[0]<gt_bbox[2]) and (gt_bbox[1]<pred_bbox[1]<gt_bbox[3]):
            return 1.0
        else:
            return 0.0
        # elif gt_action in ['type', 'select','scroll']:
        #     if calculate_f1_score(pred_input_text,gt_input_text)>=0.5:
        #         return 1.0
        #     else:
        #         return 0.0
        # else:
        #     return 1.0

    except Exception as e:
        return 0.0
    
def _compute_score(predict_str: str, ground_truth: str, think_ratio: float = 1.0, training_progress: float = None):
    format = r1gui_format_reward(predict_str)
    accuracy = r1gui_accuracy_reward(predict_str, ground_truth)
    
    base_score = accuracy + format
    # Calculate base score
    if accuracy > 0 and "<think>" not in predict_str:
        base_score += 0.2
    
    mode_ratio = think_ratio if "<think>" in predict_str else 1 - think_ratio
    scale_factor = 1 / mode_ratio
    # Apply progressive scaling based on training progress
    if training_progress is not None:
        # AdaGRPO
        decay = mode_ratio + 0.5 * (1 - mode_ratio) * (1 + math.cos(math.pi * training_progress))
        overall_score = base_score * scale_factor * decay
    else:
        # # Fallback to original behavior when training_progress is not available
        # overall_score = base_score
        # extra score for none think
        overall_score = base_score

    return {
        "overall": overall_score,
        "format": format,
        "accuracy": accuracy,
        "think_ratio": 1.0 if "<think>" in predict_str else 0.0,
        "training_progress": training_progress if training_progress is not None else 0.0,
        "decay": decay if training_progress is not None else 1.0,
        "think_acc": accuracy*scale_factor if "<think>" in predict_str else 0.0,
        "no_think_acc": accuracy*scale_factor if "<think>" not in predict_str else 0.0,
    }

def _collapse_penalty(scores, ground_truths, theta):
    # group-wise collapse penalty
    # think_ratio = sum(score["think_ratio"] for score in scores) / len(scores)
    gt2tr_list = {}
    for i, score in enumerate(scores):
        ground_truth = ground_truths[i]
        if ground_truth not in gt2tr_list:
            gt2tr_list[ground_truth] = []
        gt2tr_list[ground_truth].append(score["think_ratio"])

    gt2tr = {k: sum(v) / len(v) for k, v in gt2tr_list.items()}
    print("gt2tr:", gt2tr)

    for i, score in enumerate(scores):
        tr = gt2tr[ground_truths[i]] 

        if tr < theta:
            for score in scores:
                if score["think_ratio"]<0.5: # no think
                    score["overall"] = score["overall"] - 2
        if tr > 1-theta:
            for score in scores:
                if score["think_ratio"]>=0.5: # think
                    score["overall"] = score["overall"] - 2
        
    return scores

# def _preferential_reward(scores, ground_truths, gamma):
#     gt2nt_acc={}
#     gt2t_acc={}
#     for i, score in enumerate(scores):
#         ground_truth = ground_truths[i]
#         if score["think_ratio"] > 0.5:  # think
#             if ground_truth not in gt2t_acc:
#                 gt2t_acc[ground_truth] = []
#             gt2t_acc[ground_truth].append(score["accuracy"])
#         else:
#             if ground_truth not in gt2nt_acc:
#                 gt2nt_acc[ground_truth] = []
#             gt2nt_acc[ground_truth].append(score["accuracy"])

#     gt2t_acc = {k: sum(v) / len(v) for k, v in gt2t_acc.items()}
#     gt2nt_acc = {k: sum(v) / len(v) for k, v in gt2nt_acc.items()}

#     for i, score in enumerate(scores):
#         ground_truth = ground_truths[i]
#         if gt2t_acc[ground_truth] <= gt2nt_acc[ground_truth]:

def think_ratio(predict_strs: list[str]):
    """
    计算 predict_strs 中 <think> ... </think> 的比例。
    """
    think_count = sum(1 for s in predict_strs if "<think>" in s)
    total_count = len(predict_strs)
    
    if total_count == 0:
        return 0.0
    
    return think_count / total_count
def compute_score(predict_strs: list[str], ground_truths: list[str], training_progress: float = None):
    scores = []
    current_think_ratio = think_ratio(predict_strs)
    for predict_str, ground_truth in zip(predict_strs, ground_truths):
        scores.append(_compute_score(predict_str, ground_truth, current_think_ratio, None))

    scores = _collapse_penalty(scores, ground_truths, 0.2)
    return scores

# pr=("<think> The command 'What's on the menu at IHOP?' suggests a search for information about the menu at an IHOP restaurant. However, "
# "the current UI screenshot is a calendar application displaying holidays and significant dates for the month of October and November. There is no direct way to per"
# "form a web search or access an IHOP menu from this calendar app. Therefore, the appropriate action would be to exit the current application and open a web browser"
# "or a dedicated app for searching the IHOP menu. "                                                                                                               
# "Since the action history is 'None', the first step is to navigate away from the current app to a web browser or a search engine.</think> "
# " <answer>[{'action': 'scroll', 'point': [123, 401], 'input_text': 'left'}]</answer>")
# gt=json.dumps({"action": "scroll", "gt_bbox": [103.0, 409.18800000000005], "input_text": "LEFT"})
# print(gr_iou_accuracy_reward(pr,gt))
# print(gr_format_reward(pr))