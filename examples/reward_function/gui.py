import re
import json
from typing import Any
# import math

def extract_tool_call_json(content):
    """
    Extract and parse JSON from <tool_call></tool_call> tags.
    Returns a dict with the parsed arguments or None if parsing fails.

    Expected format:
    <tool_call>
    {"name": "mobile_use", "arguments": {"action": "...", ...}}
    </tool_call>
    """
    try:
        # Extract content within <tool_call> tags
        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if not tool_call_match:
            return None

        tool_call_content = tool_call_match.group(1).strip()

        # Parse JSON
        parsed_json = json.loads(tool_call_content)

        # Return the arguments dict if it exists
        if isinstance(parsed_json, dict) and "arguments" in parsed_json:
            return parsed_json["arguments"]

        # If no "arguments" key, assume the entire JSON is the arguments
        return parsed_json
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        return None

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
    检查 predict_str 是否符合 <thinking></thinking><tool_call></tool_call> 的格式，
    并验证 <tool_call> 中的内容是否符合正确的 JSON 格式和动作要求。
    """
    # 检查 <thinking> 和 <tool_call> 的外部结构
    outer_pattern_1 = re.compile(r"<thinking>.*?</thinking>\s*<tool_call>.*?</tool_call>", re.DOTALL)
    outer_pattern_2 = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
    if not re.fullmatch(outer_pattern_1, predict_str) and not re.fullmatch(outer_pattern_2, predict_str):
        return 0.0

    if '<thinking>' in predict_str and not re.fullmatch(outer_pattern_1, predict_str):
        return 0.0

    # 使用 JSON 解析提取 tool_call 内容
    parsed_args = extract_tool_call_json(predict_str)
    if parsed_args is None:
        return 0.0

    ui_type = json.loads(ground_truth).get("ui_type", "android_control")
    try:
        # 获取 action
        pred_action = parsed_args.get("action")
        if pred_action is None:
            return 0.0

        # 验证 action 是否符合 ui_type 要求
        if ui_type == "android_control":
            if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'open', 'wait']:
                return 0.0

        if ui_type == "gui_odyssey":
            if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'terminate']:
                return 0.0

        if ui_type == "agentnetbench":
            if pred_action not in ['key', 'type', 'mouse_move', 'left_click', 'right_click', 'double_click', 'scroll', 'terminate', 'left_click_drag']:
                return 0.0

        # 验证必需的参数
        if pred_action in ['click', 'long_press', 'mouse_move', 'left_click', 'right_click', 'double_click', 'left_click_drag']:
            coord = parsed_args.get("coordinate")
            if coord is None or not isinstance(coord, list) or len(coord) != 2:
                return 0.0
        elif pred_action in ['open','type']:
            pred_input_text = parsed_args.get("text")
            if pred_input_text is None:
                return 0.0
        elif pred_action in ['system_button']:
            button = parsed_args.get("button")
            if button is None:
                return 0.0
        elif pred_action in ['key']:
            keys = parsed_args.get("keys")
            if keys is None:
                return 0.0
        elif pred_action in ['terminate']:
            status = parsed_args.get("status")
            if status is None:
                return 0.0
        elif pred_action in ['swipe']:
            coord = parsed_args.get("coordinate")
            coord2 = parsed_args.get("coordinate2")
            if (coord is None or not isinstance(coord, list) or len(coord) != 2 or
                coord2 is None or not isinstance(coord2, list) or len(coord2) != 2):
                return 0.0
        elif pred_action in ['wait', 'scroll']:
            return 1.0
        else:
            print(f"Unknown action: {pred_action}")
            return 0.0

        # 如果所有检查均通过，返回 1.0
        return 1.0
    except Exception as e:
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
        ui_type = ground_truth["ui_type"]

        # 使用 JSON 解析提取预测内容
        parsed_args = extract_tool_call_json(predict_str)
        if parsed_args is None:
            return 0.0

        pred_action = parsed_args.get("action")
        if pred_action is None:
            return 0.0

        pred_action = pred_action.lower()

        if pred_action != gt_action:
            return 0.0

        # 验证 action 是否符合 ui_type 要求
        if ui_type == "android_control":
            if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'open', 'wait']:
                return 0.0
        if ui_type == "gui_odyssey":
            if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'terminate']:
                return 0.0
        if ui_type == "agentnetbench":
            if pred_action not in ['key', 'type', 'mouse_move', 'left_click', 'right_click', 'double_click', 'scroll', 'terminate', 'left_click_drag']:
                return 0.0

        # 验证参数准确性
        if gt_action in ["click", "long_press", "mouse_move", "left_click", "right_click", "double_click", "left_click_drag"]:
            pred_bbox = parsed_args.get("coordinate")
            if pred_bbox is None or not isinstance(pred_bbox, list) or len(pred_bbox) != 2:
                return 0.0
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
            pred_input_text = parsed_args.get("text")
            if pred_input_text is None:
                return 0.0
            if calculate_f1_score(pred_input_text, gt_input_text)>=0.5:
                return 1.0
            else:
                return 0.0
        elif pred_action in ['system_button']:
            pred_button = parsed_args.get("button")
            if pred_button is None:
                return 0.0
            if calculate_f1_score(pred_button, gt_input_text)>=0.5:
                return 1.0
            else:
                return 0.0
        elif pred_action in ['key']:
            pred_keys = parsed_args.get("keys")
            if pred_keys is None:
                return 0.0
            # Convert to string if it's not already
            pred_keys_str = str(pred_keys) if not isinstance(pred_keys, str) else pred_keys
            if 'keys=' in gt_input_text:
                gt_input_text = gt_input_text.replace('keys=','').strip()
            if calculate_f1_score(pred_keys_str, gt_input_text)>=0.5:
                return 1.0
            else:
                return 0.0
        elif pred_action in ['swipe']:
            pred_coord = parsed_args.get("coordinate")
            pred_coord2 = parsed_args.get("coordinate2")
            if (pred_coord is None or not isinstance(pred_coord, list) or len(pred_coord) != 2 or
                pred_coord2 is None or not isinstance(pred_coord2, list) or len(pred_coord2) != 2):
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
            pred_status = parsed_args.get("status")
            if pred_status is None:
                return 0.0
            if calculate_f1_score(pred_status, gt_input_text)>=0.5:
                return 1.0
            else:
                return 0.0
        elif pred_action in ['wait', 'scroll']:
            return 1.0
        else:
            print(f"Unknown action: {pred_action}")
            return 0.0

    except Exception as e:
        print(f"Error in accuracy reward calculation: {e}")
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

def _pass_at_accracy_for_each_query(scores, ground_truths):
    gt2think_acc = {}
    gt2nothink_acc = {}
    gt2acc = {}
    for i, score in enumerate(scores):
        ground_truth = ground_truths[i]
        if ground_truth not in gt2think_acc:
            gt2think_acc[ground_truth] = []
            gt2nothink_acc[ground_truth] = []
            gt2acc[ground_truth] = []
            
        if score["think_ratio"] > 0.5:
            gt2think_acc[ground_truth].append(score["think_acc"])
        else:
            gt2nothink_acc[ground_truth].append(score["no_think_acc"])
        
        gt2acc[ground_truth].append(score["accuracy"])
    
    for i, score in enumerate(scores):
        think_acc = max(gt2think_acc[ground_truths[i]]) if len(gt2think_acc[ground_truths[i]]) != 0 else 0
        pass_at_1_acc_think = gt2think_acc[ground_truths[i]][0] if len(gt2think_acc[ground_truths[i]]) != 0 else 0
        no_think_acc = max(gt2nothink_acc[ground_truths[i]]) if len(gt2nothink_acc[ground_truths[i]]) != 0 else 0
        pass_at_1_acc_nothink = gt2nothink_acc[ground_truths[i]][0] if len(gt2nothink_acc[ground_truths[i]]) != 0 else 0
        total_acc = max(gt2acc[ground_truths[i]])
        score["pass_at_accuracy"] = total_acc
        if score["think_ratio"] > 0.5:
            score["think_pass_at_accuracy"] = think_acc
            score["think_pass_at_1_accuracy"] = pass_at_1_acc_think
            score["nothink_pass_at_1_accuracy"] = 0.0
            score["nothink_pass_at_accuracy"] = 0.0
        else:
            score["nothink_pass_at_accuracy"] = no_think_acc
            score["nothink_pass_at_1_accuracy"] = pass_at_1_acc_nothink
            score["think_pass_at_1_accuracy"] = 0.0
            score["think_pass_at_accuracy"] = 0.0
    return scores

def compute_score(reward_input: list[dict[str, Any]], format_weight: float = 0.5) -> dict[str, float]:
    if not isinstance(reward_input, list):
        raise ValueError("Please use `reward_type=batch` for gui reward function.")

    scores = []
    
    predict_strs = [item["response"] for item in reward_input]
    ground_truths = [item["ground_truth"] for item in reward_input]
    current_think_ratio = think_ratio(predict_strs)
    for predict_str, ground_truth in zip(predict_strs, ground_truths):
        scores.append(_compute_score(predict_str, ground_truth, current_think_ratio, None))

    scores = _pass_at_accracy_for_each_query(scores, ground_truths)

    return scores

if __name__ == "__main__":
    # Test cases for all action types across different ui_types
    pr = [
        # 1. click (android_control) - with thinking
        "<thinking>I need to click the button</thinking>\n<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"click\", \"coordinate\": [540, 960]}}</tool_call>",
        # 2. click (android_control) - without thinking
        "<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"click\", \"coordinate\": [540, 960]}}</tool_call>",
        # 3. long_press (android_control)
        "<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"long_press\", \"coordinate\": [100, 200]}}</tool_call>",
        # 4. swipe (android_control)
        "<thinking>Swipe down to scroll</thinking>\n<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"swipe\", \"coordinate\": [540, 500], \"coordinate2\": [540, 1200]}}</tool_call>",
        # 5. type (android_control)
        "<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"type\", \"text\": \"hello world\"}}</tool_call>",
        # 6. system_button (android_control)
        "<thinking>I need to go back</thinking>\n<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"system_button\", \"button\": \"Back\"}}</tool_call>",
        # 7. open (android_control)
        "<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"open\", \"text\": \"com.example.app\"}}</tool_call>",
        # 8. wait (android_control)
        "<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"wait\"}}</tool_call>",
        # 9. terminate (gui_odyssey)
        "<thinking>Task completed</thinking>\n<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"terminate\", \"status\": \"success\"}}</tool_call>",
        # 10. mouse_move (agentnetbench)
        "<tool_call>\n{\"name\": \"computer\", \"arguments\": {\"action\": \"mouse_move\", \"coordinate\": [800, 600]}}</tool_call>",
        # 11. left_click (agentnetbench)
        "<thinking>Click the element</thinking>\n<tool_call>\n{\"name\": \"computer\", \"arguments\": {\"action\": \"left_click\", \"coordinate\": [800, 600]}}</tool_call>",
        # 12. right_click (agentnetbench)
        "<tool_call>\n{\"name\": \"computer\", \"arguments\": {\"action\": \"right_click\", \"coordinate\": [800, 600]}}</tool_call>",
        # 13. double_click (agentnetbench)
        "<tool_call>\n{\"name\": \"computer\", \"arguments\": {\"action\": \"double_click\", \"coordinate\": [800, 600]}}</tool_call>",
        # 14. key (agentnetbench)
        "<thinking>Press enter key</thinking>\n<tool_call>\n{\"name\": \"computer\", \"arguments\": {\"action\": \"key\", \"keys\": \"Return\"}}</tool_call>",
        # 15. scroll (agentnetbench)
        "<tool_call>\n{\"name\": \"computer\", \"arguments\": {\"action\": \"scroll\"}}</tool_call>",
        # 16. left_click_drag (agentnetbench)
        "<tool_call>\n{\"name\": \"computer\", \"arguments\": {\"action\": \"left_click_drag\", \"coordinate\": [100, 100]}}</tool_call>",
    ]

    gt = [
        # 1. click - exact match
        json.dumps({"action": "click", "gt_bbox": [540, 960], "input_text": "", "image_size": [1080, 1920], "ui_type": "android_control"}),
        # 2. click - exact match
        json.dumps({"action": "click", "gt_bbox": [540, 960], "input_text": "", "image_size": [1080, 1920], "ui_type": "android_control"}),
        # 3. long_press - within bbox
        json.dumps({"action": "long_press", "gt_bbox": [50, 150, 150, 250], "input_text": "", "image_size": [1080, 1920], "ui_type": "android_control"}),
        # 4. swipe - direction down
        json.dumps({"action": "swipe", "gt_bbox": [-1, -1], "input_text": "down", "image_size": [1080, 1920], "ui_type": "android_control"}),
        # 5. type - text match
        json.dumps({"action": "type", "gt_bbox": [-1, -1], "input_text": "hello world", "image_size": [1080, 1920], "ui_type": "android_control"}),
        # 6. system_button - button match
        json.dumps({"action": "system_button", "gt_bbox": [-1, -1], "input_text": "Back", "image_size": [1080, 1920], "ui_type": "android_control"}),
        # 7. open - app match
        json.dumps({"action": "open", "gt_bbox": [-1, -1], "input_text": "com.example.app", "image_size": [1080, 1920], "ui_type": "android_control"}),
        # 8. wait - always passes
        json.dumps({"action": "wait", "gt_bbox": [-1, -1], "input_text": "", "image_size": [1080, 1920], "ui_type": "android_control"}),
        # 9. terminate - status match
        json.dumps({"action": "terminate", "gt_bbox": [-1, -1], "input_text": "success", "image_size": [1920, 1080], "ui_type": "gui_odyssey"}),
        # 10. mouse_move - coordinate match
        json.dumps({"action": "mouse_move", "gt_bbox": [800, 600], "input_text": "", "image_size": [1920, 1080], "ui_type": "agentnetbench"}),
        # 11. left_click - coordinate match
        json.dumps({"action": "left_click", "gt_bbox": [800, 600], "input_text": "", "image_size": [1920, 1080], "ui_type": "agentnetbench"}),
        # 12. right_click - coordinate match
        json.dumps({"action": "right_click", "gt_bbox": [800, 600], "input_text": "", "image_size": [1920, 1080], "ui_type": "agentnetbench"}),
        # 13. double_click - coordinate match
        json.dumps({"action": "double_click", "gt_bbox": [800, 600], "input_text": "", "image_size": [1920, 1080], "ui_type": "agentnetbench"}),
        # 14. key - keys match
        json.dumps({"action": "key", "gt_bbox": [-1, -1], "input_text": "Return", "image_size": [1920, 1080], "ui_type": "agentnetbench"}),
        # 15. scroll - always passes
        json.dumps({"action": "scroll", "gt_bbox": [-1, -1], "input_text": "", "image_size": [1920, 1080], "ui_type": "agentnetbench"}),
        # 16. left_click_drag - coordinate match
        json.dumps({"action": "left_click_drag", "gt_bbox": [100, 100], "input_text": "", "image_size": [1920, 1080], "ui_type": "agentnetbench"}),
    ]

    print("Testing all action types...")
    scores = compute_score(pr, gt)

    # Print results for each test case
    action_types = [
        "click (with thinking)", "click (no thinking)", "long_press", "swipe",
        "type", "system_button", "open", "wait", "terminate", "mouse_move",
        "left_click", "right_click", "double_click", "key", "scroll", "left_click_drag"
    ]

    print("\n" + "="*80)
    print("Test Results Summary:")
    print("="*80)
    all_passed = True
    for i, (action_name, score) in enumerate(zip(action_types, scores)):
        status = "✓ PASS" if score['format'] == 1.0 and score['accuracy'] == 1.0 else "✗ FAIL"
        if score['format'] != 1.0 or score['accuracy'] != 1.0:
            all_passed = False
        print(f"{i+1:2d}. {action_name:25s} | Format: {score['format']:.1f} | Accuracy: {score['accuracy']:.1f} | {status}")

    print("="*80)
    if all_passed:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("="*80)