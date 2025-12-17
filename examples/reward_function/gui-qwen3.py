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
    检查 predict_str 是否符合 Thought:, Action: <tool_call></tool_call> 的格式，
    并验证 <tool_call> 中的内容是否符合正确的 JSON 格式和动作要求。
    """
    # 检查 <thinking> 和 <tool_call> 的外部结构
    outer_pattern_1 = re.compile(r"Thought:.*?\s*Action:.*?\s<tool_call>.*?</tool_call>", re.DOTALL)
    outer_pattern_2 = re.compile(r"Action:.*?\s<tool_call>.*?</tool_call>", re.DOTALL)
    if not re.fullmatch(outer_pattern_1, predict_str) and not re.fullmatch(outer_pattern_2, predict_str):
        return 0.0

    # if '<thinking>' in predict_str and not re.fullmatch(outer_pattern_1, predict_str):
    #     return 0.0

    # 使用 JSON 解析提取 tool_call 内容
    parsed_args = extract_tool_call_json(predict_str)
    if parsed_args is None:
        return 0.0

    # ui_type = json.loads(ground_truth).get("ui_type", "android_control")
    try:
        # 获取 action
        pred_action = parsed_args.get("action")
        if pred_action is None:
            return 0.0

        # 验证 action 是否符合 ui_type 要求
        # if ui_type == "android_control":
        #     if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'open', 'wait']:
        #         return 0.0

        # if ui_type == "gui_odyssey":
        #     if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'terminate']:
        #         return 0.0

        # if ui_type == "agentnetbench":
        #     if pred_action not in ['key', 'type', 'mouse_move', 'left_click', 'right_click', 'double_click', 'scroll', 'terminate', 'left_click_drag']:
        #         return 0.0

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
        # ui_type = ground_truth["ui_type"]

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
        # if ui_type == "android_control":
        #     if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'open', 'wait']:
        #         return 0.0
        # if ui_type == "gui_odyssey":
        #     if pred_action not in ['click', 'long_press', 'swipe', 'type', 'system_button', 'terminate']:
        #         return 0.0
        # if ui_type == "agentnetbench":
        #     if pred_action not in ['key', 'type', 'mouse_move', 'left_click', 'right_click', 'double_click', 'scroll', 'terminate', 'left_click_drag']:
        #         return 0.0

        # 验证参数准确性
        if gt_action in ["click", "long_press", "mouse_move", "left_click", "right_click", "double_click", "left_click_drag"]:
            pred_bbox = parsed_args.get("coordinate")
            if pred_bbox is None or not isinstance(pred_bbox, list) or len(pred_bbox) != 2:
                return 0.0
            if len(gt_bbox)==2:
                if ((pred_bbox[0]-gt_bbox[0]))**2+((pred_bbox[1]-gt_bbox[1]))**2 < 0.14**2:
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
    gt2acc = {}
    for i, score in enumerate(scores):
        ground_truth = ground_truths[i]
        if ground_truth not in gt2acc:
            gt2acc[ground_truth] = []
            
        
        gt2acc[ground_truth].append(score["accuracy"])
    
    for i, score in enumerate(scores):
        total_acc = max(gt2acc[ground_truths[i]])
        score["pass_at_accuracy"] = total_acc
    return scores

def compute_score(reward_input: list[dict[str, Any]], format_weight: float = 0.5) -> dict[str, float]:
    if not isinstance(reward_input, list):
        raise ValueError("Please use `reward_type=batch` for gui reward function.")

    scores = []
    
    predict_strs = [item["response"] for item in reward_input]
    ground_truths = [item["ground_truth"] for item in reward_input]
    
    # current_think_ratio = think_ratio(predict_strs)
    for predict_str, ground_truth in zip(predict_strs, ground_truths):
        format_score = r1gui_format_reward(predict_str, ground_truth)
        accuracy_score = r1gui_accuracy_reward(predict_str, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score if format_score > 0 else 0.0,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    scores = _pass_at_accracy_for_each_query(scores, ground_truths)

    return scores

if __name__ == "__main__":
    # Test cases for all action types across different ui_types
    print("=" * 80)
    print("Testing GUI Reward Function")
    print("=" * 80)

    # Test Case 1: android_control - click action (correct)
    test_cases = [
        {
            "name": "android_control - click (correct format & accuracy)",
            "predict": 'Thought: I need to click on the button.\nAction: Click the button at center.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 300]}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "click",
                "gt_bbox": [400, 200, 600, 400],  # bbox format: [x1, y1, x2, y2]
                "input_text": "",
                "ui_type": "android_control"
            }),
            "expected_format": 1.0,
            "expected_accuracy": 1.0
        },
        {
            "name": "android_control - click (correct format, wrong coordinate)",
            "predict": 'Thought: I need to click on the button.\nAction: Click the button.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [100, 100]}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "click",
                "gt_bbox": [400, 200, 600, 400],
                "input_text": "",
                "ui_type": "android_control"
            }),
            "expected_format": 1.0,
            "expected_accuracy": 0.0
        },
        {
            "name": "android_control - type action (correct)",
            "predict": 'Thought: I should type the text.\nAction: Type "Hello World".\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "type", "text": "Hello World"}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "type",
                "gt_bbox": [],
                "input_text": "Hello World",
                "ui_type": "android_control"
            }),
            "expected_format": 1.0,
            "expected_accuracy": 1.0
        },
        {
            "name": "android_control - swipe action (correct)",
            "predict": 'Thought: Swipe up to scroll.\nAction: Swipe upward.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "swipe", "coordinate": [500, 800], "coordinate2": [500, 200]}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "swipe",
                "gt_bbox": [],
                "input_text": "up",
                "ui_type": "android_control"
            }),
            "expected_format": 1.0,
            "expected_accuracy": 1.0
        },
        {
            "name": "android_control - swipe (wrong direction)",
            "predict": 'Thought: Swipe down.\nAction: Swipe downward.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "swipe", "coordinate": [500, 200], "coordinate2": [500, 800]}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "swipe",
                "gt_bbox": [],
                "input_text": "up",
                "ui_type": "android_control"
            }),
            "expected_format": 1.0,
            "expected_accuracy": 0.0
        },
        {
            "name": "gui_odyssey - terminate action (correct)",
            "predict": 'Thought: Task completed successfully.\nAction: Terminate with success.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "terminate",
                "gt_bbox": [],
                "input_text": "success",
                "ui_type": "gui_odyssey"
            }),
            "expected_format": 1.0,
            "expected_accuracy": 1.0
        },
        {
            "name": "agentnetbench - left_click action (correct)",
            "predict": 'Thought: Click on the element.\nAction: Perform left click.\n<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [640, 480]}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "left_click",
                "gt_bbox": [600, 450, 680, 510],
                "input_text": "",
                "ui_type": "agentnetbench"
            }),
            "expected_format": 1.0,
            "expected_accuracy": 1.0
        },
        {
            "name": "agentnetbench - key action (correct)",
            "predict": 'Thought: Press Enter key.\nAction: Press Enter.\n<tool_call>\n{"name": "computer_use", "arguments": {"action": "key", "keys": ["Return"]}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "key",
                "gt_bbox": [],
                "input_text": "Return",
                "ui_type": "agentnetbench"
            }),
            "expected_format": 1.0,
            "expected_accuracy": 1.0
        },
        {
            "name": "Format error - missing Thought",
            "predict": 'Action: Click the button.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 300]}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "click",
                "gt_bbox": [400, 200, 600, 400],
                "input_text": "",
                "ui_type": "android_control"
            }),
            "expected_format": 0.0,
            "expected_accuracy": None  # Won't be calculated if format is wrong
        },
        {
            "name": "Format error - missing Action",
            "predict": 'Thought: I need to click.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 300]}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "click",
                "gt_bbox": [400, 200, 600, 400],
                "input_text": "",
                "ui_type": "android_control"
            }),
            "expected_format": 0.0,
            "expected_accuracy": None
        },
        {
            "name": "Format error - invalid JSON",
            "predict": 'Thought: Click button.\nAction: Perform click.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500 300]}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "click",
                "gt_bbox": [400, 200, 600, 400],
                "input_text": "",
                "ui_type": "android_control"
            }),
            "expected_format": 0.0,
            "expected_accuracy": None
        },
        {
            "name": "Format error - missing required parameter (coordinate)",
            "predict": 'Thought: Click button.\nAction: Perform click.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click"}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "click",
                "gt_bbox": [400, 200, 600, 400],
                "input_text": "",
                "ui_type": "android_control"
            }),
            "expected_format": 0.0,
            "expected_accuracy": None
        },
        {
            "name": "Format error - invalid action for ui_type",
            "predict": 'Thought: Use keyboard.\nAction: Press key.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "key", "keys": ["Return"]}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "key",
                "gt_bbox": [],
                "input_text": "Return",
                "ui_type": "android_control"  # android_control doesn't support 'key' action
            }),
            "expected_format": 0.0,
            "expected_accuracy": None
        },
    ]

    # Run tests
    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print("-" * 80)

        format_score = r1gui_format_reward(test['predict'], test['ground_truth'])
        accuracy_score = r1gui_accuracy_reward(test['predict'], test['ground_truth'])

        print(f"Predicted: {test['predict'][:100]}...")
        print(f"Format Score: {format_score:.2f} (Expected: {test['expected_format']:.2f})")
        print(f"Accuracy Score: {accuracy_score:.2f}", end="")
        if test['expected_accuracy'] is not None:
            print(f" (Expected: {test['expected_accuracy']:.2f})")
        else:
            print(" (N/A - format check failed)")

        # Check if test passed
        format_pass = abs(format_score - test['expected_format']) < 0.01
        if test['expected_accuracy'] is not None:
            accuracy_pass = abs(accuracy_score - test['expected_accuracy']) < 0.01
        else:
            accuracy_pass = True  # Skip accuracy check if expected is None

        if format_pass and accuracy_pass:
            print("✓ PASSED")
            passed += 1
        else:
            print("✗ FAILED")
            if not format_pass:
                print(f"  Format mismatch: got {format_score:.2f}, expected {test['expected_format']:.2f}")
            if not accuracy_pass and test['expected_accuracy'] is not None:
                print(f"  Accuracy mismatch: got {accuracy_score:.2f}, expected {test['expected_accuracy']:.2f}")
            failed += 1

    # Test batch compute_score function
    print("\n" + "=" * 80)
    print("Testing Batch compute_score Function")
    print("=" * 80)

    batch_input = [
        {
            "response": 'Thought: Click the button.\nAction: Perform click.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 300]}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "click",
                "gt_bbox": [400, 200, 600, 400],
                "input_text": "",
                "ui_type": "android_control"
            })
        },
        {
            "response": 'Thought: Type text.\nAction: Input text.\n<tool_call>\n{"name": "mobile_use", "arguments": {"action": "type", "text": "test"}}\n</tool_call>',
            "ground_truth": json.dumps({
                "action": "type",
                "gt_bbox": [],
                "input_text": "test",
                "ui_type": "android_control"
            })
        }
    ]

    batch_scores = compute_score(batch_input, format_weight=0.5)
    print(f"\nBatch scores computed for {len(batch_scores)} samples:")
    for i, score in enumerate(batch_scores, 1):
        print(f"Sample {i}: Overall={score['overall']:.2f}, Format={score['format']:.2f}, "
              f"Accuracy={score['accuracy']:.2f}, Pass@Accuracy={score.get('pass_at_accuracy', 0):.2f}")

    # Summary
    print("\n" + "=" * 80)
    print(f"Test Summary: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 80)
