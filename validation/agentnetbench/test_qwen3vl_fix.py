#!/usr/bin/env python3
import sys
import json
import os
sys.path.insert(0, '.')

from agent.qwen3vl import Qwen3VL
from utils.qwen_vl_utils import smart_resize

# Create DummyParsingAgent
class DummyParsingAgent(Qwen3VL):
    def __init__(self, model, client=None, **kwargs):
        self.model = model
        self.client = None
        self.image_dir = "/data1/zlx/workspace/EasyR1/data/data/AgentNetBench/test_data/images"
        self.image_cache = {}
        self.message_cache = {}
        self.history_n = 3
        self.history_responses = []
        self.history_images = []

    def load_image(self, image_file, image_dir):
        image_path = os.path.join(image_dir, image_file)
        print(f"Loading image: {image_path}")
        if not os.path.exists(image_path):
            print(f"  WARNING: Image file does not exist!")
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(image_path, "rb") as f:
            data = f.read()
        print(f"  Loaded {len(data)} bytes")
        return data

print("=" * 70)
print("Test 1: Testing coordinate conversion")
print("=" * 70)

agent = DummyParsingAgent(model="dummy")

# Load real trajectory data
task_file = "/data1/zlx/workspace/EasyR1/data/data/AgentNetBench/test_data/s_cf6275f33375d024.json"
with open(task_file, "r") as f:
    trajectory = json.load(f)

step_idx = 0
step = trajectory['steps'][step_idx]
print(f"\nStep {step_idx} image: {step['image']}")

# Real response from the evaluation
test_response = """Thought: To achieve the goal, I need to first navigate back to the main United Airlines homepage where I can find options for booking flights. The current page is about travel add-ons, which is not relevant to searching for flights.
Action: Click on the 'Book' option in the top navigation bar to access flight booking options.
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [124, 178]}}
</tool_call>"""

print("\nParsing response...")
try:
    parsed = agent.parse_response(test_response, trajectory, step_idx)
    print(f"Parsed action: {parsed}")

    if parsed:
        print("\nExtracting actions...")
        actions = agent.extract_actions(parsed)
        print(f"Extracted actions: {actions}")

        if actions and actions[0][0] == 'click':
            pred_coord = actions[0][1]
            print(f"\nPredicted coordinates: {pred_coord}")

            # Get ground truth
            gt_action = trajectory['steps'][step_idx]['ground_truth_actions'][0]
            gt_coord = (gt_action['params']['position']['x'], gt_action['params']['position']['y'])
            gt_bbox = gt_action['metadata']['bboxes'][0]['rel_bbox']

            print(f"Ground truth coords: {gt_coord}")
            print(f"Ground truth bbox: {gt_bbox}")

            # Check if relative
            if 0 <= pred_coord[0] <= 1 and 0 <= pred_coord[1] <= 1:
                print("\n✓ SUCCESS: Coordinates are RELATIVE (0-1 range)")

                # Check if in bbox
                x, y = pred_coord
                bbox_x, bbox_y, bbox_w, bbox_h = gt_bbox
                if (bbox_x <= x <= bbox_x + bbox_w) and (bbox_y <= y <= bbox_y + bbox_h):
                    print("✓ SUCCESS: Coordinates are INSIDE the bounding box!")
                    print("   This click should score 1.0 in evaluation")
                else:
                    print("✗ FAIL: Coordinates are OUTSIDE the bounding box")
                    print(f"  Pred: ({x:.4f}, {y:.4f})")
                    print(f"  BBox: x=[{bbox_x:.4f}, {bbox_x+bbox_w:.4f}], y=[{bbox_y:.4f}, {bbox_y+bbox_h:.4f}]")
            else:
                print(f"\n✗ FAIL: Coordinates are NOT relative: {pred_coord}")
    else:
        print("✗ FAIL: parse_response returned None")
except Exception as e:
    print(f"✗ Exception occurred: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test 2: Testing double brace handling")
print("=" * 70)

test_response2 = """Thought: Testing
<tool_call>
{{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}
</tool_call>"""

print("\nParsing response with double braces...")
try:
    parsed2 = agent.parse_response(test_response2, trajectory, 0)
    print(f"Parsed action: {parsed2}")

    if parsed2:
        actions2 = agent.extract_actions(parsed2)
        print(f"Extracted actions: {actions2}")
        print("✓ SUCCESS: Double braces were properly cleaned")
    else:
        print("✗ FAIL: Could not parse double braces")
except Exception as e:
    print(f"✗ Exception: {e}")

print("\nDone!")
