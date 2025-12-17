#!/usr/bin/env python3
import os
import json
import argparse
import re
import importlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from eval import ActionEvaluator

def load_trajectory_results(file_path: Path) -> list:
    """Load trajectory results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_trajectory_results(file_path: Path, results: list):
    """Save re-evaluated trajectory results to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

def calculate_metrics(output_dir: Path) -> dict:
    """Calculate evaluation metrics from all trajectory files."""
    total_steps = 0
    total_score = 0.0
    action_type_scores = defaultdict(list)
    class_scores = defaultdict(list)
    class_dict = {
        'click': 'coord',
        'doubleclick': 'coord',
        'moveto': 'coord',
        'dragto': 'coord',
        'rightclick': 'coord',
        'scroll': 'coord',
        'write': 'content',
        'hotkey': 'content',
        'press': 'content',
        'terminate': 'function'
    }
    alt_matched_count = 0
    milestone_scores = []
    think_cnt = 0
    # Process each result file
    for result_file in output_dir.glob("*.json"):
        if result_file.name in ["metric.json", "hyperparams.json"]:
            continue
            
        with open(result_file, "r") as f:
            results = json.load(f)
            
        for result in results:
            total_steps += 1
            if "evaluation" not in result:
                continue
                
            total_score += result["evaluation"]["total"]
            
            if result.get("raw_response", ""):
                if "</thinking>" in result["raw_response"] or "<thinking>" in result["raw_response"]:
                    think_cnt += 1

            if result.get("alternative_matched", False):
                alt_matched_count += 1
            
            # Collect scores by action type
            for action_type, score in result["evaluation"]["actions"].items():
                action_type_scores[action_type].append(score)
                class_scores[class_dict.get(action_type, 'other')].append(score)
            
            # Track milestone scores separately
            # if result.get("milestone", False):
            #     milestone_scores.append(result["evaluation"]["total"])
    
    # Calculate metrics
    metrics = {
        "total_trajectories": len(list(output_dir.glob("*.json"))),  # Exclude metric.json and hyperparams.json
        "total_steps": total_steps,
        "average_score": total_score / total_steps * 100 if total_steps > 0 else 0.0,
        "alternative_matches": alt_matched_count,
        "alternative_match_percentage": (alt_matched_count / total_steps * 100) if total_steps > 0 else 0.0,
        "think_ratio": (think_cnt / total_steps * 100) if total_steps > 0 else 0.0,
        "action_type_scores": {
            action_type: sum(scores) / len(scores) * 100
            for action_type, scores in action_type_scores.items()
        },
        "class_scores": {
            cls: sum(scores) / len(scores) * 100
            for cls, scores in class_scores.items()
        },
        # "milestone_score": sum(milestone_scores) / len(milestone_scores) if milestone_scores else 0.0,
        "timestamp": datetime.now().isoformat()
    }
    
    return metrics

def get_agent_from_dir_name(dir_name: str) -> str:
    """Extract supported agent name (qwen25vl, qwen3vl, aguvis, or opencua) from directory name."""
    lower_name = dir_name.lower()
    if "qwen25vl" in lower_name or "qwen2.5-vl" in lower_name or "qwen2.5vl" in lower_name or "qwen-vl" in lower_name:
        return "qwen25vl"
    if "qwen3vl" in lower_name or "qwen3-vl" in lower_name:
        return "qwen3vl"
    if "aguvis" in lower_name:
        return "aguvis"
    if "opencua" in lower_name:
        return "opencua"
    return None

def reeval_directory(traj_list: list, output_dir: Path = None, agent_type: str = None) -> dict:
    """Re-evaluate all trajectories in the input directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
     
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ActionEvaluator()
    
    # Detect agent type from directory name
    agent_name = agent_type
    agent_module = None
    agent_instance = None
    
    print(f"Detected agent type: {agent_name}")
    
    if agent_name:
        try:
            # Support qwen25vl, aguvis, and opencua
            module_name = f"agent.{agent_name}"
            agent_module = importlib.import_module(module_name)

            # Find the agent class in the module
            for attr_name in dir(agent_module):
                attr = getattr(agent_module, attr_name)
                if isinstance(attr, type) and attr_name.lower() == agent_name.lower():
                    if agent_name.lower() == "qwen25vl":
                        # Create a specialized dummy parsing agent for qwen25vl
                        try:
                            from utils.qwen_vl_utils import smart_resize  # noqa: F401
                        except ImportError:
                            def smart_resize(height, width, max_dimension=1024):
                                aspect_ratio = width / height
                                if width >= height:
                                    new_width = min(width, max_dimension)
                                    new_height = int(new_width / aspect_ratio)
                                else:
                                    new_height = min(height, max_dimension)
                                    new_width = int(new_height * aspect_ratio)
                                return new_height, new_width
                            print("Using fallback smart_resize implementation")

                        class DummyParsingAgent(attr):
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
                                import os
                                image_path = os.path.join(image_dir, image_file)
                                with open(image_path, "rb") as f:
                                    return f.read()

                        agent_instance = DummyParsingAgent(model="dummy")
                        print(f"Successfully loaded specialized parsing agent for {attr_name}")
                    elif agent_name.lower() == "qwen3vl":
                        # Create a specialized dummy parsing agent for qwen3vl
                        try:
                            from utils.qwen_vl_utils import smart_resize  # noqa: F401
                        except ImportError:
                            def smart_resize(height, width, max_dimension=1024):
                                aspect_ratio = width / height
                                if width >= height:
                                    new_width = min(width, max_dimension)
                                    new_height = int(new_width / aspect_ratio)
                                else:
                                    new_height = min(height, max_dimension)
                                    new_width = int(new_height * aspect_ratio)
                                return new_height, new_width
                            print("Using fallback smart_resize implementation")

                        class DummyParsingAgent(attr):
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
                                import os
                                image_path = os.path.join(image_dir, image_file)
                                with open(image_path, "rb") as f:
                                    return f.read()

                        agent_instance = DummyParsingAgent(model="dummy")
                        print(f"Successfully loaded specialized parsing agent for {attr_name}")
                        break

                    elif agent_name.lower() == "opencua":
                        # Create a specialized dummy parsing agent for opencua (avoid BaseAgent init)
                        class DummyParsingAgent(attr):
                            def __init__(self, model, client=None, **kwargs):
                                self.model = model
                                self.client = None
                                self.image_dir = "/data1/zlx/workspace/EasyR1/data/data/AgentNetBench/test_data/images"
                            def load_image(self, image_file, image_dir):
                                import os
                                image_path = os.path.join(image_dir, image_file)
                                with open(image_path, "rb") as f:
                                    return f.read()

                        agent_instance = DummyParsingAgent(model="dummy")
                        print(f"Successfully loaded specialized parsing agent for {attr_name}")
                    else:
                        agent_instance = attr(model="dummy", client=None)
                        print(f"Successfully loaded agent class {attr_name}")
                    break

            if not agent_instance:
                print(f"Warning: Could not find agent class in module {module_name}")
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not import supported agent module: {e}")
    
    # Process each trajectory file
    # for file_path in tqdm(list(input_dir.glob("*.json")), desc="Re-evaluating trajectories"):
    for results in tqdm(traj_list, desc="Re-evaluating trajectories"):
        # Skip metric and hyperparams files
        # if file_path.name in ["metric.json", "hyperparams.json"]:
        #     continue
        
        # # Load trajectory results
        # results = load_trajectory_results(file_path)
        # # print(file_path)
        trajectory_path = f'/data1/zlx/workspace/EasyR1/data/data/AgentNetBench/test_data/{results[0]["task_id"]}.json'
        with open(trajectory_path, "r") as f:
            trajectory = json.load(f)
        
        # Re-evaluate each step
        for step_result in results:
            # Check if raw_response exists
            if "raw_response" not in step_result:
                print(f"Warning: No raw_response in step {step_result.get('step_num', '?')}")
                continue
                
            raw_response = step_result["raw_response"]
            
            # Re-parse the raw response using the appropriate agent
            if agent_instance and hasattr(agent_instance, 'parse_response') and hasattr(agent_instance, 'extract_actions'):
                try:
                    parsed_action = agent_instance.parse_response(raw_response, trajectory, step_result["step_num"])
                    step_result["parsed_action"] = parsed_action
                    
                    try:
                        predicted_actions = agent_instance.extract_actions(parsed_action)
                        step_result["predicted_actions"] = predicted_actions
                    except Exception as e:
                        print(f"Error extracting actions for step {step_result.get('step_num', '?')}: {e}")
                        # Keep the existing predicted_actions if any
                        if "predicted_actions" not in step_result:
                            step_result["predicted_actions"] = []
                except Exception as e:
                    print(f"Error parsing response for step {step_result.get('step_num', '?')}: {e}")
                    # Keep the existing parsed_action and predicted_actions if any
                    if "parsed_action" not in step_result:
                        step_result["parsed_action"] = None
                    if "predicted_actions" not in step_result:
                        step_result["predicted_actions"] = []
            
            # Prepare evaluation item
            # eval_item = {
            #     "ground_truth_actions": step_result["ground_truth_actions"],
            #     # "ground_truth_actions": step_result["used_actions"],
            #     "predicted_actions": step_result["predicted_actions"]
            # }
            
            # Re-evaluate the predicted actions
            # eval_scores = evaluator.evaluate_action(step_result)
            # step_result["evaluation"] = eval_scores
            step_data = trajectory['steps'][step_result['step_num']]
            if step_result.get("predicted_actions"):
                gt_eval_item = {
                    "ground_truth_actions": step_data.get("ground_truth_actions", []),
                    "predicted_actions": step_result["predicted_actions"]
                }
                gt_eval_scores = evaluator.evaluate_action(gt_eval_item)
                best_eval_scores = gt_eval_scores
                used_actions = step_data.get("ground_truth_actions", [])
                
                # Check alternative options
                if gt_eval_scores["total"] < 1.0 and "alternative_options" in step_data:
                    for alt_option in step_data.get("alternative_options", []):
                        alt_eval_item = {
                            "ground_truth_actions": alt_option,
                            "predicted_actions": step_result["predicted_actions"]
                        }
                        alt_eval_scores = evaluator.evaluate_action(alt_eval_item)
                        if alt_eval_scores["total"] > best_eval_scores["total"]:
                            best_eval_scores = alt_eval_scores
                            used_actions = alt_option
                step_result.update({
                    "used_actions": used_actions,
                    "alternative_matched": used_actions != step_data.get("ground_truth_actions", []),
                    "evaluation": best_eval_scores
                })
            
        # Save re-evaluated results
        output_file = os.path.join(output_dir, f"{results[0]['task_id']}.json")
        save_trajectory_results(output_file, results)
    
    # Calculate and save new metrics
    metrics = calculate_metrics(output_dir)
    metric_file = os.path.join(output_dir, "metric.json")
    with open(metric_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Copy hyperparams.json if it exists
    # hyperparams_file = input_dir / "hyperparams.json"
    # if hyperparams_file.exists():
    #     with open(hyperparams_file, "r") as f:
    #         hyperparams = json.load(f)
    #     output_hyperparams = output_dir / "hyperparams.json"
    #     with open(output_hyperparams, "w") as f:
    #         json.dump(hyperparams, f, indent=4)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Re-evaluate agent outputs from a directory")
    # parser.add_argument("--input_dir", type=str, help="Directory containing trajectory evaluation results")
    parser.add_argument("--data_dir", type=str, help="Test trajectory data directory")
    parser.add_argument("--response_file", type=str, help="Input trajectory evaluation results file")
    parser.add_argument("--agent_type", type=str, default=None, help="Agent type (qwen25vl, qwen3vl, aguvis, opencua) if auto-detection fails")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for re-evaluated results")
    
    args = parser.parse_args()

    with open(args.response_file, "r") as f:
        responses = [json.loads(line) for line in f]
    
    id2response = {item["uid"]: item['pred'] for item in responses}
    traj_list= []
    for file in os.listdir(args.data_dir):
        if file.endswith(".json"):
            file_path = os.path.join(args.data_dir, file)
            with open(file_path, "r") as f:
                trajectory = json.load(f)
            task_id = trajectory["task_id"]
            step_list = []
            for step_n in range(len(trajectory['steps'])):
                step_id = f"{task_id}-{step_n}"
                step_dict = {
                    'task_id': task_id,
                    'step_num': step_n,
                    'ground_truth_actions': trajectory['steps'][step_n]['ground_truth_actions'],
                    'milestone': trajectory['steps'][step_n].get('milestone', False),
                    'raw_response': id2response[step_id],
                }
                step_list.append(step_dict)
            traj_list.append(step_list)


    output_dir = Path(args.output_dir) if args.output_dir else None
    
    print(f"\n=== Starting Re-evaluation ===")
    print(f"Raw response file: {args.response_file}")
    print(f"Output directory: {output_dir if output_dir else 'auto-generated'}")

    metrics = reeval_directory(traj_list, output_dir, args.agent_type)

    print("\n=== Re-evaluation Complete ===")
    print(f"Total trajectories: {metrics['total_trajectories']}")
    print(f"Total steps: {metrics['total_steps']}")
    print(f"\nMetrics Summary:")
    print(f"Average Score: {metrics['average_score']:.3f}")
    # print(f"Milestone Score: {metrics['milestone_score']:.3f}")
    print("\nAction Type Scores:")
    for action_type, score in metrics['action_type_scores'].items():
        print(f"  {action_type}: {score:.3f}")

if __name__ == "__main__":
    main() 