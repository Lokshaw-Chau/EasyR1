#!/usr/bin/env python3
import os
import base64
import re
from typing import Dict, List, Tuple, Any, Optional

from .base_agent import BaseAgent

# Constants merged from constant_l2_action_history.py
SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."""

INSTRUCTION_PROMPT = """
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {high_level_instruction}

Previous actions:
{previous_im}"""

INNERMOLOGUE = """Thought: {thought}
Action: {low_level_instruction}"""

HISTORY_INNERMOLOGUE = """{low_level_instruction}"""

# Code merged from model_l2_action_history.py
MAX_RETRIES = 3

class Aguvis(BaseAgent):
    """AguVis agent implementation."""
    
    def __init__(self, model: str, client: Any):
        """Initialize the AguVis agent.
        
        Args:
            model: The model identifier to use for predictions
            client: The client object for making API calls
        """
        super().__init__(model, client)
        self.system_prompt = SYSTEM_PROMPT

    def prompt(self, trajectory: Dict[str, Any], current_step: int) -> List[Dict[str, Any]]:
        """Generate prompt for the agent.
        
        Args:
            trajectory: The full trajectory data
            current_step: The current step index being processed
            
        Returns:
            List of message dictionaries for the model
        """
        # Get current step data
        step = trajectory['steps'][current_step]
        
        # Build previous actions string
        previous_actions = []
        for i in range(current_step):
            prev_step = trajectory['steps'][i]
            if 'inner_monologue' in prev_step:
                previous_actions.append(f"Step {i+1}: {prev_step['inner_monologue'].get('low_level_instruction', '')}")
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        # Create instruction prompt
#         instruction_prompt = f"""
# Please generate the next move according to the UI screenshot, instruction and previous actions.

# Instruction: {trajectory['user_task_description']}

# Previous actions:
# {previous_actions_str}"""

        instruction_prompt = f"""
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {trajectory['high_level_task_description']}

Previous actions:
{previous_actions_str}"""
        
        append_text = """<|recipient|>all\n
Thought: """
        # Load and encode image
        image = self.load_image(step['image'], self.image_dir)
        image_b64 = base64.b64encode(image).decode('utf-8')

        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": instruction_prompt}
            ]},
            {"role": "assistant", "content": append_text}
        ]
        
        return messages

    def parse_response(self, response: str, trajectory: Optional[Dict[str, Any]] = None, step_idx: Optional[int] = None) -> str:
        """Parse the raw response from the agent into executable form.
        
        Args:
            response: Raw response string from the model
            trajectory: Optional trajectory data (unused)
            step_idx: Optional step index (unused)
            
        Returns:
            Parsed response string ready for evaluation
        """
        if response is None:
            return None
            
        # Split response into lines
        lines = response.strip().split("\n")
        lines = [line for line in lines if line.strip()]
        
        # Find the action line
        action_line = None
        for line in lines:
            if line.strip().startswith("pyautogui") or line.strip().startswith("computer."):
                action_line = line.strip()
                break
        
        return action_line

    def extract_actions(self, action: str) -> List[Tuple[str, Any]]:
        """Extract individual actions from parsed response.
        
        Args:
            action: Parsed action string
            
        Returns:
            List of tuples containing (action_type, action_params)
        """
        if not action:
            return []
            
        actions = []
        
        # Handle computer.terminate
        if action.startswith("computer.terminate"):
            status_match = re.search(r"status=['\"](\w+)['\"]", action)
            if status_match:
                return [("terminate", status_match.group(1))]
            return []
            
        # Handle pyautogui actions
        if action.startswith("pyautogui."):
            # Extract coordinates for click/moveTo actions (tolerate spaces around '=')
            coord_match = re.search(r"x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)", action)
            if coord_match:
                x, y = map(float, coord_match.groups())
                if "click" in action:
                    actions.append(("click", (x, y)))
                elif "moveTo" in action:
                    actions.append(("moveTo", (x, y)))
                elif "doubleClick" in action:
                    actions.append(("doubleClick", (x, y)))
                    
            # Handle scroll action with page parameter
            scroll_page_match = re.search(r"scroll\(page=([-\d.]+)\)", action)
            if scroll_page_match:
                # Use default coordinates if not specified
                default_coords = (0.5, 0.5)
                actions.append(("moveTo", default_coords))
                
                # Convert page value to direction
                page_value = float(scroll_page_match.group(1))
                direction = "up" if page_value > 0 else "down"
                actions.append(("scroll", direction))
                    
            # Extract text for write actions
            write_match = re.search(r"message=['\"](.+?)['\"]", action)
            if write_match:
                text = write_match.group(1)
                actions.append(("write", text))
                
            # Extract keys for press/hotkey actions
            keys_match = re.findall(r"keys=\[(.*?)\]", action)
            if keys_match:
                keys = [k.strip("'\" ") for k in keys_match[0].split(",") if k.strip()]
                # Always return lists for consistency with evaluator
                if len(keys) <= 1:
                    actions.append(("press", keys if keys else []))
                else:
                    actions.append(("hotkey", keys))
                
        return actions

    def load_image(self, image_file, image_dir):
        image_path = os.path.join(image_dir, image_file)
        with open(image_path, "rb") as f:
            image = f.read()
        return image
