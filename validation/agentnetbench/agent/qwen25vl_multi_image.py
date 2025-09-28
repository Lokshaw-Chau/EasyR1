#!/usr/bin/env python3
import os
import base64
import re
import json
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from utils.qwen_vl_utils import smart_resize
import logging
from functools import lru_cache

from .base_agent import BaseAgent

# Set up logging
logger = logging.getLogger("qwen25vl")

class Qwen25VL(BaseAgent):
    """Qwen2.5-VL implementation with enhanced prompting, action parsing, and multi-image support."""
    
    def __init__(self, model: str, client: Any):
        """Initialize the Qwen2.5-VL agent.
        
        Args:
            model: The model identifier to use for predictions
            client: The client object for making API calls
        """
        super().__init__(model, client)
        self.system_prompt = """You are a helpful assistant

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn\'t open, try wait and taking another screenshot.\\n* The screen\'s resolution is 168x224.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n* `type`: Type a string of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button.\\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `right_click`: Click the right mouse button.\\n* `middle_click`: Click the middle mouse button.\\n* `double_click`: Double-click the left mouse button.\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}, "keys": {"description": "Required only by `action=key`.", "type": "array"}, "text": {"description": "Required only by `action=type`.", "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}, "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
        self.history_n = 5  # Maximum number of images in history
        self.history_responses = []
        self.history_images = []
        self.image_cache = {}  # Cache for processed images
        self.message_cache = {}  # Cache for generated messages

    @lru_cache(maxsize=100)
    def _process_image(self, image_path: str) -> str:
        """Process an image and return its base64 encoding.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        # Check if image is already in cache
        if image_path in self.image_cache:
            return self.image_cache[image_path]
            
        # Load and process the image
        image_bytes = self.load_image(image_path, self.image_dir)
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        
        # Calculate resized dimensions
        resized_height, resized_width = smart_resize(height=height, width=width)
        
        image = image.resize((resized_width, resized_height))
        
        # Convert back to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        processed_bytes = buffer.getvalue()
        
        # Encode to base64
        image_b64 = base64.b64encode(processed_bytes).decode('utf-8')
        
        # Cache the result
        self.image_cache[image_path] = image_b64
        
        return image_b64

    def prompt(self, trajectory: Dict[str, Any], current_step: int) -> List[Dict[str, Any]]:
        """Generate prompt for the agent with multi-image support.
        
        Args:
            trajectory: The full trajectory data
            current_step: The current step index being processed
            
        Returns:
            List of message dictionaries for the model
        """
        # Check if we have this prompt cached
        cache_key = f"{trajectory.get('task_id', '')}_{current_step}"
        if cache_key in self.message_cache:
            return self.message_cache[cache_key]
            
        # Initialize messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            }
        ]
        
        # Get current step data
        step = trajectory['steps'][current_step]
        
        # Build previous actions string - only include actions not in the history_n window
        previous_actions = []
        history_start_idx = max(0, current_step + 1 - self.history_n)
        
        for i in range(history_start_idx):
            prev_step = trajectory['steps'][i]
            if 'inner_monologue' in prev_step:
                previous_actions.append(f"Step {i+1}: {prev_step['inner_monologue'].get('low_level_instruction', '')}")
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        # Create instruction prompt
        instruction_prompt = f"""
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {trajectory['high_level_task_description']}

Previous actions:
{previous_actions_str}"""

        # Reset history for each new conversation
        self.history_responses = []
        self.history_images = []

        # Process only steps in the history window
        for i in range(history_start_idx, current_step):
            prev_step = trajectory['steps'][i]
            
            # Get image for this step
            image_file = prev_step.get('image', '')
            if image_file and self.image_dir:
                image_bytes = self.load_image(image_file, self.image_dir)
                self.history_images.append(image_bytes)

            # Format response for this step
            if 'ground_truth_actions' in prev_step:
                # Get the response from the model for this previous step
                action_str = self._format_action_response(prev_step.get('ground_truth_actions', []), prev_step)
                if action_str:
                    self.history_responses.append(action_str)
        
        # Keep only the last history_n responses and images (should already be limited, but just in case)
        if len(self.history_responses) > self.history_n:
            self.history_responses = self.history_responses[-self.history_n:]
        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]

        # Add history responses and images to messages in dialogue format
        image_num = 0
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                # Add image if within history_n window
                if history_idx + self.history_n > len(self.history_responses):
                    # Process the image before sending it to the model
                    image_bytes = self.history_images[image_num]
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(image_bytes))
                    width, height = image.size
                    
                    # Calculate resized dimensions
                    resized_height, resized_width = smart_resize(height=height, width=width)
                    
                    # Create a new image with padding if needed
                    if resized_width > width or resized_height > height:
                        # Create a new image with white background
                        new_image = Image.new('RGB', (resized_width, resized_height), (255, 255, 255))
                        # Calculate padding to center the original image
                        x_padding = (resized_width - width) // 2
                        y_padding = (resized_height - height) // 2
                        # Paste the original image in the center
                        new_image.paste(image, (x_padding, y_padding))
                        image = new_image
                    else:
                        # If resized dimensions are smaller, resize the image
                        image = image.resize((resized_width, resized_height))
                    
                    # Convert back to bytes
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    processed_bytes = buffer.getvalue()
                    
                    image_b64 = base64.b64encode(processed_bytes).decode('utf-8')
                    
                    # If this is the first image, combine it with the instruction prompt
                    if history_idx == 0:
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                                {"type": "text", "text": instruction_prompt}
                            ]
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]
                        })
                    image_num += 1
                
                # Get the step index for this history item
                step_idx = history_start_idx + history_idx
                step = trajectory['steps'][step_idx]
                
                # Get the low-level instruction
                low_level_instruction = ""
                if 'inner_monologue' in step:
                    low_level_instruction = f"Step {step_idx + 1}: {step['inner_monologue'].get('low_level_instruction', '')}"
                
                # Add history response with both instruction and tool call
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": low_level_instruction},
                        {"type": "text", "text": "\n"},
                        {"type": "text", "text": history_response}
                    ]
                })

        # Add current step image
        if len(self.history_responses) > 0:
            # Add current step image without instruction prompt since we already have history
            step = trajectory['steps'][current_step]
            image_file = step.get('image', '')
            if image_file and self.image_dir:
                # Use the cached image processing function
                image_b64 = self._process_image(image_file)
                messages.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]
                })
        else:
            # If there's no history, add current step image with instruction prompt
            step = trajectory['steps'][current_step]
            image_file = step.get('image', '')
            if image_file and self.image_dir:
                # Use the cached image processing function
                image_b64 = self._process_image(image_file)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        {"type": "text", "text": instruction_prompt}
                    ]
                })

        append_text = f"""Step {current_step + 1}: Thought:"""
        messages.append({"role": "assistant", "content": [{"type": "text", "text": append_text}]})

        # Cache the messages
        self.message_cache[cache_key] = messages

        return messages

    def _format_action_response(self, actions: List[Dict[str, Any]], step: Dict[str, Any]) -> str:
        """Format previous actions into a response string for the history.
        
        Args:
            actions: List of action dictionaries
            step: Step data containing metadata
            
        Returns:
            Formatted action response string with proper XML tags
        """
        if not actions:
            return None
            
        # Initialize the action object
        action_obj = {
            "name": "computer_use",
            "arguments": {}
        }
        
        # Check for composite actions
        if len(actions) >= 2:
            # moveTo + scroll composite action
            if actions[0].get('type') == 'moveTo' and actions[1].get('type') == 'scroll':
                action_obj["arguments"]["action"] = "scroll"
                
                # Get coordinate from moveTo action
                move_action = actions[0]
                if 'metadata' in move_action and 'bboxes' in move_action['metadata'] and move_action['metadata']['bboxes']:
                    bbox = move_action['metadata']['bboxes'][0].get('rel_bbox', [])
                    if bbox:
                        # Calculate center point of bounding box
                        if len(bbox) == 4:  # [x, y, width, height]
                            x = bbox[0] + (bbox[2] / 2)
                            y = bbox[1] + (bbox[3] / 2)
                        else:  # [x, y]
                            x, y = bbox[0], bbox[1]
                            
                        # Get image dimensions
                        image_bytes = self.load_image(step['image'], self.image_dir) if step.get('image') else None
                        if image_bytes:
                            from PIL import Image
                            import io
                            image = Image.open(io.BytesIO(image_bytes))
                            width, height = image.size
                            
                            # Calculate resized dimensions
                            resized_height, resized_width = smart_resize(height=height, width=width)
                            
                            # Convert relative coordinates to resized coordinates
                            action_obj["arguments"]["coordinate"] = [
                                int(x * resized_width),
                                int(y * resized_height)
                            ]
                
                # Get scroll direction and amount from scroll action
                scroll_action = actions[1]
                scroll_params = scroll_action.get('params', {})
                direction = scroll_params.get('direction', 'down')
                amount = scroll_params.get('amount', -3)  # Default to -3 (down) if not specified
                
                # Convert amount to pixels
                pixels = amount * 100 if direction != 'down' else amount * -100
                action_obj["arguments"]["pixels"] = pixels
                
                # Format and return
                json_str = json.dumps(action_obj)
                return f"<tool_call>\n{json_str}\n</tool_call>"
                
            # moveTo + dragTo composite action
            elif actions[0].get('type') == 'moveTo' and actions[1].get('type') == 'dragTo':
                action_obj["arguments"]["action"] = "left_click_drag"
                
                # Get start coordinate from moveTo action
                move_action = actions[0]
                if 'metadata' in move_action and 'bboxes' in move_action['metadata'] and move_action['metadata']['bboxes']:
                    start_bbox = move_action['metadata']['bboxes'][0].get('rel_bbox', [])
                    
                    # Get end coordinate from dragTo action
                    drag_action = actions[1]
                    if 'metadata' in drag_action and 'bboxes' in drag_action['metadata'] and drag_action['metadata']['bboxes']:
                        end_bbox = drag_action['metadata']['bboxes'][0].get('rel_bbox', [])
                        
                        if start_bbox and end_bbox:
                            # Calculate center points
                            if len(start_bbox) == 4:
                                start_x = start_bbox[0] + (start_bbox[2] / 2)
                                start_y = start_bbox[1] + (start_bbox[3] / 2)
                            else:
                                start_x, start_y = start_bbox[0], start_bbox[1]
                                
                            if len(end_bbox) == 4:
                                end_x = end_bbox[0] + (end_bbox[2] / 2)
                                end_y = end_bbox[1] + (end_bbox[3] / 2)
                            else:
                                end_x, end_y = end_bbox[0], end_bbox[1]
                                
                            # Get image dimensions
                            image_bytes = self.load_image(step['image'], self.image_dir) if step.get('image') else None
                            if image_bytes:
                                from PIL import Image
                                import io
                                image = Image.open(io.BytesIO(image_bytes))
                                width, height = image.size
                                
                                # Calculate resized dimensions
                                resized_height, resized_width = smart_resize(height=height, width=width)
                                
                                # Use the end coordinates for the drag action
                                action_obj["arguments"]["coordinate"] = [
                                    int(end_x * resized_width),
                                    int(end_y * resized_height)
                                ]
                                
                                # Format and return
                                json_str = json.dumps(action_obj)
                                return f"<tool_call>\n{json_str}\n</tool_call>"
        
        # If not a composite action, handle single action
        action = actions[0]
        action_type = action.get('type', '')
        
        # Handle different action types
        if action_type == 'click':
            action_obj["arguments"]["action"] = "left_click"
            if 'metadata' in action and 'bboxes' in action['metadata'] and action['metadata']['bboxes']:
                bbox = action['metadata']['bboxes'][0].get('rel_bbox', [])
                if bbox:
                    # Calculate center point of bounding box
                    if len(bbox) == 4:  # [x, y, width, height]
                        x = bbox[0] + (bbox[2] / 2)
                        y = bbox[1] + (bbox[3] / 2)
                    else:  # [x, y]
                        x, y = bbox[0], bbox[1]
                        
                    # Get image dimensions
                    image_bytes = self.load_image(step['image'], self.image_dir) if step.get('image') else None
                    if image_bytes:
                        from PIL import Image
                        import io
                        image = Image.open(io.BytesIO(image_bytes))
                        width, height = image.size
                        
                        # Calculate resized dimensions
                        resized_height, resized_width = smart_resize(height=height, width=width)
                        
                        # Convert relative coordinates to resized coordinates
                        action_obj["arguments"]["coordinate"] = [
                            int(x * resized_width),
                            int(y * resized_height)
                        ]
        elif action_type == 'doubleClick':
            action_obj["arguments"]["action"] = "double_click"
            # Handle coordinates similar to click
            if 'metadata' in action and 'bboxes' in action['metadata'] and action['metadata']['bboxes']:
                bbox = action['metadata']['bboxes'][0].get('rel_bbox', [])
                if bbox:
                    # Similar calculation as for click
                    if len(bbox) == 4:
                        x = bbox[0] + (bbox[2] / 2)
                        y = bbox[1] + (bbox[3] / 2)
                    else:
                        x, y = bbox[0], bbox[1]
                        
                    image_bytes = self.load_image(step['image'], self.image_dir) if step.get('image') else None
                    if image_bytes:
                        from PIL import Image
                        import io
                        image = Image.open(io.BytesIO(image_bytes))
                        width, height = image.size
                        
                        resized_height, resized_width = smart_resize(height=height, width=width)
                        
                        action_obj["arguments"]["coordinate"] = [
                            int(x * resized_width),
                            int(y * resized_height)
                        ]
        elif action_type == 'type':
            action_obj["arguments"]["action"] = "type"
            action_obj["arguments"]["text"] = action.get('params', {}).get('content', '')
        elif action_type in ['press', 'hotkey']:
            action_obj["arguments"]["action"] = "key"
            keys = action.get('params', {}).get('key', [])
            if isinstance(keys, str):
                keys = [keys]
            action_obj["arguments"]["keys"] = keys
        elif action_type == 'scroll':
            action_obj["arguments"]["action"] = "scroll"
            direction = action.get('params', {}).get('direction', 'down')
            amount = action.get('params', {}).get('amount', -3)
            pixels = amount * 100 if direction != 'down' else amount * -100
            action_obj["arguments"]["pixels"] = pixels
            
        # Format the response with tool_call XML tags
        json_str = json.dumps(action_obj)
        return f"<tool_call>\n{json_str}\n</tool_call>"

    def parse_response(self, response: str, trajectory: Optional[Dict[str, Any]] = None, step_idx: Optional[int] = None) -> str:
        """Parse the raw response from the agent into executable form.
        
        Args:
            response: Raw response string from the model
            trajectory: Optional trajectory data for coordinate conversion
            step_idx: Optional step index for coordinate conversion
            
        Returns:
            Parsed response string ready for evaluation
        """
        if response is None:
            return None
            
        # Find all tool calls in the response
        tool_calls = []
        
        def process_tool_call(json_str: str) -> None:
            """Process a single tool call JSON string."""
            try:
                # Parse the JSON
                tool_call = json.loads(json_str)
                if tool_call.get("name") == "computer_use":
                    # Convert computer_use actions to pyautogui commands
                    args = tool_call["arguments"]
                    action = args["action"]
                    
                    # Get image dimensions from the current step
                    if trajectory and step_idx is not None:
                        step = trajectory['steps'][step_idx]
                        image_bytes = self.load_image(step['image'], self.image_dir)
                        # Convert bytes to PIL Image
                        from PIL import Image
                        import io
                        image = Image.open(io.BytesIO(image_bytes))
                        width, height = image.size  # PIL Image.size returns (width, height)
                        
                        # Calculate resized dimensions
                        resized_height, resized_width = smart_resize(
                            height=height,
                            width=width
                        )
                    else:
                        # Default dimensions if no trajectory/step_idx provided
                        logger.warning("No trajectory/step_idx provided")
                        resized_width, resized_height = 1024, 768
                    
                    if action == "left_click":
                        x, y = args["coordinate"]
                        # Convert from resized to relative coordinates
                        rel_x = x / resized_width
                        rel_y = y / resized_height
                        tool_calls.append(f"pyautogui.click(x={rel_x}, y={rel_y})")
                    elif action == "right_click":
                        x, y = args["coordinate"]
                        rel_x = x / resized_width
                        rel_y = y / resized_height
                        tool_calls.append(f"pyautogui.rightClick(x={rel_x}, y={rel_y})")
                    elif action == "double_click":
                        x, y = args["coordinate"]
                        rel_x = x / resized_width
                        rel_y = y / resized_height
                        tool_calls.append(f"pyautogui.doubleClick(x={rel_x}, y={rel_y})")
                    elif action == "type":
                        text = args["text"]
                        tool_calls.append(f"pyautogui.write(message='{text}')")
                    elif action == "key":
                        keys = args["keys"]
                        # Fix possible formatting issues in the keys parameter
                        if isinstance(keys, list):
                            # Clean up any formatting issues in the keys
                            cleaned_keys = []
                            for key in keys:
                                # Check if the key has the "keys=[" prefix or "]" suffix
                                if isinstance(key, str):
                                    # Remove "keys=[" prefix if present
                                    if key.startswith("keys=["):
                                        key = key[6:]
                                    # Remove "]" suffix if present
                                    if key.endswith("]"):
                                        key = key[:-1]
                                    # Handle case where string contains representation of list items like "['ctrl"
                                    if key.startswith("['") or key.startswith("[\""):
                                        # Extract the actual key value from the string representation
                                        key = key[2:] if len(key) > 2 else key
                                    # Handle case where string contains end of list items like "c']"
                                    if key.endswith("']") or key.endswith("\"]"):
                                        # Extract the actual key value from the string representation
                                        key = key[:-2] if len(key) > 2 else key
                                    # Strip any extra whitespace
                                    key = key.strip()
                                    # Add to cleaned keys
                                    cleaned_keys.append(key)
                                else:
                                    cleaned_keys.append(key)
                            keys = cleaned_keys
                            
                        # Use hotkey for multiple keys, press for single key
                        if isinstance(keys, list) and len(keys) > 1:
                            tool_calls.append(f"pyautogui.hotkey(keys={keys})")
                        else:
                            tool_calls.append(f"pyautogui.press(keys={keys})")
                    elif action == "scroll":
                        pixels = args["pixels"]
                        tool_calls.append(f"pyautogui.scroll({pixels})")
                    elif action == "wait":
                        time = args["time"]
                        tool_calls.append(f"pyautogui.sleep({time})")
                    elif action == "terminate":
                        status = args["status"]
                        tool_calls.append(f"computer.terminate(status='{status}')")
                    elif action == "mouse_move":
                        x, y = args["coordinate"]
                        rel_x = x / resized_width
                        rel_y = y / resized_height
                        tool_calls.append(f"pyautogui.moveTo(x={rel_x}, y={rel_y})")
                    elif action == "left_click_drag":
                        x, y = args["coordinate"]
                        rel_x = x / resized_width
                        rel_y = y / resized_height
                        tool_calls.append(f"pyautogui.dragTo(x={rel_x}, y={rel_y})")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing tool call: {e}")
        
        # Extract tool calls from response
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, response, re.DOTALL)
        
        if tool_call_matches:
            for match in tool_call_matches:
                process_tool_call(match)
        else:
            # Fallback: try to find JSON in the response
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Skip lines that start with "Action:" as they are descriptions
                if line.startswith("Action:"):
                    continue
                    
                # try to parse as JSON
                try:
                    json.loads(line)
                    process_tool_call(line)
                except json.JSONDecodeError:
                    continue
        
        # Return joined actions or None if none found
        return "\n".join(tool_calls) if tool_calls else None

    async def _process_step_async(self, step_idx: int, messages: List[Dict[str, Any]], trajectory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Override _process_step_async to customize the async processing for Qwen25VL.
        
        Args:
            step_idx: Index of the current step
            messages: List of message dictionaries for the model
            trajectory: Optional trajectory data for coordinate conversion
            
        Returns:
            Dictionary containing the step results
        """
        # Create a base result with raw response
        result = {
            'step_idx': step_idx,
            'instruction_prompt': None,
            'raw_response': None,
            'parsed_action': None,
            'actions': [],
            'parsing_error': False,
            'error_type': None,
            'error_message': None
        }
        
        # Get model response
        instruction_prompt, response = await self.predict_async(messages)
        result['instruction_prompt'] = instruction_prompt
        result['raw_response'] = response
        
        # If response is None, save the result with error info and return
        if response is None:
            logger.error(f"Failed to get response for step {step_idx}")
            result['parsing_error'] = True
            result['error_type'] = 'no_response'
            return result
        
        # Try to parse response
        try:
            # Use standard parse_response but run it in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            parsed_action = await loop.run_in_executor(
                None, 
                lambda: self.parse_response(response, trajectory, step_idx)
            )
            result['parsed_action'] = parsed_action
            
            # If parsed_action is None, mark as parsing error but continue
            if parsed_action is None:
                logger.error(f"Failed to parse response for step {step_idx}")
                result['parsing_error'] = True
                result['error_type'] = 'parse_error'
                return result
            
            # Try to extract actions
            try:
                # Run extract_actions in thread pool as well
                actions = await loop.run_in_executor(
                    None,
                    lambda: self.extract_actions(parsed_action)
                )
                result['actions'] = actions
                
                # If no actions extracted, mark as extraction error
                if not actions:
                    logger.error(f"No actions extracted for step {step_idx}")
                    result['parsing_error'] = True
                    result['error_type'] = 'extraction_error'
                
            except Exception as e:
                logger.error(f"Action extraction error for step {step_idx}: {e}")
                result['parsing_error'] = True
                result['error_type'] = 'extraction_exception'
                result['error_message'] = str(e)
            
        except Exception as e:
            logger.error(f"Parsing error for step {step_idx}: {e}")
            result['parsing_error'] = True
            result['error_type'] = 'parse_exception'
            result['error_message'] = str(e)
        
        return result

    def extract_actions(self, action: str) -> List[Tuple[str, Any]]:
        """Extract individual actions from parsed response.
        
        Args:
            action: Parsed action string, possibly containing multiple commands
            
        Returns:
            List of tuples containing (action_type, action_params)
        """
        if not action:
            return []
            
        actions = []
        last_move_action = None
        
        # Split by newline to handle multiple actions
        action_lines = action.strip().split('\n')
        
        for line in action_lines:
            line = line.strip()
            
            # Handle computer.terminate
            if line.startswith("computer.terminate"):
                status_match = re.search(r"status=['\"](\w+)['\"]", line)
                if status_match:
                    actions.append(("terminate", status_match.group(1)))
                    continue
            
            # Handle computer.triple_click
            if line.startswith("computer.triple_click"):
                coord_match = re.search(r"x=([\d.]+),\s*y=([\d.]+)", line)
                if coord_match:
                    x, y = map(float, coord_match.groups())
                    actions.append(("triple_click", (x, y)))
                    continue
                
            # Handle pyautogui actions
            if line.startswith("pyautogui."):
                # Extract coordinates for click/moveTo/dragTo actions (allow spaces around '=')
                coord_match = re.search(r"x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)", line)
                if coord_match:
                    x, y = map(float, coord_match.groups())
                    if "click" in line:
                        actions.append(("click", (x, y)))
                    elif "moveTo" in line:
                        last_move_action = ("moveTo", (x, y))
                        actions.append(last_move_action)
                    elif "doubleClick" in line:
                        actions.append(("doubleClick", (x, y)))
                    elif "rightClick" in line:
                        actions.append(("rightClick", (x, y)))
                    elif "dragTo" in line:
                        actions.append(("dragTo", (x, y)))
                        
                # Extract text for write actions
                write_match = re.search(r"message=['\"](.+?)['\"]", line)
                if write_match:
                    text = write_match.group(1)
                    actions.append(("write", text))
                    
                # Extract keys for press/hotkey actions
                keys_match = re.findall(r"keys=\[(.*?)\]", line)
                if keys_match:
                    # Extract and clean keys more thoroughly
                    key_string = keys_match[0]
                    
                    # Handle the case where we have a stringified list inside the list
                    # First, check if the key_string contains list-like strings with quotes
                    list_representation = re.search(r"\[['\"](.*?)['\"]", key_string)
                    if list_representation:
                        # This might be a case of ["['ctrl", "c']"]
                        # Let's extract the content directly
                        cleaned_string = key_string.replace("['", "").replace("']", "").replace("[\"", "").replace("\"]", "")
                        # Use regex to extract properly quoted strings from the cleaned string
                        key_list = re.findall(r"['\"]([^'\"]*)['\"]|(\w+)", cleaned_string)
                    else:
                        # Standard case, use regex to extract properly quoted strings
                        key_list = re.findall(r"['\"]([^'\"]*)['\"]|(\w+)", key_string)
                    
                    # Flatten the tuples and remove empty strings
                    keys = [match[0] or match[1] for match in key_list if match[0] or match[1]]
                    
                    # Normalize 'cmd' to 'ctrl' for cross-platform consistency
                    normalized_keys = []
                    for k in keys:
                        k = k.strip()  # Remove any extra spaces
                        if k.lower() == 'cmd' or k.lower() == 'command':
                            normalized_keys.append('ctrl')
                        else:
                            normalized_keys.append(k)
                    
                    # Correctly differentiate between press and hotkey actions
                    if "hotkey" in line:
                        actions.append(("hotkey", normalized_keys))
                    else:
                        actions.append(("press", normalized_keys))
                    
                # Extract scroll amount
                scroll_match = re.search(r"pyautogui\.scroll\(([-\d]+)\)", line)
                if scroll_match:
                    scroll_amount = int(scroll_match.group(1))
                    # If there's no previous moveTo action, add one to center of screen
                    if not last_move_action:
                        actions.append(("moveTo", (0.5, 0.5)))
                    actions.append(("scroll", scroll_amount))
                    
        return actions
