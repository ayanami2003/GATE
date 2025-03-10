from .base_action import Action
from typing import Optional, List, Union, Dict
import re, json5
from .utils import remove_quote

class ToolRequest(Action):
        
    argument: dict = {}
    action_name: str = "Tool_request"
    
    @classmethod
    def get_action_description(cls) -> str:
        return """
## Tool_request Action
* Signature:
{
    "action_name": "tool_request",
    "argument": {
         "request": [
             ...
         ]
    }
}
The Tool Request Action allows you to send tool requirements to the ToolAgent and request it to create appropriate tools. You need to provide the action in a JSON format, where the argument field contains a request parameter that accepts a list. Each element in the list is a string describing the desired tool.
* Example:
{
    "action_name": "tool_request",
    "argument": {
         "request": [
             "I need a tool solves a system of two linear equations with two variables. You need to input six coefficients [a1, b1, c1, a2, b2, c2], representing the equations a1 * x + b1 * y = c1 and a2 * x + b2 * y = c2. The tool returns a list [x, y], where x and y are the solutions to the equations. If the system has no solution or infinitely many solutions, it returns an error message or an empty list."
         ]
    }
}
"""

    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional[Action]:
        pattern = re.compile(r'\{.*\}', re.DOTALL)
        match = pattern.search(text)
        
        if match:
            json_string = match.group(0)
            json_string = json_string.replace('\n', ' ')
            json_string = json_string.replace('\t', '')
            try:
                action_json = json5.loads(json_string)
            except Exception as e:
                return None
        else:
            return None

        action_name = action_json.get("action_name", None)
        if not action_name:
            return None
        elif action_name.lower() != "tool_request":
            return None
        argument = action_json.get("argument", None)
        if not argument:
            return None
        elif "request" not in argument.keys():
            return None
        
        return cls(argument=argument)
    
    def print_action(self) -> str:
        return f"""
{{
  "action_name": "tool_request",
  "argument": {self.argument}
}} 
"""

class Terminate(Action):
    
    argument: dict = {}
    action_name: str = "Terminate"
    @classmethod
    def get_action_description(cls) -> str:
        return """
# Terminate Action
* Signature: Terminate(result=the result of the task)
* Description: The Terminate action ends the process and provides the task result. The `result` argument contains the outcome or status of task completion.
* Examples:
  - Example1: Terminate(result="A")
  - Example2: Terminate(result="1.23")
"""

    @classmethod
    def parse_action_from_text(cls, text) -> Optional[Action]:
        matches = re.findall(r'Terminate\(result=(.*)\)', text, flags=re.DOTALL)
        if matches:
            output = matches[-1] 
            return cls(argument={"result": output})
        return None

    def print_action(self):
        return f"""
Terminate(result={self.argument.get("result", "")})
"""
