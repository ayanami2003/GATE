import signal
from typing import Dict, Union, List, Optional
import os
import xml.etree.ElementTree as ET
from functools import wraps
import ast, re


TIMEOUT_DURATION = 60

class timeout:
    def __init__(self, seconds=TIMEOUT_DURATION, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def extract_function_head(function):
    '''
    Given a python function, use rule-based method to extract the function name.
    :param function: a python function described in string.
    :return: the function name.
    '''
    function = function.strip().split("\n")
    function = [func for func in function if "def" in func][0]
    function_head = []
    for func in function:
        if func.startswith("def"):
            func = func[3:]
            if func.startswith(':'):
                func = func[:-1]
            function_head.append(func.strip())
    return function_head

def extract_tool_used(function, tools: list):
    function = function.strip().split('\n')
    used_tools = set()
    for line in function:
        for tool in tools:
            if tool in line and not line.startswith('def'):
                used_tools.add(tool)
    
    return used_tools


def count_args(function_head):
    '''
    Given a python function head, count the number of arguments.
    :param function_head: a python function head.
    :return: the number of arguments.
    '''
    function_head = function_head.strip()
    if function_head.endswith(")"):
        function_head = function_head[:-1]
    if "(" in function_head:
        args = function_head.split("(")[1].strip()
        if args == "":
            return 0
        else:
            return len(args.split(","))
    else:
        return 0

def extract_function_docstring(function):
    '''
    Given a python function, use rule-based method to extract the function docstring.
    :param function: a python function described in string.
    :return:
    '''
    function = function.strip()
    if function.startswith("def"):
        function = function[3:]
    if function.endswith(":"):
        function = function[:-1]
    # return function
    if '"""' in function:
        items = function.split('"""')
    else:
        assert "'''" in function, print(function)
        items = function.split("'''")

    docstring = items[1].strip()
    explanation = docstring.split("\n")[0].strip()
    return (explanation, docstring)



def _validate_message(messages: Union[List[Dict], Dict]):
        require_fields = ["obs", "response", "name"]
        messages = messages if isinstance(messages, list) \
            else [messages]
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("Variable Message Must be dict")
            elif list(message.keys()) != require_fields:
                raise KeyError(f"Input Message's Keys Must be {require_fields}")
        
        return messages
    
def encode_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) == 2:
            messages = args[-1]
        elif 'messages' in kwargs:
            messages = kwargs["messages"]
        messages = _validate_message(messages)
        encoded_messages = []
        for message in messages:
            encoded_obs = {
                "role": "user",
                "content": f"Observation: {message['obs']}"  
            }
            encoded_thought_action = {
                "role": "assistant",
                "content": message['response']
            }
            encoded_messages.append(encoded_obs)
            encoded_messages.append(encoded_thought_action)
        if len(args) == 2:
            args[-1] = encoded_messages
        elif 'messages' in kwargs:
            kwargs["messages"] = encoded_messages
        
        return func(*args, **kwargs)
    return wrapper

def find_function_calls_in_function(source_code, function_name):
    tree = ast.parse(source_code)
    
    class FunctionCallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.calls = []
            self.in_target_function = False
        
        def visit_FunctionDef(self, node):
            # Process only the function matching the given function name
            if node.name == function_name:
                self.in_target_function = True
                self.generic_visit(node)
                self.in_target_function = False
            else:
                # Skip other functions
                pass
        
        def visit_Call(self, node):
            # Only collect calls if we are inside the target function
            if self.in_target_function and isinstance(node.func, ast.Name):
                self.calls.append(node.func.id)
            self.generic_visit(node)
    
    visitor = FunctionCallVisitor()
    visitor.visit(tree)
    return visitor.calls


def extract_format(input_string):

    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    answer_names = []
    answers = []
    for name, value in matches:
        answer_names.append(name)
        value = value.strip("'\"")
        if ',' in value:
            value_list = [v.strip("'\" ").strip() for v in value.split(',')]
        else:
            value_list = value
        answers.append(value_list)
    

    return answer_names, answers

def is_equal(response, label):
    try:
        if response.strip() == label.strip():
            return True
        else:
                return abs(float(response) - float(label)) < 1e-6
    except:
        return False




def extract_first_number(string):
    # This regular expression will match any number including decimals and negative numbers,
    # and possibly followed by a percentage sign.
    match = re.search(r'-?\d+\.?\d*%?', string)
    if match:
        return match.group()
    else:
        return None

def calc_acc(data, pred, gold):
    
    error_scale = 0.04
    
    correct_num = False
     
    if data['meta_data']['question_type'] == 'numerical':
        if gold[-1] != '%':
            gold_float = float(gold)
        else:
            gold_float = float(gold[:-1]) / 100

        try:
            pred_float = extract_first_number(pred)
            if pred_float[-1] != '%':
                pred_float = float(pred_float)
            else:
                pred_float = float(pred_float[:-1]) / 100

            if gold_float == 0:
                return abs(pred_float - gold_float) <= error_scale / 4

            lower_bound = min(gold_float * (1-error_scale), gold_float * (1+error_scale))
            upper_bound = max(gold_float * (1-error_scale), gold_float * (1+error_scale))
            
            print(pred_float)
            if lower_bound < pred_float and upper_bound > pred_float:
                correct_num = True
        except:
            return False
                
    else:  # question type is multiple choice 
        if gold.lower() == pred[:len(gold)].lower():
            return True

    return correct_num


import ast
import astor

def remove_function_docstring(code: str) -> str:
    """
    Removes the docstring from all functions in the provided Python code.

    Args:
        code (str): The Python code as a string.

    Returns:
        str: The modified Python code with function docstrings removed.
    """
    class DocstringRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if (len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Str)):
                # Remove the first statement if it is a docstring
                node.body.pop(0)
            return self.generic_visit(node)

    # Parse the code into an AST
    tree = ast.parse(code)
    # Remove docstrings
    remover = DocstringRemover()
    tree = remover.visit(tree)
    # Generate modified code
    modified_code = astor.to_source(tree)

    # Remove empty lines
    lines = modified_code.split('\n')
    cleaned_lines = [line for line in lines if line.strip()]  # Remove empty lines
    return '\n'.join(cleaned_lines)


TEXTCRAFT_HEADER = """
import sys, re
from typing import List
from textcraft.env import TextCraft
# You need to rename this path
env = TextCraft(minecraft_dir="path/to/textcraft")
global done
done = False
def step(command: str) -> str:
    global done
    obs, _, local, _, _ = env.step(command)
    return obs, local
def check_inventory() -> str:
    obs, _ = step('inventory')
    # return the inventory present in the observation
    # Example output: Inventory: [oak planks] (2)
    return obs
def get_object(target: str) -> None:
    obs, _ = step("get " + target)
    print(obs)
def craft_object(target: str, ingredients: List[str]) -> None:
    obs, _ = step("craft " + target + " using " + ", ".join(ingredients))
    print(obs)
\n
"""

TOOL_DESCRIPTION = """
Please package the tool into a suitable API based on your historical tool usage and understanding of the tool. In this round, use the following action to respond:
## Send api Action
* Signature: {
    "action_name": "Send_api",
    "argument": [{
        "api_name": api's name, which should be the same as the tool,
        "docstring": The functional description of an API.
        "note": A brief description of the specific scenario or problem this API is intended to address.
        "demo": usage example of api,
    }]
}
* Description: The Send_api Action is used to submit finalized APIs or tools to the SolvingAgent. The api_name corresponds to the name of the API or tool generated by the ToolAgent, `docstring` provides detailed usage information, `note` includes a brief description of the specific scenario or problem this API is intended to address, and `demo` includes a practical usage example. This action requires sending the data in JSON format, and if multiple APIs need to be sent, they should be packaged in a list format.
Please note that this action should only be used after you've checked all the tools and selected the appropriate APIs and tools. It signifies the completion of the tool creation task.
* Examples:
{
    "action_name": "Send_api",
    "argument": [
        {
            "api_name": "linear_equation_solver",
            "docstring": "`linear_equation_solver` solves a linear equation of the form ax + b = c, where a is non-zero. It takes the coefficients (a, b, c) as inputs and returns the solution for x. If the equation is invalid due to a being zero, it raises an error."
            "note": "The API must accurately solve for x in a linear equation ax + b = c, where a \neq 0, and return the solution as a numeric value.",
            "demo": "Query: Solve 2x + 3 = 7; \nExample: linear_equation_solver(2, 3, 7)\nExplanation: This example solves the linear equation 2x + 3 = 7 by isolating x to find its value."
        }
    ]
}
"""