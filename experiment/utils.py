import re, subprocess
from typing import Optional, Dict, List, Literal

def encoded_message(message: Optional[str],  role: Literal['system', 'user', 'assistant']) -> Dict:
    _valid_roles = ['system', 'user', 'assistant']
    assert role in _valid_roles, f"role should be one of the {_valid_roles}, but got {role}."
    assert isinstance(message, str) and len(message) > 0, "Message should be a String which has more than one character."
    message = {
            "role": role,
            "content": [
                {
                    "type": "text",
                    "text": message if role != "user" else f"Observation: {message}"
                }
            ]
        }
    return message


def process_code(code,is_api=True, mode="standard"):
    try:
        # Extract all Python code blocks enclosed in triple backticks
        all_code_pieces = re.findall(r"```(?:\s*python\s*)(.*?)```", code, re.DOTALL)
    except:
        # Fallback method for extracting code if regex fails
        all_code_pieces = []
        while "```python" in code:
            start_idx = code.index("```python") + len("```python")
            end_idx = code.index("```", start_idx) if "```" in code[start_idx:] else len(code)
            all_code_pieces.append(code[start_idx:end_idx])
            code = code[:start_idx] + code[end_idx + 3:] if "```" in code[start_idx:] else code[:start_idx]
    code_pieces = []
    code_lines = []
    if len(all_code_pieces) == 0:
         return 'Please only generate a single, complete code block wrapped in One Single triple backticks ```python\nYour Code Here\n```', False
    for i, code_piece in enumerate(all_code_pieces):
        if not is_api:
            # Strip and reformat code when not in API mode
            code_pieces.append("\n".join(code_piece.split("\n")).strip())
        else:
            # Process code for API mode
            code_lines.extend(code_piece.strip().split("\n"))
            # Ensure the final output is displayed using print
            code_pieces.append("\n".join(code_lines).strip())
    
    code = '\n'.join(code_pieces)
    if not any("print(" in line for line in code_lines) and is_api:
        return 'Please only generate a single, complete code block with a print method to show final result wrapped in One Single triple backticks ```python\nYour Code Here\n```', False
        
    return code, True


def execute_code(code, code_file="Datasets/MATH/toolkit/tmp3"):

    f = open(f"{code_file}.py", "w")
    code = code.split("\n")

    f.write("\n".join([
        "import math",
        "import numpy",
        "import numpy as np",
        "import sympy, scipy",
        "from sympy import symbols, Eq, solve",
        "import pandas as pd",
        "import pandas",
        "from tools import *"
        "\n"
    ]))

    f.write("\n".join(code))

    f.close()
    i = 0
    while i < 3:
        try:
            result = subprocess.run(
                ['python', f'{code_file}.py'], capture_output=True, check=False, text=True, timeout=10)
        except Exception as e:
            if code_file in str(e):
                i += 1
                continue
            else:
                return False, e
        else:
            if result.returncode != 0:
                error_msg = result.stderr.strip()
                msgs = error_msg.split("\n")
                new_msgs = []
                want_next = False
                for m in msgs:
                    if "Traceback" in m:
                        new_msgs.append(m)
                    elif m == msgs[-1]:
                        new_msgs.append(m)
                    elif code_file in m:
                        st = m.find('"/') + 1
                        ed = m.find(f'/{code_file}.py') + 1
                        if st > 0 and ed > 0:
                            clr = m[st:ed]
                            m = m.replace(clr, "")
                            new_msgs.append(m)
                            want_next = True
                    elif want_next:
                        new_msgs.append(m)
                        want_next = False
                error_msg = "\n".join(new_msgs)
                return False, error_msg.strip()
            else:
                output = result.stdout
                return True, output.strip()
           
    return False, "Code run time out!"


def remove_quote(text: str) -> str:
    """ 
    If the text is wrapped by a pair of quote symbols, remove them.
    In the middle of the text, the same quote symbol should remove the '/' escape character.
    """
    for quote in ['"', "'", "`"]:
        if text.startswith(quote) and text.endswith(quote):
            text = text[1:-1]
            text = text.replace(f"\\{quote}", quote)
            break
    return text.strip()

def parse_retrieval_msg(text: str) -> Optional[Dict]:
    pattern=[r'Retrieve_api\(api_name=(.*?)\).*?```docstring[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',r'Retrieve_api\(api_name=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',
                r'Retrieve_api\(api_name=(.*?)\).*?```docstring[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',r'Retrieve_api\(api_name=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```']
    for p in pattern:
        matches = re.findall(p, text, flags=re.DOTALL)
        if matches:
            tool_name = matches[-1][0].strip()
            docstring = matches[-1][2].strip()
            return {"tool_name": remove_quote(tool_name), "docstring": docstring}
    
    return None

def extract_functions_and_imports(code):
    lines = code.splitlines()
    functions = []
    imports = []
    inside_function = False
    current_function = []

    for line in lines:
        stripped_line = line.strip()
        
        if stripped_line.startswith("import ") or stripped_line.startswith("from "):
            imports.append(line)
        
        elif stripped_line.startswith("def ") and not inside_function:
            inside_function = True
            current_function = [line]
            
        elif inside_function:
            if line.startswith(" ") or line.startswith("\t"):
                current_function.append(line)
            else:
                functions.append("\n".join(current_function))
                inside_function = False

    if inside_function:
        functions.append("\n".join(current_function))

    all_code = "\n".join(imports) + "\n\n" + "\n\n".join(functions)
    return all_code