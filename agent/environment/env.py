from ..action import *
from typing import List, Dict, Union, Optional
import pandas as pd
from .utils import *
from collections import defaultdict
from .notebook import *
import logging
import subprocess, os
import numpy as np


TextCraft_Header = """
import sys, re
import re
from pathlib import Path 
from temp_tools import *
from tools import *     
"""     

DABench_Header = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy import stats, optimize
import sys
from pathlib import Path
from temp_tools import *
from tools import *    
"""


TAB_Header = """
import pandas as pd
from temp_tools import *
from tools import * 
"""

MATH_Header = """
import numpy as np
import math
import scipy
import sympy as sp
import cmath
from temp_tools import *
from tools import * 
"""

DATE_Header = """
from datetime import date, time, datetime
from dateutil.relativedelta import relativedelta
from temp_tools import *
from tools import * 
"""

class Env:
    def __init__(self, toolkit_path, basic_tools, **kwargs):
        self.basic_tool = basic_tools
        self.tmp_dir = kwargs.get("tmp_dir", os.path.join(os.getcwd(), "mnt"))
        self.temp_code = os.path.join(self.tmp_dir, "tmp0.py")
        self.toolkit_path = toolkit_path
        self.notebook = Notebook(timeout=30, cwd=self.tmp_dir)
        self.temp_toolkit = {}
        self.temp_tool_code = os.path.join(self.tmp_dir, "temp_tools.py")
        self.task_type = kwargs.get("task_type", "")
        self.times = 0
        self.duplication_tool = []
        self.use_api=True
        
    def clear_env(self):
        self.temp_toolkit = {}
        self.notebook = Notebook(timeout=30, cwd=self.tmp_dir)
        self.duplication_tool = []
        self.times = 0
        self.use_api = True
        return 
    
    def detect_tool(self, tool_code, skills):       
        lines = self.count_effective_lines(tool_code)
        if lines <= 2:
            return "Failed to Create/Edit tool. Your current code contains very few actual lines, as it appears you are merely calling an existing function or the logic is overly simplistic. Please take the time to think more deeply and aim to develop a tool that is both more generalizable and robust."
        pattern = r"def\s+(\w+)"
        function_names = re.findall(pattern, tool_code)
        skills_names = [skills[skill]["name"] for skill in skills]
        redundant_skills = list(set(function_names) & set(skills_names))

        if redundant_skills and self.times == 0:
            self.times += 1
            redundant_skill = ", ".join(redundant_skills)
            redundant_code = []
            for skill in redundant_skill:
                for existing_skill in skills:
                    if skill == skills[existing_skill]["name"]:
                        redundant_code.append(skills[existing_skill]["code"])
                        self.temp_toolkit[existing_skill] = {
                            "code": code, "docstring": skills[existing_skill]["docstring"], "level": skills[existing_skill]["level"], "demo": skills[existing_skill]["demo"],
                            "created": False, "api": False
                        }
            redundant_code = "```python\n" + "\n".join(redundant_code) + "\n```"
            return f"The code includes redundant tool: {redundant_skill} within the existing functions." + \
                """If you think the code of redundant skill can directly solve the current task through calling, please respond with a JSON Format:
Tool: {  
    "tool_name": "Name of Existing tools"
}
e.g.
Thought: ...
Tool: {
    "tool_name": "calculate_power"
}
""" + f"""
Additionally, if the redundant skill’s code is outdated or no longer effectively addresses the current task, you can regenerate a more reusable and refined function to rewrite the skill. For example, if the original function is limited to mining 3 items while the task requires mining 5, you would need to rewrite the function. Similarly, if the original function uses a stone pickaxe to mine materials but an iron pickaxe is now available, the function would also become outdated and require updating.
The reduandt skill's code:
{redundant_code}
""" 
        if self.times == 0:
            for skill in skills:
                score = calculate_similarity(tool_code, skills[skill]["code"], "python")
                if score > 0.99:
                    self.duplication_tool.append(skill)
                    self.times +=1
            if self.duplication_tool:
                duplicate_code = ""
                for tool in self.duplication_tool:
                    code = skills[tool]["code"]
                    duplicate_code += "```python\n" + code + "\n```\n"
                return f"Your code is highly similar in structure to the following existing tools. This might be due to a lack of innovation in your implementation. If your intended functionality is similar to these programs, consider taking a step back and building a more abstract and reusable tool based on them to address the problem more effectively.\n{duplicate_code}"

        return None
    

    def makeplan(self, action, **kwargs):
        plan = action.argument.get("plan", "")

        return ("You've Successfully make a plan.", True) if plan else ("Please Provide a valid plan", False)
    
    def remove_comments(self, code_string):
        code_string = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', '', code_string, flags=re.DOTALL)
        code_string = re.sub(r'#.*', '', code_string)
        return code_string

    def count_effective_lines(self, code_string):
        cleaned_code = self.remove_comments(code_string)
        lines = cleaned_code.split('\n')
        effective_lines = [line for line in lines if line.strip()]
        return len(effective_lines)

    def send_api(self, action: Action, **kwargs):
        argument = action.argument if isinstance(action.argument, list) \
            else [action.argument]
        toolnet = kwargs.get("toolnet")
      
        wrong_apis = [api.get("api_name", "") for api in argument 
            if api.get("api_name", "") not in self.temp_toolkit.keys() and api.get("api_name", "") not in self.basic_tool]
        
        if wrong_apis:
            return f"{', '.join(wrong_apis)}'s tool does not exist, please check again", False
        
        results = []
        for api in argument:
            name, docstring, demo = api["api_name"], api.get("docstring"), api.get("demo")
            if not docstring or not demo:
                return f"You haven't provided either the demo or the docstring for {name}. Please check again.", False
            if name in self.temp_toolkit:
                self.temp_toolkit[name].update({"docstring": docstring, "demo": demo,"api": True})
            if name in toolnet.skills.keys():
                results.append(f'### Tool: `{toolnet.skills[name]["name"]}`\n**Docstring**: {toolnet.skills[name]["docstring"]}\n**Usage Examples**: {toolnet.skills[name]["demo"]}\n')
                continue
            results.append(f'### Tool: `{name}`\n```python\n{self.temp_toolkit[name]["code"]}\n```\n**Usage Example**: {demo}\n')
    
        return '\n'.join(results), True
        
    def tool_request(self, action: Action, **kwargs):
        request = action.argument.get("request", "")
        if not request:
            return "The request cannot be empty. Please provide a valid request.", False
        return request, True

    def reply(self, action: Action, **kwargs):
        message = action.argument.get("message", "")
        if not message:
            return "The message cannot be empty. Please provide a valid reply.", False
        
        return  f"You recieve a reply from ToolAgent: {message}", True
        
    def feedback(self, action: Action, **kwargs):
        feedback = action.argument.get("feedback", "")
        passed = action.argument.get("passed", False)
        
        if not passed and not feedback:
            return "If you believe the tool did not pass the test, you must provide detailed feedback in the feedback field.", False
    
        return "", True
        
    def create_tool(self, action: Action, **kwargs):
        toolnet = kwargs.get("toolnet")
        tool_name = action.argument.get("tool_name")
        code = action.argument.get("code")
        if len(toolnet.skills) < 10:
            temp_tools = list(self.temp_toolkit.keys()) + self.basic_tool
            self.use_api = False
        else:
            temp_tools = list(self.temp_toolkit.keys())
         
        if 'Create_tool' in code and '```' in code:
            return "Failed to create tools. Only one tool can be created per response.", False
        
        used_functions = extract_tool_used(code)
        used_api = len(set(temp_tools) & set(used_functions))
        
        if used_api < 1 and self.use_api and len(toolnet.skills) >= 10:
            self.use_api = False
            return ("Failed to create tools. The created code must include calls to some Existing Tools. "
                    "Please check again."), False

        function_count = sum(1 for line in code if line.startswith("def") and line.strip().endswith(":"))
        if function_count > 1:
            return ("Failed to create the tool because a single Create_tool action can only create one tool. "
                    "For multiple tools, call Create_tool multiple times."), False
        # Add tool to toolkit and write to file
        self.temp_toolkit[tool_name] = {"code": code, "docstring": "", "level": 0, "demo": "", "created": True, "api": False, "name": tool_name}
        
        with open(self.temp_tool_code, 'w') as f:
            f.write('from tools import *\nimport re\n')
            for _, info in self.temp_toolkit.items():
                if info["created"]:
                    f.write(info["code"] + '\n\n')
                                        
        return f"You have successfully created the tool '{tool_name}'", True
            
    def edit_tool(self, action: Action, **kwargs):
        tool_name = action.argument.get("tool_name")
        code = action.argument.get("code")
        toolnet = kwargs.get("toolnet")
        
        
        if len(toolnet.skills) < 10:
            tool_names = list(self.temp_toolkit.keys()) + self.basic_tool
            self.use_api  = False
        else:
            tool_names = list(self.temp_toolkit.keys())
    
        # Check if the tool exists
        if tool_name not in tool_names:
            return (f"The tool '{tool_name}' has not been created. Please check again. "
                    "If you want to create this tool, please use the 'Create_tool' action."), False
    
        # Check if the created tool uses existing APIs
        used_functions = extract_tool_used(code)
        used_api = len(set(tool_names) & set(used_functions))
       
        if used_api < 1 and self.use_api and len(toolnet.skills) >= 10:
            self.use_api = False
            return ("Failed to edit tools. The edited code must include calls to some retrieved APIs in the custom library. "
                    "Please check again."), False

        # Ensure that the tool is editable and check for identical code
        temp_code = self.temp_toolkit[tool_name]["code"]
        if not self.temp_toolkit[tool_name]["created"]:
            return f"Failed to edit tools. The tool '{tool_name}' is a retrieved tool; you do not have permission to edit it.", False
        if temp_code.strip() == code.strip():
            return f"Failed to edit tools. The new code for '{tool_name}' is identical to the previous version. Please review it.", False
        if 'Edit_tool' in code and '```' in code:
            return 'Failed to edit tools. You should only edit one tool per response.', False
    
        # Count functions and check for disallowed print statements
        functions = extract_function_head(code)
        if len(functions) != 1:
            return ("Failed to edit the tool. A single Edit_tool action can only edit one tool. "
                    "To edit multiple tools, please call Edit_tool multiple times."), False
        
         # Add tool to toolkit and write to file
        
        function = functions[0]

        if function != tool_name:
            self.temp_toolkit[function] = self.temp_toolkit[tool_name]
            self.temp_toolkit[function]["code"] = code
            self.temp_toolkit.pop(tool_name)
        else:
            # Update the tool code
            self.temp_toolkit[tool_name]["code"] = code
            
        # Write the updated tool to the file
        with open(self.temp_tool_code, 'w') as f:
            f.write('from tools import *\nimport re\n')
            for _, info in self.temp_toolkit.items():
                if info["created"]:
                    f.write(info["code"] + '\n\n')
        
        return (f'You have successfully edited the tool "{tool_name}".', True) if function == tool_name else \
           (f"You have successfully edited the tool {tool_name} to {function}", True)
    
    def retrieve_api(self, action: Action, **kwargs):
        toolnet = kwargs.get('toolnet')
        docstring = action.argument.get("docstring", "")
        if not docstring:
            return "Please provide a valid docstring for the tool you want to retrieve.", False

        retrieval_func = toolnet.retrieve_skills
        top_k_tools = retrieval_func(query=docstring)
        if not top_k_tools:
            return "Sorry, ToolFlowNet currently only has basic APIs. Please use these basic apis to create tools.", True
        # Store retrieved tools in temp_toolkit
        for tool in top_k_tools:
            self.temp_toolkit[tool["name"]] = {
                "code": tool["code"],
                "demo": tool["demo"],
                "level": tool["level"],
                "created": False,
                "docstring": tool["docstring"],
                "api": True
            }
        # Prepare tool information for display
        results = []
        for idx, tool in enumerate(top_k_tools):
            tool_info = (f"{idx + 1}. Tool Name: {tool['name']}\n"
                        f"Source Code:\n```python\n{tool['code']}\n```"
                        )
            results.append(tool_info)
        return '\n'.join(results), True
    
    def notebookblock(self, action: Action, **kwargs):
        if self.task_type.lower().strip() == "textcraft":
            header = TextCraft_Header 
        elif self.task_type.lower().strip() == "dabench":
            header = DABench_Header
        elif self.task_type.lower().strip() == "math":
            header = MATH_Header
        elif self.task_type.lower().strip() == "date":
            header = DATE_Header
        elif self.task_type.lower().strip() == "tabmwp":
            header = TAB_Header
        else:
            header = ""
        if len(self.notebook.notebook.cells) == 0 and header:
            self.notebook.add_block(header)
        
        agent_name = kwargs.get("agent_name", "")
        code = action.argument.get("code", "")
        code_lines = code.split("\n")
        new_code = []
        issues = [] 
        # Process code lines
        for line in code_lines:
            # Disallow external library imports
            if line.strip().startswith(('import', 'from')) and 'import' in line:
                if any(kw in line for kw in ['tool', 'custom', 'api']):
                    continue
            # Disallow function definitions for SolvingAgent
            if line.strip().startswith("def") and line.strip().endswith(":"):
                if agent_name == "SolvingAgent":
                    issues.append("Function definitions are not allowed; you can directly call the API from the custom library or directly use the tool created by the ToolAgent.")
                    continue
            for tool in self.temp_toolkit:
                if not self.temp_toolkit[tool]["created"] and f' {tool}(' in line and line.strip().startswith('def') and line.strip().endswith(':'):
                    issues.append(f"{tool} is a provided API. Please call it directly.")
            for tool in self.basic_tool:
                if f' {tool}(' in line and line.strip().startswith('def') and line.strip().endswith(':'):
                    issues.append(f"{tool} is a basic API. Please call it directly.")
            new_code.append(line)

        code = "\n".join(new_code)
        if issues:
            return "\n".join(issues), False

        result, _ = self.notebook.add_or_replace_block(code)
        return result, True
    
    
    def python(self, action: Action, **kwargs):
        agent_name = kwargs.get("agent_name")
        code = action.argument["code"]
        
        # Ensure at least one print statement exists
        if "print(" not in code:
            return "Please include at least one print statement to display the output.", False
        
        # Open file to write temporary code
        with open(self.temp_code, "w") as f:
            code_lines = code.split("\n")
            new_code = []
            issues = []  # Collect issues here
            # Process code lines
            for line in code_lines:
                # Disallow external library imports
                if line.strip().startswith(('import', 'from')) and 'import' in line:
                    if any(kw in line for kw in ['tool', 'custom', 'api']):
                        continue
                # Disallow function definitions for SolvingAgent
                # if line.strip().startswith("def") and line.strip().endswith(":"):
                #    if agent_name == "SolvingAgent":
                #        issues.append("Function definitions are not allowed; you can directly call the API from the custom library or directly use the tool created by the ToolAgent.")
                #        continue
                # Check for direct API calls in temp toolkit
                # for tool in self.temp_toolkit:
                #     if not self.temp_toolkit[tool]["created"] and f' {tool}(' in line and line.strip().startswith('def') and line.strip().endswith(':'):
                #         issues.append(f"{tool} is a provided API. Please call it directly.")
                # Check for direct API calls in basic tools
                # for tool in self.basic_tool:
                #    if f' {tool}(' in line and line.strip().startswith('def') and line.strip().endswith(':'):
                #       issues.append(f"{tool} is a basic API. Please call it directly.")
                new_code.append(line)

            # If there are issues, return them before proceeding
            if issues:
                return "\n".join(issues), False

            # Write the updated code with imports
            f.write("\n".join([
                "from tools import *",
                "from temp_tools import *",
                "import re",
                "\n"
            ]))
            f.write("\n".join(new_code))

        # Run the code with retries
        for _ in range(3):
            try:
                result = subprocess.run(
                    ['python', self.temp_code], capture_output=True, check=False, cwd=self.tmp_dir,
                    text=True, timeout=30
                )
            except Exception as e:
                if self.temp_code in str(e):
                    continue
                return str(e), False

            # Handle errors or return output
            if result.returncode != 0:
                error_msgs = [m for m in result.stderr.strip().split("\n") if "Traceback" in m or m == result.stderr.strip().split("\n")[-1]]
                return "\n".join(error_msgs).strip(), False
            else:
                return "Code Executed Successfully:\n"+ result.stdout.strip(), True

        return "Code execution timed out!", False
    
    def terminate(self, action: Action, **kwargs):
        result = action.argument.get("result", None)
        if not result:
            return "Please Check your action format, like Terminate(result=...)", False
        
        return result, True
            
    def extract_all_tool_created(self):
        tool_created = []
        for tool in self.temp_toolkit.keys():
            if self.temp_toolkit[tool]["created"] and self.temp_toolkit[tool]["api"]:
                tool_created.append(tool)
            
        return set(tool_created)
    
    def extract_all_tool_used(self):
        solution_code = self.notebook.merge_all_blocks()

        for tool in self.temp_toolkit.keys():
            if self.temp_toolkit[tool]["api"]:
                solution_code += "\n" + self.temp_toolkit[tool]["code"] + "\n"

        tool_names = set(list(self.temp_toolkit.keys()) + self.basic_tool)
        function_names = extract_tool_used(solution_code)
        
        tool_used = set(tool_names) & set(function_names)
        return set(tool_used)

    def extract_tool_dependency(self):
        tool_created = self.extract_all_tool_created()
        tool_used = self.extract_all_tool_used()
        
        tool_created = set(tool_created) & set(tool_used)
        if not tool_created:
            return None

        dependency_relation = defaultdict(list)
    
        for tool in tool_created:
            calls = extract_tool_used(self.temp_toolkit[tool]["code"])
            dependency_relation[tool] = []
            if not calls:
                continue
            dependency_relation[tool] = list(set([(call, 1.0) for call in calls if call in tool_used and call != tool]))
        
        dependency_relation = dict(sorted(
            dependency_relation.items(), 
            key=lambda item: len(list(filter(lambda call: call[0] not in tool_created, item[1])))
        ))
        
        return dependency_relation
    
    def step(self, action: Action, **kwargs):
        action_name = action.action_name.lower()
        action_method = getattr(self, action_name)
       
        if not callable(action_method):
            raise ValueError(f"Please check whether the Method {action_name} exist.")
        
        try:
            with timeout(60,"Action execution time exceeded!"):
               observation, done = action_method(action, **kwargs)
        except TimeoutError as e:
            observation = str(e)
            done = False
        return observation, done