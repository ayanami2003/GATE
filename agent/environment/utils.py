import signal
import os
import hashlib
import shutil
from typing import Dict, Union, List, Optional
import os
import pandas as pd
import json
import xml.etree.ElementTree as ET
from functools import wraps
import yaml
import ast

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
    function = [func for func in function if "def" in func]
    function_head = []
    for func in function:
        if func.startswith("def"):
            func = func.strip()[3:]
            if func.startswith(':'): 
                func = func[:-1]
            func = func.split("(")[0]
            function_head.append(func.strip())
    return function_head


def extract_tool_used(program):
    """Get all function calls that are not part of the header and map aliases to full module names"""
    parsed = ast.parse(program)
    func_names = []
    header_func_names = ["print"]
    alias_map = {}

    # Step 1: Extract aliases and module names from import statements
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                alias_map[alias.asname or alias.name] = alias.name
                func_names.append(alias.name)  # Add module name (e.g., numpy)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            # Extract the main library name (e.g., 'scipy' from 'scipy.integrate')
            main_module_name = module_name.split('.')[0]
            func_names.append(main_module_name)  # Add main library name
            
            for alias in node.names:
                # Map function to its module and record both function and module names
                alias_map[alias.asname or alias.name] = module_name
                func_names.append(alias.name)  # Add function name (e.g., quad)

    # Step 2: Extract function calls and replace aliases
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                try:
                    base = node.func.value.id  # e.g., 'np'
                    func_name = node.func.attr  # e.g., 'full'
                    # Replace alias with full module name if exists
                    full_base = alias_map.get(base, base)
                    func_names.append(full_base)  # Add module name only
                except AttributeError:
                    pass
            elif isinstance(node.func, ast.Name):
                try:
                    func_name = node.func.id
                    # Replace alias with full module name if exists
                    full_func_name = alias_map.get(func_name, func_name)
                    func_names.append(full_func_name)  # Add module or function name
                except AttributeError:
                    pass

    # Remove header functions and duplicates
    func_names = list(set(func_names) - set(header_func_names))
    
    return func_names



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
        require_fields = ["obs", "thought", "action", "name"]
        messages = messages if isinstance(messages, list) \
            else [messages]
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("Variable Message Must be dict")
            elif list(message.keys()) != require_fields:
                print(message.keys())
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
                "content": [
                    {
                        "type": "text",
                        "text": f"Observation: {message['obs']}"    
                    }
                    ]
            }
            encoded_thought_action = {
                "role": "assistant",
                "content":[
                    {
                        "type": "text",
                        "text": f'"Thought": {message["thought"]},\n"Action": {message["action"]}'
                    }
                ]
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


import math
import numpy
import ast
import esprima
 
Similarity = []

def point(x, y):
    return '[' + str(x) + ',' + str(y) + ']'
 
 
class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.seq = []
 
    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.seq.append(type(node).__name__)
 
    def visit_FunctionDef(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.seq.append(type(node).__name__)
 
    def visit_Assign(self, node):
        self.seq.append(type(node).__name__)

class JavaScriptCodeVisitor:
    def __init__(self):
        self.seq = []

    def visit(self, node):
        if not hasattr(node, "type"):
            return  # 如果节点没有类型属性，直接跳过

        # 添加当前节点类型到序列
        self.seq.append(node.type)

        # 根据节点类型提取详细信息
        if node.type == "BinaryExpression":
            self.seq.append(f"BinOp({node.operator})")  # 操作符，如 '+', '-', '*'
        elif node.type == "Identifier":
            self.seq.append(f"Name({node.name})")  # 变量名
        elif node.type == "Literal":
            self.seq.append(f"Literal({node.value})")  # 字面量
        elif node.type == "ReturnStatement":
            self.seq.append("Return")
        elif node.type == "VariableDeclarator":
            if hasattr(node, "id") and hasattr(node.id, "name"):
                self.seq.append(f"Name({node.id.name})")
        elif node.type == "CallExpression":
            if hasattr(node.callee, "name"):
                self.seq.append(f"Call({node.callee.name})")

        # 递归处理子节点
        for key in node.__dict__:
            value = getattr(node, key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, esprima.nodes.Node):
                        self.visit(item)
            elif isinstance(value, esprima.nodes.Node):
                self.visit(value)
                
 
class CodeParse(object):
    def __init__(self, codeA, codeB, languageA="python", languageB="python"):
        self.visitorB = None
        self.visitorA = None
        self.codeA = codeA
        self.codeB = codeB
        self.languageA = languageA
        self.languageB = languageB
        self.seqA = ""
        self.seqB = ""
        self.work()

    def parse_python_code(self, code):
        """Parse Python code to AST sequence."""
        node = ast.parse(code)
        visitor = CodeVisitor()
        visitor.visit(node)
        return visitor.seq

    def parse_javascript_code(self, code):
        """Parse JavaScript code to AST sequence."""
        js_ast = esprima.parseScript(code, tolerant=True)
        visitor = JavaScriptCodeVisitor()
        visitor.visit(js_ast)
        return visitor.seq

    def work(self):
        """Parse both code snippets based on their language."""
        if self.languageA == "python":
            self.seqA = self.parse_python_code(self.codeA)
        elif self.languageA == "javascript":
            self.seqA = self.parse_javascript_code(self.codeA)
        else:
            raise ValueError(f"Unsupported language: {self.languageA}")

        if self.languageB == "python":
            self.seqB = self.parse_python_code(self.codeB)
        elif self.languageB == "javascript":
            self.seqB = self.parse_javascript_code(self.codeB)
        else:
            raise ValueError(f"Unsupported language: {self.languageB}")
 
 
class CalculateSimilarity(object):
    def __init__(self, A, B, W, M, N):
        self.A = A
        self.B = B
        self.W = W
        self.M = M
        self.N = N
        self.similarity = []
        self.SimthWaterman(self.A, self.B, self.W)
 
    def score(self,a, b):
        if a == b:
            return self.M
        else:
            return self.N
 
    def traceback(self,A, B, H, path, value, result):
        if value:
            temp = value[0]
            result.append(temp)
            value = path[temp]
            x = int((temp.split(',')[0]).strip('['))
            y = int((temp.split(',')[1]).strip(']'))
        else:
            return
        if H[x, y] == 0:  
            xx = 0
            yy = 0
            sim = 0
            for item in range(len(result) - 2, -1, -1):
                position = result[item]
                x = int((position.split(',')[0]).strip('['))
                y = int((position.split(',')[1]).strip(']'))
                if x == xx:
                    pass
                elif y == yy:
                    pass
                else:
                    sim = sim + 1
                xx = x
                yy = y
            self.similarity.append(sim * 2 / (len(A) + len(B)))
 
        else:
            self.traceback(A, B, H, path, value, result)
 
    def SimthWaterman(self, A, B, W):
        n, m = len(A), len(B)
        H = numpy.zeros([n + 1, m + 1], int)
        path = {}
        for i in range(0, n + 1):
            for j in range(0, m + 1):
                if i == 0 or j == 0:
                    path[point(i, j)] = []
                else:
                    s = self.score(A[i - 1], B[j - 1])
                    L = H[i - 1, j - 1] + s
                    P = H[i - 1, j] - W
                    Q = H[i, j - 1] - W
                    H[i, j] = max(L, P, Q, 0)
 
                    path[point(i, j)] = []
                    if math.floor(L) == H[i, j]:
                        path[point(i, j)].append(point(i - 1, j - 1))
                    if math.floor(P) == H[i, j]:
                        path[point(i, j)].append(point(i - 1, j))
                    if math.floor(Q) == H[i, j]:
                        path[point(i, j)].append(point(i, j - 1))
        end = numpy.argwhere(H == numpy.max(H))
        for pos in end:
            key = point(pos[0], pos[1])
            value = path[key]
            result = [key]
            self.traceback(A, B, H, path, value, result)
 
    def Answer(self): 
        return sum(self.similarity) / len(self.similarity) if self.similarity else 0.0

def calculate_similarity(codeA, codeB, language):
    if language.lower() not in ["python", "javascript"]:
        raise ValueError(f"Unsupported language: {language}")
    try:
        AST = CodeParse(codeA, codeB, language, language)
    except:
        return 0.0
    RES = CalculateSimilarity(AST.seqA, AST.seqB, 1, 1, -1/3)
    return RES.Answer()



