import re
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional

@dataclass
class Action(ABC):
   
    argument: dict = field(
        metadata={"help": 'argument of action'}
    )
    
    @classmethod
    def get_action_description(cls) -> str:
        return """
Action: action format
Description: detailed definition of this action type.
Usage: example cases
Observation: the observation space of this action type.
"""

  
    @abstractmethod
    def print_action(self):
        return str({
            "action_name": self.action_name,
            "argument": self.argument
        })
    
    @classmethod
    @abstractmethod
    def parse_action_from_text(cls, text: str):
        raise NotImplementedError

    
class Python(Action):
    
    argument: dict = {}
    action_name: str = "Python"
    
    @classmethod
    def get_action_description(cls) -> str:
        return """
## Python Action
* Signature: 
Python(file_path=python_file):
```python
executable_python_code
```
* Description: The Python action will create a python file in the field `file_path` with the content wrapped by paired ``` symbols. If the file already exists, it will be overwritten. After creating the file, the python file will be executed. Remember You can only create one python file.
* Example
- Example1: 
Python(file_path="solution.py"):
```python
# Calculate the area of a circle with a radius of 5
radius = 5
area = 3.1416 * radius ** 2
print(f"The area of the circle is {area} square units.")
```
- Example2:
Python(file_path="solution.py"):
```python
# Calculate the area of a triangle with a base of 10 and a height of 7
base = 10
height = 7
area = 0.5 * base * height
print(f"The area of the triangle is {area} square units.")
```
"""
        
    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional[Action]:
        pattern=[r'Python\(file_path=(.*?)\).*?```python[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',r'Python\(file_path=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',
                 r'Python\(filepath=(.*?)\).*?```python[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',r'Python\(filepath=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',
                 r'(.*)```python[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```']
        for p in pattern:
            matches = re.findall(p, text, flags=re.DOTALL)
            if matches:
                file_path = matches[-1][0].strip() if matches[-1][0].strip() else "solution.py"
                code = matches[-1][-1].strip()
                return cls(argument={"code": code, "file_path": file_path})
        return None
        

    def print_action(self):
        return f"""
Python(file_path="{self.argument.get("file_path")}"):
```python
{self.argument.get("code")}
```
"""



    
class NotebookBlock(Action):
    argument: dict = {}
    action_name: str = "NotebookBlock"
    @classmethod
    def get_action_description(cls) -> str:
        return """
## NotebookBlock Action
* Signature: 
NotebookBlock():
```python
executable python script
```
* Description: The NotebookBlock action allows you to create and execute a Jupyter Notebook cell. The action will add a code block to the notebook with the content wrapped inside the paired ``` symbols. If the block already exists, it can be overwritten based on the specified conditions (e.g., execution errors). Once added or replaced, the block will be executed immediately.
* Restrictions: Only one notebook block can be managed or executed per action.
* Example
- Example1: 
NotebookBlock():
```python
# Calculate the area of a circle with a radius of 5
radius = 5
area = 3.1416 * radius ** 2
area
```
"""
        
    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional[Action]:
        pattern=[r'NotebookBlock\((.*?)\).*?```python[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',r'NotebookBlock\((.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',
                 r'NotebookBlock\((.*?)\).*?```python[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',r'NotebookBlock\((.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',
                 r'(.*)```python[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```']
        for p in pattern:
            matches = re.findall(p, text, flags=re.DOTALL)
            if matches:
                code = matches[-1][-1].strip()
                return cls(argument={"code": code})
        return None
        

    def print_action(self):
        return f"""NotebookBlock():
```python
{self.argument.get("code")}
```
"""
