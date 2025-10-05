from .base_action import Action
from typing import Optional, List, Union, Dict
import re
from .utils import remove_quote

class Retrieval(Action):
    argument: dict = {}
    action_name: str = "Retrieve_api"
    
    @classmethod
    def get_action_description(cls) -> str:
        return """
## Retreive API Action
* Signature:  Retrieve_api(api_name=The name of the tool you want to retrieve):
```docstring
A description of the api's functionality.
```
* Description: The Retrieve API Action searches the ToolNet for relevant APIs based on query similarity. You must provide two parameters: api_name and docstring. The api_name parameter specifies the name of the API you need, while the docstring parameter describes the functionality and usage of the desired API.
* Examples:
  - Example1: 
Retrieve_api(api_name="calculate_mean"):
```docstring
calculate_mean(data) calculates the mean (average) of a given dataset. It takes one input parameter: 'data', which is a list of numerical values, and returns the mean of the dataset.
```
 - Example2: 
Retrieve_api(api_name="linear_regression_fit"):
```docstring
linear_regression_fit(X, y) fits a linear regression model to the given dataset. It takes two input parameters: 'X', the feature matrix, and 'y', the target variable, and returns the fitted model.
```
"""

    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional[Action]:
        pattern=[r'Retrieve_api\(api_name=(.*?)\).*?```docstring[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',r'Retrieve_api\(api_name=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',
                 r'Retrieve_api\(api_name=(.*?)\).*?```docstring[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',r'Retrieve_api\(api_name=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```']
        for p in pattern:
            matches = re.findall(p, text, flags=re.DOTALL)
            if matches:
                tool_name = matches[-1][0].strip()
                docstring = matches[-1][2].strip()
                return cls(argument={"tool_name": remove_quote(tool_name), "docstring": docstring})
        return None
    
   
    def print_action(self):
        return f"""
Retrieve_api(api_name={self.argument.get("tool_name")}):
```docstring
{self.argument.get("docstring")}
```
"""    

class CreateTool(Action):
    argument: dict = {}
    action_name: str = "Create_tool"
    
    @classmethod
    def get_action_description(cls) -> str:
        return '''
## Create tool Action
* Description: The Create Tool action allows you to develop a new tool and temporarily store it in a private repository accessible only to you. Each invocation creates a single tool at a time. You can repeatedly use this action to build smaller components, which can later be assembled into the final tool.
* Signature: 
Create_tool(tool_name=The name of the tool you want to create):
```python
The source code of tool
```
* Examples:
Create_tool(tool_name="calculate_correlation"):
```python
def calculate_correlation(data_x, data_y):
    """
    Calculate the Pearson correlation coefficient between two datasets.

    Parameters:
        data_x (list): A list of numerical values representing the first dataset.
        data_y (list): A list of numerical values representing the second dataset.

    Returns:
        float: The Pearson correlation coefficient between the two datasets.
    """
    if len(data_x) != len(data_y):
        return "Datasets must have the same length."
    mean_x = sum(data_x) / len(data_x)
    mean_y = sum(data_y) / len(data_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(data_x, data_y))
    denominator = (sum((x - mean_x)**2 for x in data_x) * sum((y - mean_y)**2 for y in data_y))**0.5
    if denominator == 0:
        return "Undefined correlation (division by zero)."
    return numerator / denominator
```
'''
    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional[Action]:
        pattern=[r'Create_tool\(tool_name=(.*?)\).*?```python[ \t]*(\w+)?[ \t]*\r?\n(.*?)[\r\n \t]*```',r'Create_tool\(tool_name=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*?)[\r\n \t]*```',
                 r'Create_tool\(tool_name=(.*?)\).*?```python[ \t]*(\w+)?[ \t]*\r?\n(.*?)[\r\n \t]*```',r'Create_tool\(tool_name=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*?)[\r\n \t]*```']
        for p in pattern:
            matches = re.findall(p, text, flags=re.DOTALL)
            if matches:
                tool_name = matches[-1][0].strip()
                docstring = matches[-1][2].strip()
                return cls(argument={"tool_name": remove_quote(tool_name), "code": docstring})
        return None
    
    def print_action(self):
        return f"""
Create_tool(tool_name={self.argument.get("tool_name")}):
```python
{self.argument.get("code")}
```
"""    

class EditTool(Action):
    argument: dict = {}
    action_name: str = "Edit_tool"
    
    @classmethod
    def get_action_description(cls) -> str:
        return '''
## Edit tool Action
* Description: The Edit Tool action allows you to modify an existing tool and temporarily store it in a private repository that only you can access. You must provide the name of the tool to be updated along with the complete, revised code. Please note that only one tool can be edited at a time.
* Signature: Edit_tool(tool_name=The name of the tool you want to create):
```python
The edited source code of tool
```
* Examples:
Edit_tool(tool_name="calculate_standard_deviation"):
```python
def calculate_standard_deviation(data):
    """
    Calculate the standard deviation of a given dataset.

    Parameters:
        data (list): A list of numerical values.

    Returns:
        float: The standard deviation of the dataset.
    """
    if len(data) == 0:
        return "Dataset is empty."
    mean = sum(data) / len(data)
    variance = sum((x - mean)**2 for x in data) / len(data)
    std_dev = variance**0.5
    return std_dev
```
'''
    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional[Action]:
        pattern=[r'Edit_tool\(tool_name=(.*?)\).*?```python[ \t]*(\w+)?[ \t]*\r?\n(.*?)[\r\n \t]*```',r'Edit_tool\(tool_name=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*?)[\r\n \t]*```',
                 r'Edit_tool\(tool_name=(.*?)\).*?```python[ \t]*(\w+)?[ \t]*\r?\n(.*?)[\r\n \t]*```',r'Edit_tool\(tool_name=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*?)[\r\n \t]*```']
        for p in pattern:
            matches = re.findall(p, text, flags=re.DOTALL)
            if matches:
                tool_name = matches[-1][0].strip()
                docstring = matches[-1][2].strip()
                return cls(argument={"tool_name": remove_quote(tool_name), "code": docstring})
        return None
    
   
    def print_action(self):
        return f"""
Edit_tool(tool_name={self.argument.get("tool_name")}):
```python
{self.argument.get("code")}
```
"""