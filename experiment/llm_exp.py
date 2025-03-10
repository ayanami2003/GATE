from pydantic import BaseModel, Field, ValidationError, field_validator
from collections import deque
import numpy as np
from typing import Literal, Union, Dict, List, Optional
from toolgraph.net_sturcture import SkillManager
from toolgraph.embedding_model import OpenAIEmbeddingModel
from agent import OpenAILLM
import logging, jsonlines, copy
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from .utils import *
from grader import *
from datetime import datetime

class TestingFrameworkConfig(BaseModel):
    llm_cfg: Union[str, Dict] = Field(..., description="Configuration for the LLM.")
    toolkit_path: str = Field(..., description="Path to the toolkit directory.")
    task_type: str = Field(..., description="Type of task.")
    train_output: str = Field(..., description="Path to the training output file.")
    train_embed_path: str = Field(..., description="Path to store training query's embedding file.")
    top_k_query: int = Field(3, ge=0, description="Number of top queries to retrieve.")
    top_k_tool: int = Field(3,ge=0, description="Number of top tools to retrieve.")
    test_mode: Literal["pipeline", "standard"] = Field(
        "standard", description="Testing mode: 'pipline' or 'standard'."
    )
    basic_tools: List[str] = Field(default_factory=list, description="List of basic tools.")
    retry_times: int = Field(3, ge=0, description="Number of retry attempts.")
    debug_times: int = Field(3, ge=0, description="Number of debug attempts (0 means no debugging).")
    is_primitive: bool = Field(False, description="Whether to run in primitive mode.")
    has_demo: bool = Field(True, description="Whether demo examples are enabled.")
    tool_mode: Literal["api", "direct_tool", "bfs_tool"] = Field(
        "direct_tool", description="Mode of tool operation."
    )
    
    @field_validator("toolkit_path", "train_output")
    def validate_paths(cls, value: str):
        """Ensure provided paths are non-empty strings."""
        if not value.strip():
            raise ValueError(f"Path '{value}' cannot be empty.")
        if not os.path.exists(value):
            raise ValidationError(f"Path '{value}' does not exist")
        return value
    
    @field_validator("llm_cfg")
    def validate_llm_cfg(cls, value: Union[str, Dict]):
        """Ensure llm_cfg is either a valid string or dictionary."""
        if isinstance(value, str) and not value.strip():
            raise ValueError("llm_cfg string cannot be empty.")
        if isinstance(value, dict) and not value:
            raise ValueError("llm_cfg dictionary cannot be empty.")
        return value
    

class TestingFramework:
    def __init__(self, **kwargs):
        try:
            # Validate input arguments using pydantic
            config = TestingFrameworkConfig(**kwargs)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}")
        # Assign validated values to instance attributes
        self.llm = OpenAILLM(config.llm_cfg)
        self.train_output = config.train_output
        self.toolkit_path = config.toolkit_path
        self.train_embed_path = config.train_embed_path
        self.tool_mode = config.tool_mode
        self.top_k_query = config.top_k_query
        self.top_k_tool = config.top_k_tool
        self.retry_times = config.retry_times
        self.debug_times = config.debug_times
        self.mode = config.test_mode
        self.task_type = config.task_type
        self.basic_tools = config.basic_tools
        self.primitive = config.is_primitive
        self.has_demo = config.has_demo
        self.embedding_model = OpenAIEmbeddingModel()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.code_exec = f"{self.task_type}_{self.llm.model}_{timestamp}"
        # Initialize ToolFlowNet if not in primitive mode
        if not self.primitive:
            self.ToolNet = SkillManager(
                retrieval_top_k=config.top_k_tool,
                basic_tools=config.basic_tools,
                ckpt_dir=config.toolkit_path,
                resume=True
            )
            
    def encoded_message(self, message: Optional[str],  role: Literal['system', 'user', 'assistant']) -> Dict:
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
                
    def retrieve_task(self, query: str, **kwargs):        
        tool_name = kwargs.get("tool_name", "")
    
        with jsonlines.open(self.train_output, 'r') as f:
            train = [
                data for data in f
                if data.get("success") and (not tool_name or (
                    tool_name in data.get("tool_used", "") and f" {tool_name}(" in data.get("code", "")
                ))
            ]
        if not train:
            return [] 
        os.makedirs(self.train_embed_path, exist_ok=True)
        query_embed = np.array(self.embedding_model.create_embedding(query))
        train_embeds = []
        indices = []
        missing_embeds = []
        
        for idx, data in enumerate(train):
            npy_path = os.path.join(self.train_embed_path, f"{data['id']}.npy")
            if os.path.exists(npy_path):
                train_embeds.append(np.load(npy_path))
            else:
                missing_embeds.append((idx, data["question"]))
            indices.append(idx)
        if missing_embeds:
            questions = [q for _, q in missing_embeds]
            generated_embeds = []
            generated_embeds = [self.embedding_model.create_embedding(question) for question in questions]
            for (idx, _), embed in zip(missing_embeds, generated_embeds):
                train_embeds.append(embed)
                npy_path = os.path.join(self.train_embed_path, f"{train[idx]['id']}.npy")
                np.save(npy_path, embed)
        if not train_embeds:
            return [] 
        
        train_embeds = np.array(train_embeds)
        query_norm = np.linalg.norm(query_embed)
        train_norms = np.linalg.norm(train_embeds, axis=1)
        cosine_similarities = np.dot(train_embeds, query_embed) / (train_norms * query_norm)
        max_index = np.argmax(cosine_similarities)
        top_task = [(train[max_index], cosine_similarities[max_index])]
        return top_task
    
    def get_response(self, messages: List):
        assert len(messages) > 0, "Messages to LLM must be a List whose length is more than 0."
        for _ in range(self.retry_times):
            import time
            time.sleep(2)
            status, response = self.llm.get_response(messages)
            if not status:
                if response.lower() in ["context_length_exceeded","rate_limit_exceeded","max_tokens"]:
                    messages.pop(-1)
                    messages = [messages[0]] + messages[2:]
                else:
                    raise Exception(f"Failed to call LLM, response: {response}")
            else:
                break
        
        return response

            
    def retrieve_tool(self, query: str, sys_prompt: str):
        logging.info("Tool Retrieval Begin!!")
        '''
        if not os.path.exists(sys_prompt):
            raise FileExistsError(f"{sys_prompt} does not exist")
        else:
            with open(sys_prompt, 'r') as f:
                sys_prompt = f.read()
            sys_prompt = sys_prompt.replace("===task===", query)
    
        messages = []
        messages.append(self.encoded_message(sys_prompt, role="system"))  
        obs = "Please provide retrieval message based on the task. Let's think step by step."
        messages.append(self.encoded_message(obs, role="user"))
        retrieve_query = {}
        for attempt in range(self.retry_times):
            logging.info(f"Attempt {attempt + 1} Observation: {obs}")
            response = self.get_response(messages)
            logging.info(f"Attempt {attempt + 1} Response: {response}")
            retrieve_query = parse_retrieval_msg(response)
            if retrieve_query is None:
                obs = "You should provide a valid retrieval query, following the format 'Retrieve_api(api_name=API Nanme)```docstring```'. Please check again. For example, Retrieve_api(api_name=\"calculate_sum_of_infinite_circles_areas\"):\n```docstring\ncalculate_sum_of_infinite_circles_areas(radius) calculates the sum of the areas of an infinite sequence of circles based on the radius of the first circle. It returns the total sum of the areas.\n```"
                messages.append(self.encoded_message(response, role="assistant"))
                messages.append(self.encoded_message(obs, role="user"))
                continue
            else:
                break
        '''
        retrieve_query = {"docstring": query}
        if retrieve_query:
            retrieved_tools = self.ToolNet.retrieve_skills(retrieve_query["docstring"])
        else:
            logging.error("Failed to Retrieve Tools")
            raise RuntimeError(f"Failed to Retrieve Tools")

        return retrieved_tools
    
    def bfs_tool(self, retrieval_tools):
        tools_result = copy.deepcopy(retrieval_tools)
        for tool in retrieval_tools:
            tool_name = tool["name"]
            node = self.ToolNet.tool_name_to_nodes[tool_name]
            existing_tool_names = {t["name"] for t in tools_result}
            if tool_name not in existing_tool_names:
                tools_result.append(node._load_to_dict())
                existing_tool_names.add(node.tool_name)
            queue = deque([node])
            while queue:
                current_node = queue.popleft()
                if current_node.tool_name not in existing_tool_names:
                    tools_result.append(current_node._load_to_dict())
                    existing_tool_names.add(current_node.tool_name)
                for idx, weight in enumerate(self.ToolNet.weight_matrix[:, current_node.index]):
                    if weight != np.inf:
                        linked_tool_node = self.ToolNet.index_to_nodes[idx]
                        if linked_tool_node.level >= current_node.level:
                            continue
                        if linked_tool_node.tool_name not in existing_tool_names:
                            queue.append(linked_tool_node)
        
        tools_result = sorted(tools_result, key=lambda t: (t["level"], -t["freq"]))
        tools_result = "\n".join(tool["code"] for tool in tools_result)
        lines = tools_result.split("\n")
        import_statements = set()
        code_lines = []
        for line in lines:
            if line.startswith("from ") or line.startswith("import "):
                import_statements.add(line)
            else:
                code_lines.append(line)
        unique_imports = "\n".join(sorted(import_statements))
        code = '\n'.join(code_lines)
        tools_result = f"{unique_imports}\n\n{code}"
    
        return tools_result
    
    def direct_tool(self, retrieval_tools):
        
        tools_result = copy.deepcopy(retrieval_tools)
        tools_result = "\n".join(tool["code"] for tool in tools_result)
        lines = tools_result.split("\n")
        import_statements = set()
        code_lines = []
        for line in lines:
            if line.startswith("from ") or line.startswith("import "):
                import_statements.add(line)
            else:
                code_lines.append(line)
        unique_imports = "\n".join(sorted(import_statements))
        code = '\n'.join(code_lines)
        tools_result = f"{unique_imports}\n{code}"
        
        return tools_result

    def api(self, retrieval_tools):
        tools_result = copy.deepcopy(retrieval_tools)
        tools_result = sorted(tools_result, key=lambda t: (t["level"], -t["freq"]))
        api_info = []
        for idx, tool in enumerate(tools_result):
            tool_info = (
                f"{idx + 1}."
                f"\tAPI Name: {tool['tool_name']}"
                f"\tDocstring: {tool['docstring']}"
                f"\tUsage Note: {tool['note']}"
                f"\tUsage Example: {tool['demo'][0]}"
            )
            api_info.append(tool_info)
            
        return '\n'.join(api_info)
    
    def get_demo(self, query, retrieval_tools):
        """
        Generate examples and decision examples based on query and retrieval tools.
        Args:
            query (str): The input query.
            retrieval_tools (list): List of tools used for retrieving tasks.
        Returns:
            tuple: A tuple containing two strings - examples and decision_examples.
        """
        filter_top_k_tasks = []
        examples =[]
        retrieval_tools_names = [t["name"] for t in retrieval_tools]

        
        for tool in retrieval_tools:
            top_tasks = self.retrieve_task(query, tool_name=tool["name"])
            for task, score in top_tasks[:1]:
                if self.mode == "pipeline":
                    tool_used = task.get("tool_used", [])
                    # Check if all required tools exist in ToolNet
                    if not any(t in retrieval_tools_names for t in tool_used):
                        continue
                    question = task["question"]
                    example = f"Query: {question}\n" 
                    code = '\n'.join(line for line in task["code"].splitlines() if line.strip())
                    example += f"\nProgram:\n```python\n{code}\n```\n"
                    examples.append((example, score))
                    filter_top_k_tasks.append((task, score))
                else:
                    question = task["question"]
                    if self.task_type.lower() == "date":
                        code = "# import relevant packages\nfrom datetime import date, time, datetime\nfrom dateutil.relativedelta import relativedelta\n" + "\n".join(line.strip() for line in task["code"].split('\n') if line.strip())
                    else:
                        code = "\n".join(line.strip() for line in task["code"].split('\n') if line.strip())
                    if self.task_type.lower() == "tabmwp":
                        solution = question
                        solution += f"\n### Solution code\n```python\n{code}\n```\n"
                    else:
                        solution = f"\nQuery: {question}\n"
                        solution += "Program:\n```python\n" + code + "\n```\n"
                    examples.append((solution, score))
        
        decision_examples = []
        if self.mode == "pipeline":
            for task, score in filter_top_k_tasks:
                tool_create_code = '\n'.join(
                    self.ToolNet.tool_name_to_nodes[t].code
                    for t in task["tool_create"] if t in self.ToolNet.tool_name_to_nodes
                ) if task["tool_create"] \
                else '\n'.join(
                    self.ToolNet.tool_name_to_nodes[t].code
                    for t in task["tool_used"] if t in self.ToolNet.tool_name_to_nodes.keys() and t not in self.basic_tools
                )
                code = '\n'.join(line.strip() for line in task["code"].split('\n') if line.strip())
                example = (
                    f"### Question\n{task['question']}\n"
                    f"### Tools\n```python\n{tool_create_code}\n```\n"
                    f"### Solution Code\n```python\n{code}\n```\n"
                )
                decision_examples.append((example, score))
        header = '### Tools Usgae Examples\nHere are concise examples of how to use the tools in "Tools That Might Help":' if self.mode == "pipeline" \
            else ""
        examples = sorted(examples, key=lambda example: example[-1], reverse=True)
        demos = header + "\n".join(example for example, _ in examples)
        decision_examples = sorted(decision_examples, key=lambda example: example[-1], reverse=True)
        return demos, "\n".join(example for example, _ in decision_examples)
        
    def retrieve(self, query: str, retrieve_sys_prompt: str):
       
        retrieval_tools = self.retrieve_tool(query, retrieve_sys_prompt)
            
        tool_mode_method = getattr(self, self.tool_mode, None)
        if tool_mode_method is None:
            raise AttributeError(f"Method '{self.tool_mode}' not found in {self.__class__.__name__}.")
        # Get Tools Info
        tools_info = tool_mode_method(retrieval_tools)
    
        # Check if demo generation is enabled and retrieve examples accordingly
        examples, decision_examples = self.get_demo(query, retrieval_tools) if self.has_demo else ("", "")
        
        return retrieval_tools, tools_info, examples, decision_examples
    
    
    def generate_code(self, task: str, prompt_path: str, retrieval_msg: str, examples: str, is_api: bool = False, retrieval_tools = []):
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"{prompt_path} does not exist")
        with open(prompt_path, 'r') as f:
            prompt = f.read()
        prompt = prompt.replace("===api===", retrieval_msg)
        prompt = prompt.replace("===example===", examples)

        if self.task_type.lower() == "math":
            sys_prompt = "You are a helpful assistant in answering math competition problems."
        elif self.task_type.lower() == "date":
            sys_prompt = "You are a helpful assistant in answering date reasoning problems."
        elif self.task_type.lower() == "tabmwp":
            sys_prompt = "You are a helpful assistant in answering table reasoning"
        
        obs = prompt.replace("===task===", task)
        
        
        messages = [
            self.encoded_message(sys_prompt, role="system")
        ]
        
        answer_code, attempt= "", 0
        apis = [tool for tool in self.basic_tools] + ([tool["name"] for tool in retrieval_tools] if not self.primitive else [])

        for attempt in range(self.retry_times + 1):
            # Log the current retry attempt and observation
            logging.info(f"Retry {attempt + 1} - Observation: {obs}")
            messages.append(self.encoded_message(obs, role="user"))
            # Get the assistant's response and log it
            response = self.get_response(messages)
            logging.info(f"Retry {attempt + 1} - Response: {response}")
            messages.append(self.encoded_message(response, role="assistant"))
            # Process the response to extract code and status
            code, status = process_code(response, is_api, mode=self.mode)
            if status:  # Successful processing
                answer_code = code
                break
            else:
                obs=code
       
        
        return (answer_code, True) if answer_code else ("Failed to generate solution code", False)

        
    def debug_code(self, task: str, exec_result: str, response: str, sys_prompt_path: str):
        """
        Debugs the provided code by iteratively processing, executing, and rectifying it
        based on a system prompt template.

        Args:
            task (str): The task description.
            exec_result (str): Initial execution result or error message.
            response (str): Initial response containing code to debug.
            sys_prompt_path (str): Path to the system prompt template.

        Returns:
            str: The final processed and debugged code.
        """
        if not os.path.exists(sys_prompt_path):
            raise FileNotFoundError(f"{sys_prompt_path} does not exist")

        # Load the system prompt template
        with open(sys_prompt_path, 'r') as f:
            sys_prompt_template = f.read()
        retry_count = 0
        messages = []
        for attempt in range(1, self.debug_times + 1):
            # Process the response to extract code and status
            processed_code, is_valid_code = process_code(response, is_api=True, mode=self.mode)
            # Retry processing if the code is invalid and retry limit is not reached
            if not is_valid_code and retry_count < self.retry_times:
                logging.info(f"Processing failed. Retrying... Attempt {retry_count + 1}")
                messages.append(self.encoded_message(response, "assistant"))
                messages.append(self.encoded_message(processed_code, "user"))
                response = self.get_response(messages)
                retry_count += 1
                continue
            # Attempt to execute the processed code
            success, exec_result = execute_code(processed_code, os.path.join(self.toolkit_path, self.code_exec))
            if success:
                logging.info("~~~ Execution succeeded ~~~")
                return processed_code  # Exit and return the successful code
            else:
                logging.info(f"!!! Execution failed on attempt {attempt} !!!")
                logging.info(f"Execution Result: {exec_result}")
                # Generate rectification prompt based on the system template
                rectify_prompt = (
                    sys_prompt_template
                    .replace("===task===", task)
                    .replace("===ori===", processed_code)
                    .replace("===err===", exec_result)
                )
                messages = [self.encoded_message(rectify_prompt, "user")]
                response = self.get_response(messages)

        logging.warning("~~~ Debugging process completed without success ~~~")
        return processed_code
            
    def run(self, task: str, **kwargs):
        """
        Main function to run the task using the defined pipeline or standard mode.

        Args:
            task (str): The task to execute.
            max_retry (int): Maximum retry attempts for generating or executing code.
            **kwargs: Optional arguments for customizing system prompts.

        Returns:
            tuple: A tuple containing execution result and success status (result, True/False).
        """
        # Retrieve system prompt paths from kwargs or use default
        creating_sys_prompt = kwargs.get(
            "creating_sys_prompt", 
            f"experiment/{self.task_type}/prompt_lib/prompt_CREATOR_creation_with_retrieval.txt"
        )
        solving_sys_prompt = kwargs.get(
            "solving_sys_prompt", 
            f"experiment/{self.task_type}/prompt_lib/prompt_CREATOR_decision.txt" if self.mode == "pipeline" \
                else f"experiment/{self.task_type}/prompt_lib/prompt_solving.txt" 
        )
        retrieval_sys_prompt =  kwargs.get(
            "retrieval_sys_prompt", 
            f"experiment/{self.task_type}/prompt_lib/prompt_retrieval.txt"
        )
        rec_sys_prompt =  kwargs.get(
            "rec_sys_prompt", 
            f"experiment/{self.task_type}/prompt_lib/prompt_rectification_{self.mode}.txt"
        )
        # Initialize variables
        retrieval_tools, tool_results, examples, decision_examples = [], "", "", ""
        # Retrieve APIs if not in primitive mode
        if not self.primitive:
            logging.info("Retrieving APIs...")
            retrieval_tools, tool_results, examples, decision_examples = self.retrieve(task, retrieval_sys_prompt)
        if self.mode == "pipeline":
            logging.info("Generating tools in pipeline mode...")
            # Generate tool code
            tool_code, status = self.generate_code(
                task, creating_sys_prompt, tool_results, examples, is_api=False, retrieval_tools=retrieval_tools
            )
            if not tool_code or not status:
                return retrieval_tools, "", "", False
            logging.info("Generating decision code...")
            # Generate decision code
            code, status = self.generate_code(
                task, solving_sys_prompt, tool_code, decision_examples, is_api=True, retrieval_tools=retrieval_tools
            )
            answer_code = f"{tool_code}\n\n{code}"
            logging.info("Executing generated code...")
            # Execute generated code
            exec_status, exec_result = execute_code(answer_code, os.path.join(self.toolkit_path, self.code_exec))
            logging.info(f"Execution Result: {exec_result}")
            if not exec_status:
                # Debug the code if execution fails
                answer_code = (
                    self.debug_code(task, exec_result, "```python\n"  + answer_code + "\n```", rec_sys_prompt) 
                    if self.debug_times != 0 else answer_code
                )
                exec_status, exec_result = execute_code(answer_code, os.path.join(self.toolkit_path, self.code_exec))
                logging.info(f"Execution Result: {exec_result}")
        else:  # Standard mode
            logging.info("Executing in standard mode...")
            # Generate tool code
            answer_code, status = self.generate_code(
                task, solving_sys_prompt, tool_results, examples, is_api=True, retrieval_tools=retrieval_tools
            )
            if not answer_code or not status:
                return retrieval_tools, "", "", False
            logging.info("Executing generated code...")
            # Execute generated code
            exec_status, exec_result = execute_code(answer_code, os.path.join(self.toolkit_path, self.code_exec))
            logging.info(f"Execution Result: {exec_result}")
            if not exec_status:
                # Debug the code if execution fails
                answer_code = (
                    self.debug_code(task, exec_result, "```python\n"  + answer_code + "\n```", rec_sys_prompt)
                    if self.debug_times != 0 else answer_code
                )
                exec_status, exec_result = execute_code(answer_code, os.path.join(self.toolkit_path, self.code_exec))
                logging.info(f"Execution Result: {exec_result}")
                
        return (retrieval_tools, answer_code, exec_result, True) if exec_status else (retrieval_tools, answer_code, "", False)
         
    def check_is_answer_right(self, task: str, gt_answer, pre_answer):
        if self.task_type.lower() == "math" or self.task_type.lower() == "tabmwp":
            if "Final Answer:" in pre_answer:
                model_ans = [pre_answer.split("Final Answer:")[1].strip()]
            else:
                logging.info("Getting Answer by Directly Extracting Number ...")
                model_ans = re.findall(r'-?\d+\.?\d*', pre_answer)
            correct_flag = False
            try:
                for ans in model_ans:
                    if grade_answer(str(ans), str(gt_answer)) or \
                    abs( (eval(str(ans)) - eval(str(gt_answer))) / eval(str(gt_answer)) ) < 0.01 or \
                    round(eval(str(ans)), 2) == round(eval(str(gt_answer)), 2):
                        logging.info("~~~ Correct Answer ~~~")
                        correct_flag = True
                        break
            except:
                correct_flag = False
        elif self.task_type.lower() == "date":

            try:
                date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
                ref_dates = list(set(re.findall(date_pattern, gt_answer)))[0]
                pred_dates = list(set(re.findall(date_pattern, pre_answer)))[0]
                correct_flag = True if ref_dates == pred_dates else False
            except:
                correct_flag = False
                
        if not correct_flag:
            logging.info("~~~ Wrong Answer ~~~")
            
        return correct_flag
        
    
    def test(self, tasks: Union[Dict,List[Dict]], output_path: str, **kwargs):
        if not output_path.endswith(".jsonl"):
            raise ValueError("Output file should be JSONL format.")
        if not os.path.exists(output_path):
            open(output_path, 'w').close()
        tasks = tasks if isinstance(tasks, list) else [tasks]
        for task in tasks:
            result = {}
            result.update(task)
            question = task["question"]
            logging.info(f"Question: {question}")
            std_answer = task["answer"]
            result.pop("answer", "")
            result["retrieved_api"] = []
            
            retrieval_tools, answer_code, llm_answer, status = self.run(question, **kwargs)
            for tool in retrieval_tools:
                result["retrieved_api"].append(tool["name"])
            result["code"] = answer_code
            result["used_tool"] = []
            result["std_answer"] = task["answer"]
            result["llm_answer"] = "" if not status else llm_answer
            for api in retrieval_tools:
                if ' ' + api["name"] + '(' in answer_code:
                    result["used_tool"].append(api["name"])
            logging.info(f"API used in Final Code: {result['used_tool']}")     
            if status:
                logging.info("Check Answer")
                is_right = self.check_is_answer_right(question, std_answer, llm_answer)
                result["success"] = is_right
            else:
                result["success"] = False
            
            with jsonlines.open(output_path, 'a') as f:
                f.write(result)
            
                
            
        
        
        
                
        
        
        
        
        
        
        
        
        
        
            




    







                