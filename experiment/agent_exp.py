from pydantic import BaseModel, Field, ValidationError, field_validator
from collections import deque
import numpy as np
from typing import Literal, Union, Dict, List, Optional
from toolgraph.net_sturcture import SkillManager
from toolgraph.embedding_model import OpenAIEmbeddingModel
from agent import *
import logging, jsonlines, copy
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from .utils import *
from grader import *

class TestingFrameworkConfig(BaseModel):
    agent_cfg: Union[str, Dict] = Field(..., description="Configuration for the LLM.")
    toolkit_path: str = Field(..., description="Path to the toolkit directory.")
    task_type: str = Field(..., description="Type of task.")
    exp_name: str = Field(..., description="experiment Name")
    train_output: str = Field(..., description="Path to the training output file.")
    train_embed_path: str = Field(..., description="Path to store training query's embedding file.")
    top_k_query: int = Field(3, ge=0, description="Number of top queries to retrieve.")
    top_k_tool: int = Field(3,ge=0, description="Number of top tools to retrieve.")
    test_mode: Literal["pipeline", "standard"] = Field(
        "standard", description="Testing mode: 'pipline' or 'standard'."
    )
    output_dir: str=Field(...,description="Folder path to store notebook")
    basic_tools: List[str] = Field(default_factory=list, description="List of basic tools.")
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
    
    @field_validator("agent_cfg")
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
        self.agent = Agent(AgentConfig(config.agent_cfg))
        self.llm = OpenAILLM(config.agent_cfg.get("llm_config", {}))
        self.train_output = config.train_output
        self.toolkit_path = config.toolkit_path
        self.train_embed_path = config.train_embed_path
        self.tool_mode = config.tool_mode
        self.output_dir = config.output_dir
        self.top_k_query = config.top_k_query
        self.top_k_tool = config.top_k_tool
        self.mode = config.test_mode
        self.basic_tools = config.basic_tools
        self.task_type = config.task_type
        self.embedding_model = OpenAIEmbeddingModel()
        self.tmp_dir =  os.path.join(os.getcwd(), f"mnt_{self.agent.llm.model.split('/')[-1]}_{self.task_type}_{config.exp_name}")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.env = Env(self.toolkit_path, basic_tools=self.basic_tools, task_type=self.task_type, tmp_dir=self.tmp_dir)
        self.primitive = config.is_primitive
        self.has_demo = config.has_demo

        # Initialize ToolFlowNet if not in primitive mode
        if not self.primitive:
            self.ToolNet = SkillManager(
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
    
    def get_response(self, messages: List):
        assert len(messages) > 0, "Messages to LLM must be a List whose length is more than 0."
        temperature = self.llm.temperature
        for _ in range(3):
            self.llm.temperature = 0
            status, response = self.llm.get_response(messages)
            if not status:
                if response.lower() in ["context_length_exceeded","rate_limit_exceeded","max_tokens"]:
                    messages.pop(-1)
                    messages = [messages[0]] + messages[2:]
                else:
                    continue
            else:
                break
        self.llm.temperature = temperature
        return response

    def retrieve_tool(self, query: str, sys_prompt: str):
        logging.info("Tool Retrieval Begin!!")
 
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
        for attempt in range(5):
            logging.info(f"Attempt {attempt + 1} Observation: {obs}")
            response = self.get_response(messages)
            logging.info(f"Attempt {attempt + 1} Response: {response}")
            retrieve_query = parse_retrieval_msg(response)
            if retrieve_query is None:
                obs = "You should provide a valid retrieval query, following the format 'Retrieve_api(api_name=API Nanme)```docstring```'. Please check again"
                messages.append(self.encoded_message(response, role="assistant"))
                messages.append(self.encoded_message(obs, role="user"))
                continue
            else:
                break
    
        if retrieve_query:
            retrieved_tools = self.ToolNet.retrieve_skills(retrieve_query["docstring"])
        else:
            logging.error("Failed to Retrieve Tools")
            raise RuntimeError(f"Failed to Retrieve Tools")

    
        
        return retrieved_tools
            
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
        tools_info = []
        for tool in tools_result:
            info = f"### Tool `{tool['name']}`\n"
            info += f"Source Code:\n```python\n{tool['code']}\n```\n"
           
            tools_info.append(info)

        tools_result = "\n".join(tools_info)
        return tools_result

    def api(self, retrieval_tools):
        tools_result = copy.deepcopy(retrieval_tools)
        tools_result = sorted(tools_result, key=lambda t: (t["level"], -t["freq"]))
        api_info = []
        for idx, tool in enumerate(tools_result):
            tool_info = (
                f"### Tool `{tool['name']}`\n"
                f"Docstring: {tool['docstring']}\n"
                f"Usage Note: {tool['note']}\n"
                f"Usage Example: {tool['demo'][0]}\n"
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
                    example = f"#### Question\n{question}\n" 
                    code = '\n'.join(line for line in task["code"].splitlines() if line.strip())
                    example += f"\n#### Solution\n```python\n{code}\n```\n"
                    examples.append((example, score))
                    filter_top_k_tasks.append((task, score))
                else:
                    question = "Example Goal: " + task["query"] if self.task_type == "textcraft" else task["question"]
                    code = f"### Solution\n```python\n{task['code']}\n```"
                    solution = f"### Task\n{question}\n"
                    solution += code
                    examples.append((solution, score))
        
        header = '# Tool Usage Examples #\nHere are concise examples demonstrating how to use the tools in the “Custom Library.” These combined notebook blocks illustrate when and how to call the appropriate tools, serving as a guide rather than a full code generation reference.\n' if self.task_type.lower() == "textcraft" \
            else '## Tool Usage Examples\nHere are concise examples demonstrating how to use the tools in the "Tools that Might Help".\n'
        examples = sorted(examples, key=lambda example: example[-1], reverse=True)
        demos = "\n".join(example for example, _ in examples)
        demos = header + demos
       
        return demos, ""
        
    def retrieve(self, query: str, retrieve_sys_prompt: str):
        
        retrieval_tools = self.retrieve_tool(query, retrieve_sys_prompt)

        tool_mode_method = getattr(self, self.tool_mode, None)
        if tool_mode_method is None:
            raise AttributeError(f"Method '{self.tool_mode}' not found in {self.__class__.__name__}.")
        # Get Tools Info
        tools_info = tool_mode_method(retrieval_tools)
        examples = []
        
        examples, decision_examples = self.get_demo(query, retrieval_tools) if self.has_demo else ("", "")
        examples = examples if self.has_demo else ""
        
        logging.info(f"Retrieval Tools:\n {tools_info}")

        return retrieval_tools, tools_info, examples, ""
    
    def solving_loop(self, task, obs, **kwargs):

        solving_steps = kwargs.get("solving_steps", 20)
        self.agent.memory.clear_memory()
        
        done = False
        result = ""
        step_idx, retry_count = 0, 0
        last_action = None
        repeat_action = 0
        finished = False
        logging.info("Sovling Loop Begin")
        failed_action = 0
        while step_idx < solving_steps:
            
            logging.info(f"Solving Loop: Step {step_idx + 1} SolvingAgent: {obs}")
           
            res, action = self.agent.generate(task, obs)
            logging.info(f"Solving Loop: Step {step_idx + 1} SolvingAgent response: {res}")
            if action is None:
                logging.error("Failed to parse action from response.")
                if retry_count >= 3:
                    logging.error(f"Failed to extract the action {retry_count} consecutive times. Stop...")
                    return None, False
                retry_count += 1
                obs = "Failed to parse action from your response, please make sure you provide a valid action."
                continue
            else:
                if last_action is not None and last_action == action:
                    if repeat_action >= 3:
                        logging.error(f"Error: Took the same action {repeat_action} consecutive times.")
                        return None, False
                    obs = "Observation: The action is the same as the last one, please provide a different action."
                    repeat_action += 1
                    continue
                else:
                    # Execute Action
                    obs, done = self.step(action)
                    obs = f"Observation: {obs}"
                    last_action = action
                    repeat_action = 0
            if done:
                failed_action = 0
                if isinstance(action, Terminate):
                    result = action.argument["result"]
                    logging.info("The task is done.")
                    finished = True
                    break
            else:
                failed_action += 1
                if failed_action >= 3:
                    return None, False
                continue
            step_idx += 1
            
        return result, finished

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
        solving_sys_prompt = kwargs.get(
            "solving_sys_prompt", 
            f"experiment/{self.task_type}/prompt_lib/prompt_CREATOR_decision.txt" if self.mode == "pipeline" \
                else f"experiment/{self.task_type}/prompt_lib/prompt_solving.txt" 
        )

        retrieval_sys_prompt =  kwargs.get(
            "retrieval_sys_prompt", 
            f"experiment/{self.task_type}/prompt_lib/prompt_retrieval.txt"
        )
        
        
        # Initialize variables
        retrieval_tools, tool_results, examples, decision_examples = [], "", "", ""
        # Retrieve APIs if not in primitive mode
        if not self.primitive:
            logging.info("Retrieving APIs...")
            retrieval_tools, tool_results, examples, decision_examples = self.retrieve(task, retrieval_sys_prompt)
        if self.mode == "pipeline":
            logging.info("Runing Task in pipeline mode...")
            # Generate tool code
            pass
        else:  # Standard mode
            logging.info("Executing in standard mode...")
            with open(solving_sys_prompt, 'r') as f:
                sys_prompt = f.read()
            sys_prompt = sys_prompt.replace("===api===", tool_results)
            sys_prompt = sys_prompt.replace("===example===", examples)
            sys_prompt = sys_prompt.replace("===task===", task)
            self.agent.system_prompt = sys_prompt
            logging.info(f"System Prompt:\n{sys_prompt}")
            # Generate tool code
            solving_obs = "Let's think step by step. Please start by creating a plan based on the task."
            result, finished = self.solving_loop(task, solving_obs)
            if not finished:
                return retrieval_tools, ""
            
        return (retrieval_tools, result)

    def check_is_answer_right(self, task: str, gt_answer, pre_answer):
       self.env.notebook.add_block("check_inventory()")
       result, status = self.env.notebook.execute_last_block()
       logging.info(f"Checking Environment... : {result}")
       self.env.notebook.delete_block(len(self.env.notebook.notebook.cells) - 1)
       if not status:
           return False
       else:
           if f'[{gt_answer}]' in result:
               return True
    
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

            retrieval_tools, llm_answer = self.run(question, **kwargs)
            for tool in retrieval_tools:
                result["retrieved_api"].append(tool["name"])
                
            result["std_answer"] = task["answer"]
            result["llm_answer"] = llm_answer
            
            is_right = self.check_is_answer_right(task,gt_answer=std_answer, pre_answer="")
            result["success"] = True if is_right else False
            
            result_file = os.path.join(self.output_dir, f"{result['id']}.ipynb")
            if is_right:
                logging.info("~~~~~Correct Answer~~~~~")
            else:
                logging.info("~~~~~Wrong Answer~~~~~")
            self.env.notebook.save_notebook(result_file)
            self.env.clear_env()
            self.agent.clear()
            with jsonlines.open(output_path, "a") as f:
                f.write(result)
    
    def get_plan(self, task: str, sys_prompt: str):
        with open(sys_prompt, 'r') as f:
            sys_prompt = f.read()
        sys_prompt = sys_prompt.replace("===task===", task)
        
        messages = [self.encoded_message(sys_prompt, "user")]
        response = self.get_response(messages)
        
        return response    
            
    def plan_to_args(self, plan, keyword = 'Step', lkey = 'execution order'):
        
        def fetch_args(args_lookup, logic_exp):
            out = copy.deepcopy(logic_exp)
            assert 'steps' in logic_exp.keys()
            for s, step in enumerate(logic_exp['steps']):
                if isinstance(step, int):
                    out['steps'][s] = args_lookup[step]
                elif isinstance(step, dict):
                    out['steps'][s] = fetch_args(args_lookup, step)
            return out
        def parse_expression(expression):
            stack = []
            current = {}
            for token in re.findall(r'Step \d+|AND|OR|\(|\)', expression):
                if token.startswith('Step'):
                    if 'steps' not in current:
                        current['steps'] = []
                    current['steps'].append(int(token.split()[1]))
                elif token in ('AND', 'OR'):
                    current['logic'] = token
                elif token == '(':
                    stack.append(current)
                    current = {}
                elif token == ')':
                    closed = current
                    current = stack.pop()
                    if 'steps' not in current:
                        current['steps'] = []
                    current['steps'].append(closed)
            return current
        
        args = []
        lines = plan.split('\n')
        for line in lines:
            if line.startswith(keyword): args.append(re.sub(r'{} \d+: '.format(keyword), '', line))
            if lkey in line.lower():
                logic = line.split(': ')[-1]
        args_lookup = {i+1: args[i] for i in range(len(args))}
        try:
            return fetch_args(args_lookup, parse_expression(logic))
        except: 
            return {'steps': args, 'logic': 'AND'}

    def plan_and_run(self, task, sys_prompt):
        plan = self.get_plan(sys_prompt)
        plan_steps = self.plan_to_args(plan)
        
        if len(plan_steps['steps']) == 1: 
            plan_steps = plan_steps['steps'][0]
            if type(plan_steps) == str: plan_steps={'steps':[plan_steps]}
            if 'logic' not in plan_steps.keys():
                try:
                    logic = plan_steps['logic']
                except: logic = "AND"; plan_steps['logic'] = logic
        for sub_task in plan_steps['steps']:
            result, succ = self.solving_loop(task, obs="", solving_steps = 8)
            if plan_steps['logic'].lower() == 'or':
                if succ:
                    return result, True
            # If reached here you have succeeded.
            if succ:
                return result, True

    
    def step(self, action: Action, **kwargs):
        observation, done = self.env.step(action, agent_name="SolvingAgent")
        observation = self._handle_observation(observation)
        return observation, done
    
    def _handle_observation(self, observation):
        max_length = 2000
        if isinstance(observation, float) or isinstance(observation, int) or isinstance(observation, list):
            return observation
        if len(observation.split(' ')) > max_length:
            truncated_observation = '\n'.join(observation.split('\n')[:max_length]) + "\n[Observation too long, truncated; Try other commands to get the left part.]"
            return truncated_observation
        return observation
            
            
            
