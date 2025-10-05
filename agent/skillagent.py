import logging
import os
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from toolgraph import SkillManager
from .agents import Agent
from .config import MultiAgentsConfig
from typing import List, Dict, Union
from .utils import *
from collections import defaultdict
from .action import *
import jsonlines, json
from .llm import OpenAILLM
from .environment.env import Env

DEFAULT_TIME_OUT = 120
MAX_OBS_LENGTH = 3000
MAX_STEPS = 35
CURRENT_WORKING_DICTIONARY = os.getcwd()


class SkillAgent:
    
    def __init__(self, cfg_path: Union[str, Dict], toolkit_path: str, task_type: str, basic_tools=[], resume: bool=False):
        self.config = MultiAgentsConfig(cfg_path)
        self.max_steps = MAX_STEPS
        self.task_type = task_type
        self.tmp_dir = os.path.join(os.getcwd(), f"mnt_{self.task_type}")
        self.env = Env(toolkit_path=toolkit_path, basic_tools=basic_tools, tmp_dir=self.tmp_dir, task_type=task_type)
        self.toolnet = SkillManager(basic_tools=basic_tools, ckpt_dir=toolkit_path, resume=resume)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.toolkit_path = toolkit_path
        self._load_cfg_to_agent()
            
    def _load_cfg_to_agent(self):
        
        for config in self.config.config:
            if config.agent_name == "SolvingAgent":
                self.solving_agent = Agent(config)
            elif config.agent_name == "ToolAgent":
                self.tool_agent = Agent(config)
            elif config.agent_name == "CheckStage1Agent":
                self.check_agent_stage1 = Agent(config)
            elif config.agent_name == "CheckStage2Agent":
                self.check_agent_stage2 = Agent(config)
            elif config.agent_name == "RetrievalAgent":
                self.retrieval_agent = Agent(config)
                
        self.main_agent = self.solving_agent   
        
    def check_is_answer_right(self, task: str, gt_answer, pre_answer):
        if self.task_type.lower() == "textcraft":
            is_right = False
            self.env.notebook.add_block("check_inventory()")
            result, status = self.env.notebook.execute_last_block()
            logging.info(f"Checking Environment... : {result}")
            self.env.notebook.delete_block(len(self.env.notebook.notebook.cells) - 1)
            if not status:
                return False
            else:
                if f'[{gt_answer}]' in result:
                    return True
        elif self.task_type.lower() == "dabench":
            label_answers = {ans[0]: ans[1] for ans in gt_answer}
            pred_answer_names, pred_answers = extract_format(pre_answer)
            extracted_answers = {key.lower().strip(): value for key, value in zip(pred_answer_names, pred_answers)}
            gt_answer_lower = {
                    key.lower().strip(): (
                        [v.strip("'\" ").strip() for v in value.split(',')]
                        if ',' in value else value.strip("'\" ").strip() 
                    ) if isinstance(value, str) else value
                    for key, value in label_answers.items()
                }
            correct_answers = {}
            for ans_name in gt_answer_lower.keys():
                pred_value = extracted_answers.get(ans_name)
                gt_value = gt_answer_lower[ans_name]                
                # Check if both values are lists
                if isinstance(pred_value, list) and isinstance(gt_value, list):
                    is_matching = all(
                        any(is_equal(gt_item, pred_item) for pred_item in pred_value) for gt_item in gt_value
                    )
                    correct_answers[ans_name] = is_matching
                else:
                    correct_answers[ans_name] = is_equal(pred_value, gt_value)
            is_right =  all(correct_answers.values()) if correct_answers else False
        elif self.task_type.lower() in ["math", "date", "tabmwp"]:
            llm = OpenAILLM({
                "model": "gpt-4o",
                "temperature": 0.0,
                "max_tokens": 200,
                "top_p": 0.9})
            pre_answer = pre_answer if pre_answer else "Fails to generate answer."
            with open(r'Datasets/MATH/prompt_lib/answerchecker.txt', 'r') as f:
                system_prompt = f.read()
            message = [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": system_prompt
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"Standard Answer: {gt_answer}\nGiven Answer:{pre_answer}" 
                    }]
                }
            ]
            results = []
            while len(results) < 3:
                _, result = llm.get_response(messages=message)
                logging.info(f"Answer Checking {len(results) + 1} times: {result}")
                patterns = [ r'["\']?Response["\']?:? (.*?)Response', 
                    r'["\']?Response["\']?:? (.*?)Thought', r'["\']?Response["\']?:? (.*?)$']
                response = ""
                for pattern in patterns:
                    match = re.search(pattern, result, flags=re.DOTALL)
                    if match:
                        response = match.group(1).strip()
                        break
                if not response: 
                    continue
                if 'true' in response.lower():
                    results.append(True)
                elif 'false' in response.lower():
                    results.append(False)
            is_right = all(results)
        
        return is_right
                    
    def _post_run(self):
        tool_created = self.env.extract_all_tool_created()
        tool_used = self.env.extract_all_tool_used()

        if not tool_created and not tool_used:
            return [], []

        dependency_relation = self.env.extract_tool_dependency()
        
        for tool in tool_used:
            if tool in self.toolnet.skills:
                self.toolnet.skills[tool]["freq"] += 1

        logging.info(f"Dependency relations in this task: {dependency_relation}")

        tool_to_level = defaultdict(lambda: 2)
        
        def compute_and_assign_levels(tool):
            if tool not in dependency_relation or not dependency_relation[tool]:
                return tool_to_level[tool]
            
            tool_to_level[tool] = max(
                compute_and_assign_levels(dep[0]) for dep in dependency_relation[tool]
            ) + 1
            return tool_to_level[tool]

        for tool in tool_created:
            compute_and_assign_levels(tool)

        for tool in tool_created:
            if tool in self.env.temp_toolkit:
                tool_info = self.env.temp_toolkit[tool]
                self.toolnet.add_new_skill(
                    {
                        "name": tool,
                        "code": tool_info["code"],
                        "docstring": tool_info["docstring"],
                        "demo": tool_info["demo"],
                        "level": tool_to_level[tool],
                    },
                    duplicate_tool=self.env.duplication_tool,
                    neighbor_indices=dependency_relation.get(tool, []),
                )

        id2skills = {self.toolnet.skills[node]["id"]: node for node in self.toolnet.skills.keys()}

        def update_subnode_frequency(node_index):
            current_node = id2skills[node_index]
            for idx, weight in enumerate(self.toolnet.skills_matrix[node_index, :]):
                if not np.isinf(weight) and idx in id2skills:
                    sub_node = id2skills[idx]
                    if self.toolnet.skills[sub_node]["level"] < self.toolnet.skills[current_node]["level"]:
                        self.toolnet.skills[sub_node]["freq"] += 1  
                        update_subnode_frequency(idx)

        for tool in tool_used:
            if tool in self.toolnet.skills:
                self.toolnet.skills[tool]["freq"] += 1
                if self.toolnet.skills[tool]["level"] > 2:
                    update_subnode_frequency(self.toolnet.skills[tool]["id"])

        return tool_created, list(tool_used)
    
    def solving_loop(self, task, observation, **kwargs):
        solving_steps = kwargs.get("solving_steps", 10)
        max_tool_requests = kwargs.get("max_tool_request", 5)
        
        step_index = 0
        tool_request_count = 0
        retry_count = 0
        repeated_action_count = 0
        failed_action_count = 0
        last_action = None
        task_finished = False
        result = ""

        logging.info("Solving Loop Started")

        while step_index < solving_steps:
            logging.info(f"Step {step_index + 1}: Observing input: {observation}")
            response, action = self.solving_agent.generate(task, observation)
            logging.info(f"Step {step_index + 1}: Agent response: {response}")

            if action is None:
                retry_count += 1
                if retry_count >= 3:
                    logging.warning("Exceeded maximum retries for parsing action. Exiting...")
                    return None, False
                observation = "Failed to parse action from your response. Please ensure you provide a valid action."
                continue

            if action == last_action:
                repeated_action_count += 1
                if repeated_action_count >= 3:
                    logging.warning(f"Action repeated {repeated_action_count} consecutive times. Exiting...")
                    return None, False
                observation = "The action is the same as the last one. Please provide a different action or argument."
                continue
            else:
                repeated_action_count = 0  # 重置重复计数

            if isinstance(action, ToolRequest):
                if tool_request_count >= max_tool_requests:
                    logging.info("Maximum tool requests reached. Forcing task resolution without further tools.")
                    observation = "You have reached the maximum number of tool requests. Please resolve the issue on your own."
                else:
                    tool_request_count += 1

            last_action = action
            observation, is_task_completed = self.step(action)

            if is_task_completed:
                failed_action_count = 0
                if isinstance(action, ToolRequest):
                    tool_observation, tool_completed = self.tool_loop(task, observation)
                    if tool_completed:
                        tool_observation_text = f"\n{tool_observation}" if tool_observation else ""
                        observation = "You received APIs from ToolAgent and CheckingAgent. Use them to solve the task." + tool_observation_text

                if isinstance(action, Terminate):
                    result = action.argument["result"]
                    logging.info("Task completed successfully.")
                    task_finished = True
                    break
            else:
                failed_action_count += 1
                if failed_action_count >= 3:
                    logging.warning("Failed action limit reached. Exiting...")
                    return None, False

            step_index += 1

        return result, task_finished
    
    def retrieval(self, task, query):
        obs = f"Below is the tool request from the SolvingAgent. Please generate a tool retrieval message using the Retrieval Tool Action, considering the provided tool request and task requirements:\n{query}"    
        logging.info("============== Retrieval Stage =================")
        for attempt in range(3):
            response, action =  self.retrieval_agent.generate(task, obs, add_history=False)
            logging.info(f"Attempt {attempt + 1} Response: {response}")
            if action is None:
                logging.error("Failed to parse action from response.")
            elif not isinstance(action, Retrieval):
                logging.error("Failed to parse Retrieval action from response.")
                obs = "Failed to parse Retrieval action from your response. Please make sure you provide a valid Retrieval action."
                continue
            else:
                obs, done = self.step(action)
                if done:
                    logging.info("Retrieval Tool is done.")
                    break
        return obs
    
    def parse_program_json(self, msg ):
        import json
        skills_names = [skill for skill in self.toolnet.skills]
        pattern = r"(?:Tool|Action):\s*(?:\n\s*)*({\s*['\"]tool_name['\"]:\s*['\"][^'\"]+['\"]\s*})"

        parse_str = re.search(pattern, msg)
        parse_str = parse_str.group(1) if parse_str else None
        if not parse_str:
            return None, True
        else:
            try:
                parse_str = parse_str.replace("\t", "").replace("\n", "")
                parse_dict = json.loads(parse_str)
            except Exception as e:
                logging.error(f"JSON parsing failed: {e}")
                return None, True
            
        if "tool_name" in parse_dict.keys():
            tool_name = parse_dict["tool_name"]
            if not tool_name:
                return "Please Provide a valid program name", False
            else:
                if not tool_name in skills_names:
                    return f"{tool_name} not in current High-Level Program, please check again.", False
                return tool_name, True
        return "", True
    
    def create_tool(self, task, request):
        obs = request
    
        tool_code, tool_name = "", ""
        for i in range(5):
            logging.info(f"Observation: {obs}")
            response, action = self.tool_agent.generate(task, obs)
            name, status = self.parse_program_json(response)
            if name and status:
                tool_name = name
                tool_code = self.toolnet.skills[tool_name]["code"]
                break
            logging.info(f"Tool Agent Response: {response}")
            if action is None:
                logging.error("Failed to parse response.")
                continue
            else:
                obs, done = self.step(action)
                if done:
                    tool_code = action.argument["code"]
                    tool_name = action.argument["tool_name"]
            
            if tool_code:
                detect_msg = self.env.detect_tool(tool_code, skills=self.toolnet.skills)
                if detect_msg:
                    logging.info(f"Detect Redundant Tool: {detect_msg}")
                    continue  
                else:
                    break
            else:
                continue
          
        return tool_code, tool_name
        
    def tool_loop(self, task, tool_request: List[str]):
        success_tools = []
        
        for request in tool_request:
            stage1, stage2=False, False
            self.check_agent_stage1.clear()
            self.check_agent_stage2.clear()
            retrieval_result = self.retrieval(task, request)
            obs = f"### Existing Tools\n{retrieval_result}\n### Tool Request\n{request}"
            tool_code = ""
            tool_name = ""
            for i in range(5):
                tool_code, tool_name = self.create_tool(task, obs)
                if not tool_code or not tool_name:
                    continue
                if tool_name in self.toolnet.skills.keys():
                    tool_description = f"### Tool `{tool_name}`\n**Source Code**:\n```python\n{tool_code}\n```"
                    logging.info(f"Directly Use Existing Function: {tool_name}")
                    success_tools.append(tool_description)
                    break
                message = f"**Tool Name: `{tool_name}`**\n**Source Code**:\n```python\n{tool_code}\n```"    
                check_result = self.check_tool(task, retrieval_result, message, stage1, stage2)
                if check_result["pass_all"]:
                    tool_description = self.generate_tool_description(task, message)
                    logging.info(f"Successfully generated tool description:\n{tool_description}")
                    success_tools.append(tool_description)
                    break
                else:
                    logging.info(f"Tool {tool_name} failed self-check: {check_result}")
                    obs = ""
                    if not check_result["reuse"]["passed"]:
                        stage1=False
                        obs += f"Reusability Check Failed:\n {check_result['reuse']['feedback']}"
                    if not check_result["bug_free"]["passed"]:
                        stage1=True
                        stage2=False
                        obs += f"Bug-Free Check Failed:\n {check_result['bug_free']['feedback']}"
                    continue
        
        success_tools = "\n".join(success_tools)
        return (success_tools, True) if success_tools else ("", False)
    
    def generate_tool_description(self, task, message):
        logging.info("======== Generating tool description ==========")
        message = TOOL_DESCRIPTION + "\n" + f"Please wrap the following tool into an API:\n{message}"
        obs=message
        for i in range(3):
            response, action = self.check_agent_stage2.generate(task, obs, add_history=False)
            logging.info(f"Response: {response}")
            if not isinstance(action, SendAPI):
                continue
            else:
                obs, done = self.step(action)
                if done:
                    return obs
                else:
                    message = message + "\n" + obs
        
        return ""
            
        
    def check_tool(self, task, retrieval_result: str, message: Optional[str]=None, stage1=False, stage2=False):
       
        obs = f"### Existing Tools\n{retrieval_result}\n### Tool You need to Check\n{message}"
        check_result = {"reuse": {"passed": False, "feedback": ""}, "bug_free": {"passed": False, "feedback": ""}, "pass_all": False}
        if not stage1:
            logging.info("========= Self-Check Stage 1: Check Reuseablity ==========")
            for i in range(3):
                logging.info(f"Observation: {obs}")
                response, action = self.check_agent_stage1.generate(task, obs)
                logging.info(f"Response: {response}")
        
                if not isinstance(action, Feedback):
                    obs = "Please Review the code, and generate Feedback Action to feedback it."
                    continue
                else:
                    feedback_dict = action.argument
                    if not feedback_dict.get("passed", False):
                        if not feedback_dict.get("feedback", ""):
                            continue
                        else:
                            check_result["reuse"]["feedback"] = feedback_dict.get("feedback", "")
                            break
                    else:
                        check_result["reuse"]["passed"] = True
                        break
        
            if not check_result["reuse"]["passed"]:
                return check_result
        if not stage2:
            obs = f"### Existing Tools\n{retrieval_result}\n### Tool You need to Check\n{message}"   
            logging.info("========= Self-Check Stage 2: Bug-Free ==========")
            for i in range(3):
                feedback_dict = self.check_bug_free(task, obs)
                if not feedback_dict.get("passed", False):
                    if not feedback_dict.get("feedback", ""):
                        continue
                    else:
                        check_result["bug_free"]["feedback"] = feedback_dict["feedback"]
                        break
                else:
                    check_result["bug_free"]["passed"] = True
                    check_result["pass_all"] = True
                    break
                
            return check_result


    def check_bug_free(self, task, message):
        message = "Please use the following tool code to simply check for errors.\n" + message 
        last_action = None
        obs = message
        for i in range(5):
            logging.info(f"Observation: {obs}")
            response, action = self.check_agent_stage2.generate(task, obs)
            logging.info(f"Response: {response}")
            if action is None:
                logging.error("Failed to parse action from response.")
                obs = f"Fail to parse the action from your response. Please provide a valid action from your Action Space."
                continue
            else:
                if last_action is not None and last_action == action:
                    obs = "The action is the same as the last one, please provide a different action or different argument."
                    continue
                else:
                    last_action = action
                    obs, done = self.step(action)
                    if done and isinstance(action, Feedback):
                        
                        return action.argument
                    else:
                        continue
        return {}
              
    def generate(self, agent: Agent, obs: str):
        return agent.generate(obs)
            
    def step(self, action: Action, **kwargs):
        use_api = kwargs.get("use_api", True)
        observation, done = self.env.step(action, agent_name=self.main_agent.agent_name,
            toolnet=self.toolnet, use_api=use_api)

        observation = self._handle_observation(observation)
        return observation, done
    
    def run(self, task: str, **kwargs):
        solving_steps = kwargs.get("solving_steps", 15)
        tool_steps = kwargs.get("tool_steps", 15)
        max_tool_request = kwargs.get("max_tool_request", 6)
        obs = kwargs.get("obs", "")
        is_retry = kwargs.get("is_retry", False)
        is_retry = False
        if not is_retry:
            if self.task_type.lower() == "dabench" or self.task_type.lower() == "textcraft":
                obs = "You are now going to solve the task. Please follow the specified plan for the task and send tool requests to the ToolAgent."
            else:
                obs = "You are going to solve the task now. Let's think step by step. Please come up with a reasonable tool request first.​" 
        else:
            obs = "After comparing with the standard answer, your terminate answer is incorrect. Please reflect carefully to identify the errors in your solution, which may involve task reasoning, code implementation, or tool usage. Please thoroughly analyze the issues, correct the solution and get a correct answer, which should differ from the current one."
        logging.info(f"task: {task}")
        result, finished = self.solving_loop(task, obs, solving_steps=solving_steps, tool_steps=tool_steps, max_tool_request=max_tool_request)
        return result, finished

    def train(self, tasks: Union[List[Dict], Dict], output_path: str, save_dir: str, turn: int):  
        import shutil 
        tasks = tasks if isinstance(tasks, list) else [tasks]
        results = []
        
        os.makedirs(save_dir, exist_ok=True)
        for idx, task in enumerate(tasks):     
            self.main_agent = self.solving_agent
            
            self.clear()
            query = self.load_task(task)
            gt_answer = task.get('answer')
            result, finished = self.run(task=query)
            check = False
            if not finished:
                task["success"] = False
                results.append(task)
            else:
                check = self.check_is_answer_right(task, gt_answer, result) if finished else False
            retry_check = False
            if not check or not finished:
                retry_result, retry_finished = self.retry_fails(task)
                if not retry_finished:
                    task["success"] = False
                    results.append(task)
                else:
                    retry_check = self.check_is_answer_right(task, gt_answer, retry_result)
                    result = retry_result
            if check or retry_check:
                logging.info("The answer is right")
                notebook_result = self.env.notebook.get_all_execution_outputs()[1:]
                code = ""
                for block, _ in notebook_result:
                    block = "\n".join([line for line in block.splitlines() if line.strip()])
                    code += f"{block}\n" 
                    
                for tool in self.env.temp_toolkit.keys():
                    if self.env.temp_toolkit[tool]["created"]:
                        if self.env.temp_toolkit[tool]["code"] in code:
                            code = code.replace(self.env.temp_toolkit[tool]["code"], "")
                task["code"]  = code               
                tool_created, tool_used = self._post_run(turn=idx + turn)

                task["success"] = True
                task["tool_create"] = list(tool_created)
                task["tool_used"] = list(tool_used)
                task["llm_answer"] = result
            else:
                logging.info("The answer is wrong")
                task["success"] = False
                task["llm_answer"] = result
                results.append(task)

            filename = self.task_type + "_" + str(task["id"]).strip()
            self.env.notebook.save_notebook(os.path.join(save_dir, f"{filename}.ipynb"))
            with jsonlines.open(output_path, 'a') as f:
                f.write(task)

    def _handle_observation(self, observation):
        max_length = MAX_OBS_LENGTH  
        if isinstance(observation, float) or isinstance(observation, int) or isinstance(observation, list):
            return observation
        if len(observation.split(' ')) > max_length:
            truncated_observation = '\n'.join(observation.split('\n')[:max_length]) + "\n[Observation too long, truncated; Try other commands to get the left part.]"
            return truncated_observation
        return observation
    
    def retry_fails(self, task):
        task = task.get('question') if isinstance(task, dict) else task      
        self.main_agent = self.solving_agent    
        obs = "Your answer is incorrect. Please reflect on it, identify the mistakes, make the necessary corrections in 4 Steps, and then resubmit the revised answer."
        result, finished = self.run(task,solving_steps=8, tool_steps=9, obs=obs, max_tool_request=2, is_retry=True)
        return result, finished
    
    def clear(self):
        self.env.clear_env()
        self.main_agent.clear()
        self.solving_agent.clear()
        self.tool_agent.clear()
    
        with open(os.path.join(self.toolkit_path, "tools.py"), 'w') as f:
            if self.task_type.lower() == "textcraft":
                f.write(TEXTCRAFT_HEADER)
            for tool in self.toolnet.skills:
                f.write(self.toolnet.skills[tool]["code"])
                f.write("\n\n")
        with open(os.path.join(self.toolkit_path, "temp_tools.py"), 'w') as f:
            f.write("from tools import *\nimport re\nimport pandas as pd\n")
            
    def load_task(self, task: Dict):
        prompt = ""
        import shutil
        if self.task_type.lower() == "dabench":
            question = task["question"]
            filename = task["file_name"]
            constraints = task["constraints"]
            format = task["format"]
            prompt = (
                f"Task Description\n:{question}\n",
                f"Task File: `{filename}`\n",
                f"Task Constraints\n: {constraints}\n",
                f"Answer Format\n: {format}\n"
            )
            prompt = ''.join(prompt)
            table_dir = Path(self.toolkit_path).resolve().parent / "dataset" / "da-dev-tables"
            table_file = table_dir / filename
            tgt_file = Path(self.tmp_dir).resolve() / filename
            shutil.copy(table_file, tgt_file)
        elif self.task_type.lower() == "textcraft":
            id = task["id"]
            question = task["question"]
            command = task["commands"]
            prompt = f"{command}\n"
            prompt += f"Goal: {question}"
            os.makedirs(Path(self.tmp_dir, "textcraft"), exist_ok=True)
            shutil.copytree(Path(self.toolkit_path, "textcraft"), Path(self.tmp_dir, "textcraft"), dirs_exist_ok=True)
       
        else:
            id = task["id"]
            question = task["question"]
            prompt = f"Task: {question}"
        
        
        open(Path(self.tmp_dir) / "tmp0.py", 'w').close()
        open(Path(self.tmp_dir) / "temp_tools.py", "w").close()
        
        with open(Path(self.tmp_dir) / "tools.py", "w") as f:
            src = open(Path(self.toolkit_path) / "tools.py", 'r')
            f.write(src.read())
            src.close()
        return prompt
