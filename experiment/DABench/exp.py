
from experiment.agent_exp import TestingFramework
from agent.action import *
from tqdm import tqdm
import json, os, copy, re
from typing import Union, Dict, List
from collections import deque
from pydantic import ValidationError
import logging
import jsonlines
import re
import copy
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from agent.llm import OpenAILLM


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
        error_scale = 3e-2
        # Normalize inputs: strip whitespace and convert to lowercase for case-insensitive comparison
        response_str = response.strip().lower() if isinstance(response, str) else str(response).strip().lower()
        label_str = label.strip().lower() if isinstance(label, str) else str(label).strip().lower()

        # Check for exact (case-insensitive) string match
        if response_str == label_str:
            return True

        # Convert to float for numerical comparison
        response_float = float(response_str)
        label_float = float(label_str)
        
        # Absolute error comparison for zero labels
        if label_float == 0.0 or response_float == 0.0:
            return abs(label_float- response_float) <= error_scale / 3

        lower_bound = min(label_float * (1-error_scale),label_float * (1+error_scale))
        upper_bound = max(label_float * (1-error_scale), label_float * (1+error_scale))
        
        if lower_bound < response_float and upper_bound > response_float:
            return True
        return False
    except (ValueError, AttributeError, TypeError):
        # Handle cases where inputs are not convertible to floats or invalid
        return False
    
    
    
def fetch_args(args_lookup, logic_exp):
    out = copy.deepcopy(logic_exp)
    assert 'steps' in logic_exp.keys()
    for s, step in enumerate(logic_exp['steps']):
        if isinstance(step, int):
            out['steps'][s] = args_lookup[step]
        elif isinstance(step, dict):
            out['steps'][s] = fetch_args(args_lookup, step)
    return out


def plan_to_args(plan, keyword='Step'):
    """
    Convert a plan text to a structured dictionary with steps.

    Args:
        plan (str): Plan in text format.
        keyword (str): Keyword prefix for steps (default: "Step").

    Returns:
        dict: Structured plan with steps as a list.
    """
    args = []
    lines = plan.split('\n')
    for line in lines:
        if line.startswith(keyword):
            args.append(re.sub(r'{} \d+: '.format(keyword), '', line))

    return {'steps': args}  # Only return the steps, no logic


def parse_expression(expression):
    """
    Parse a logical expression to extract step relationships.

    Args:
        expression (str): Logical expression containing steps.

    Returns:
        dict: Parsed logical structure with steps and relationships.
    """
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

plan_prompt = '''
Your task is to develop a concise plan to help me solve data science problems effectively.

Here are some examples:

## Example 1
Task Description: Calculate the mean value of the "Close Price" column.
Task File: `GODREJIND.csv`
Task Constraints: Use the built-in Python (numpy or pandas) to calculate the mean. Do not use any pre-built packages or libraries for mean calculation other than numpy or pandas. The calculation should be done on the whole "Close Price" column. Values in this column should not be rounded or changed in any way before the calculation.
Answer Format: @mean_close_price[mean_value], where "mean_value" is a float number rounded to two decimal places. This value should be between the highest and lowest "Close Price" given in the dataset.
## Plan
# Think: To calculate the mean value of the "Close Price" column, I need to load the dataset and extract the "Close Price" values.
# I will use pandas to read the CSV file and compute the mean using the built-in method.
Step 1: Load the dataset using pandas
# Think: Now that I have loaded the dataset, I will extract the "Close Price" column.
Step 2: Extract the "Close Price" column from the dataset
# Think: With the column extracted, I will calculate the mean value using pandas' built-in method.
Step 3: Compute the mean of the "Close Price" column using pandas
# Think: The computed mean value should be rounded to two decimal places as per the requirements.
Step 4: Round the mean value to two decimal places
# Think: Finally, I will format the answer in the required format: @mean_close_price[mean_value].
Step 5: Output the result in the format "@mean_close_price[mean_value]"

## Example 2
Task Description: Which country has the highest number of deaths recorded in a single year?
Task File: `estimated_numbers.csv`
Task Constraints: Calculate the maximum value in the 'No. of deaths' column. Convert the data type of 'No. of deaths' column from Object (string) to Int64 before performing calculations. Ignore those records where 'No. of deaths' column value is Null or empty. Identify the corresponding country and year for the highest number of deaths.
Answer Format: @max_deaths_country[country_name] @max_deaths_year[year] where "country_name" is a string indicating the name of the country and "year" is an integer indicating the year in which the maximum deaths occurred.
## Plan
# Think: To determine the country with the highest number of deaths recorded in a single year, I need to load the dataset and analyze the "No. of deaths" column.
# I will use pandas to read the CSV file and preprocess the data before identifying the maximum value.
Step 1: Load the dataset using pandas  
# Think: Now that I have loaded the dataset, I will check the "No. of deaths" column and convert its data type from Object (string) to Int64 for numerical operations.
Step 2: Convert the "No. of deaths" column from Object to Int64, ignoring Null or empty values  
# Think: With the column converted to numerical data, I will identify the maximum value in the "No. of deaths" column.
Step 3: Find the maximum value in the "No. of deaths" column  
# Think: Now that I have the maximum number of deaths, I will locate the corresponding country and year from the dataset.
Step 4: Extract the country and year corresponding to the maximum number of deaths  
# Think: Finally, I will format the answer as required: @max_deaths_country[country_name] @max_deaths_year[year].
Step 5: Output the result in the format "@max_deaths_country[country_name] @max_deaths_year[year]"

## Example 3
Task Description: What is the distribution of ages among the male passengers who did not survive? Is it significantly different from the distribution of ages among the female passengers who did not survive?
Task File: `titanic_train.csv`
Task Constraints: Calculating the distribution of ages should use a Kernel Density Estimation (KDE) method. Perform a two-sample Kolmogorov-Smirnov test to compare the distributions. Use a significance level (alpha) of 0.05. If the p-value is less than 0.05, conclude the distributions are significantly different. If the p-value is greater than or equal to 0.05, conclude the distributions are not significantly different.
Answer Format: @is_significantly_different[answer] where "answer" is a boolean indicating the result of the test. For example, if the distributions are significantly different, the answer should be "True". If not, the answer should be "False".

## Plan
# Think: To analyze the age distribution of male and female passengers who did not survive, I need to filter the dataset based on gender and survival status.
# I will use pandas to load the dataset and extract relevant subsets of the data.
Step 1: Load the dataset using pandas  
# Think: Now that I have the dataset, I will filter out male and female passengers who did not survive.
Step 2: Filter the dataset for male passengers who did not survive  
Step 3: Filter the dataset for female passengers who did not survive  
# Think: With both subsets ready, I will extract the "Age" column and remove any missing values.
Step 4: Extract and clean the "Age" column for both male and female non-survivors  
# Think: Now I will use Kernel Density Estimation (KDE) to visualize the age distributions.
Step 5: Compute the KDE for both male and female non-survivors  
# Think: To compare the distributions statistically, I will perform a two-sample Kolmogorov-Smirnov test.
Step 6: Conduct the Kolmogorov-Smirnov test between the two distributions  
# Think: Based on the p-value from the test, I will determine if the distributions are significantly different using a significance level of 0.05.
Step 7: Compare the p-value to 0.05 and determine the result  
# Think: Finally, I will format the answer as required: @is_significantly_different[answer].
Step 8: Output the result in the format "@is_significantly_different[answer]"


Here is a different goal with different data processing steps. Your task is to come up with a short plan to help me accomplish my goal in a couple of steps using at most ONE of the provided data processing methods. You can take the help of the methods below to transform or analyze the data. Keep in mind that:
- It is okay to generate more insights or results than strictly required.
- Be very careful with numerical calculations, ensuring consistency with the provided operations.
- You cannot modify a predefined data processing step; i.e., if an operation applies to an entire column, you CANNOT alter it to apply only to a subset unless explicitly stated.
'''


class DABenchFramework(TestingFramework):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

            
    def plan_llm(self, prompt):
        prompt = prompt
        sys_prompt = "You are an helpful assistant helping me deal with data science task.'"
        messages = [
            self.encoded_message(sys_prompt, "system"),
            self.encoded_message(plan_prompt + "\n" + prompt, "user")
        ]
        
        response = self.get_response(messages)
        
        return response
    
    
    def get_plan(self, task, max_attempts=3):
        """
        Generate a plan for a given task using LLM, with retry mechanism.

        Args:
            task (str): The specific task to accomplish.
            
            prompt_template (str): Template for prompting the LLM to generate a plan.
            max_attempts (int): Maximum number of retry attempts.
            verbose (bool): If True, prints detailed logs.

        Returns:
            dict: Parsed plan with steps and logic (e.g., {"steps": [...], "logic": "AND"}).
        """
        custom_prompt = task
        
        for attempt in range(max_attempts):
            try:
                logging.info(f"Attempt {attempt + 1}: Generating plan with LLM...")
                # Call the LLM to generate a plan
                plan_text = self.plan_llm(custom_prompt)
                logging.info(f"Plan generated by LLM:\n{plan_text}")
                # Parse the plan text into a structured format
                plan = plan_to_args(plan_text)
                if not plan["steps"]:
                    continue
                logging.info(f"Parsed Plan:\n{plan}")
                return plan
            except Exception as e:
                logging.info(f"Error in attempt {attempt + 1} while generating or parsing the plan: {e}")
        
        # If all attempts fail, return a default empty plan
        logging.info(f"Failed to generate a valid plan after {max_attempts} attempts. Returning default plan.")
        return {"steps": []}
    
    
    def execute_plan(self, plan):
        """
        Execute a given plan in the environment based on AND/OR logic.

        Args:
            plan (dict): The parsed plan, including steps and logic.
          
        Returns:
            tuple: (success, info_prop, action_checkpoint, step_results)
                - success (bool): Whether the plan execution was successful.
                - info_prop (str): Updated environmental state or inventory.
                - action_checkpoint (list): Updated list of executed actions.
                - step_results (list): Results of each step execution (e.g., success/failure).
        """
        
        action_checkpoint = []
        obs_checkpoint = []
        steps = plan.get("steps", [])
        
        logging.info(f"Executing plan: {plan}")

        step_results = []
        prompt = ""
        for step in steps:
            logging.info(f"Executing step: {step}")
            
            # Execute the step using the environment and LLM
            success, action_history, obs_history, result = self.execute_step(
                step, prompt)
            step_results.append({"step": step, "success": success, "result": result})

            # Handle AND logic: any step failure causes plan failure
            if not success:
                logging.info(f"Step failed (AND logic): {step}")
                return False,  step_results
            
            prompt += f"Goal: {step}\n"
            for action, obs in zip(action_history, obs_history):
                prompt += f"Action: {action.print_action()}\n"
                prompt += f"{obs}\n"
            action_checkpoint.extend(action_history)
            obs_checkpoint.extend(obs_history)
            
        # If we complete all steps for AND, or none succeed for OR
        overall_success = True
        logging.info(f"Plan execution completed.")
        return overall_success, step_results
    
    def execute_step(self, step, prompt, max_attempts=10):
        """
        Execute a single step in the environment as a multi-step process.
        Args:
            step (str): The step to execute.
            env: The environment to interact with.
            prompt (str): The LLM prompt to guide action generation.
            info_prop (str): Current environmental state or inventory.
            action_checkpoint (list): Actions executed so far.
            max_attempts (int): Maximum number of attempts to execute the step.
            verbose (bool): Whether to print detailed logs.

        Returns:
            tuple: (success, updated_info_prop, updated_action_checkpoint)
        """
        
        action_history = []
        obs_history = []
        done = False
        result = ""
        step_idx, retry_count = 0, 0
       
        finished = False
        logging.info("Sovling Loop Begin")
        failed_action = 0
        
        obs = f"Here’s a summary of the steps you’ve completed so far:\n{prompt}\nNow, please proceed to tackle the current goal."
        step = f"Goal: {step}"
        
        logging.info(f"Step: {step}")
        
        while step_idx < max_attempts:
            logging.info(f"Solving Loop: Step {step_idx + 1} SolvingAgent: {obs}")
            # import pdb
            # pdb.set_trace()
            res, action = self.agent.generate(step + f"\nFile Name: `{self.filename}`\nAnswer Constraint: {self.constraints}", obs)
            logging.info(f"Solving Loop: Step {step_idx + 1} SolvingAgent response: {res}")
            if action is None:
                logging.error("Failed to parse action from response.")
                if retry_count >= 3:
                    logging.error(f"Failed to extract the action {retry_count} consecutive times. Stop...")
                    return False, action_history, obs_history, None
                retry_count += 1
                obs = "Failed to parse action from your response, please make sure you provide a valid action."
                continue
            else:             
                obs, done = self.step(action)
                obs = f"Observation: {obs}"

            if done:
                failed_action = 0
                action_history.append(action)
                obs_history.append(obs)
                if isinstance(action, Terminate):
                    result = action.argument["result"]
                    logging.info("The task is done.")
                    finished = True
                    break
            else:
                failed_action += 1
                if failed_action >= 3:
                    return False, action_history, obs_history, None
                continue
            step_idx += 1
            
        return finished, action_history, obs_history, result
    
    def run(self, task: Dict, **kwargs):
        
        solving_sys_prompt = kwargs.get(
            "solving_sys_prompt", 
            f"experiment/{self.task_type}/prompt_lib/prompt_CREATOR_decision.txt" if self.mode == "pipeline" \
                else f"experiment/{self.task_type}/prompt_lib/prompt_solving.txt" 
        )

        retrieval_sys_prompt =  kwargs.get(
            "retrieval_sys_prompt", 
            f"experiment/{self.task_type}/prompt_lib/prompt_retrieval.txt"
        )
        import shutil
        shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)
        # Initialize variables
        retrieval_tools, tool_results, examples, decision_examples = [], "", "", ""
        query = self.load_task(task)
        
        # Retrieve APIs if not in primitive mode
        if not self.primitive:
            logging.info("Retrieving APIs...")
            retrieval_tools, tool_results, examples, decision_examples = self.retrieve(query, retrieval_sys_prompt)
        
    
        with open(solving_sys_prompt, 'r') as f:
                sys_prompt = f.read()
        sys_prompt = sys_prompt.replace("===api===", tool_results)
        sys_prompt = sys_prompt.replace("===example===", examples)
        if self.mode == "pipeline":
            logging.info("Generating tools in pipeline mode...")
            # Generate tool code

            plan = self.get_plan(query)
            self.agent.system_prompt = sys_prompt
            succ, result = self.execute_plan(plan)
            if not succ:
                return retrieval_tools, ""
        else:  # Standard mode
            logging.info("Executing in standard mode...")
            sys_prompt = sys_prompt.replace("===task===", query)
            self.agent.system_prompt = sys_prompt
            logging.info(f"System Prompt:\n{sys_prompt}")
            # Generate tool code
            solving_obs = "Observation: Think step by step. Please provide one Thought and one action from the action space."
            result, finished = self.solving_loop(query, solving_obs)
            if not finished:
                return retrieval_tools, ""
            
        return (retrieval_tools, result)
    
    def test(self, tasks: Union[Dict,List[Dict]], output_path: str, **kwargs):
        if not output_path.endswith(".jsonl"):
            raise ValueError("Output file should be JSONL format.")
        if not os.path.exists(output_path):
            open(output_path, 'w').close()
        tasks = tasks if isinstance(tasks, list) else [tasks]
        for task in tasks:
            result = {}
            result.update(task)
            
            logging.info(f"Question: {task['question']}")
            std_answer = task["answer"]
            result.pop("answer", "")
            result["retrieved_api"] = []

            retrieval_tools, llm_answer = self.run(task, **kwargs)
            for tool in retrieval_tools:
                result["retrieved_api"].append(tool["name"])
                
            result["std_answer"] = task["answer"]
            result["llm_answer"] = llm_answer
        
            is_right = self.check_is_answer_right(task,gt_answer=std_answer, pre_answers=llm_answer)
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
                
    def solving_loop(self, task, obs, **kwargs):
        solving_steps = kwargs.get("solving_steps", 15)
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
    
    def check_is_answer_right(self, task, gt_answer, pre_answers):
        try:
            if self.mode == "pipeline":
                pre_answers = [ans["result"] for ans in pre_answers]
            else:
                pre_answers = [pre_answers]
        except:
            pre_answers = [""]

        llm = OpenAILLM({
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 200,
            "top_p": 0.9
        })

        with open(r'experiment/DABench/prompt_lib/answerchecker.txt', 'r') as f:
            system_prompt = f.read()

        for pre_answer in pre_answers:
            pre_answer = pre_answer if pre_answer else "Fails to generate answer."
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
                        "text": f"Standard Answer: {gt_answer}\nGiven Answer: {pre_answer}"
                    }]
                }
            ]
            results = []

            for _ in range(6):
                if len(results) == 3:
                    break
                _, result = llm.get_response(messages=message, api_base="https://api2.aigcbest.top/v1")
                logging.info(f"Answer Checking {len(results) + 1} times: {result}")

                patterns = [r'["\']?Response["\']?:? (.*?)Response', 
                            r'["\']?Response["\']?:? (.*?)Thought', 
                            r'["\']?Response["\']?:? (.*?)$']
                response = ""
                for pattern in patterns:
                    match = re.search(pattern, result, flags=re.DOTALL)
                    if match:
                        response = match.group(1).strip()
                        break
                if not response:
                    continue
                if 'true' in response.lower():
                    return True
                elif 'false' in response.lower():
                    results.append(False)

        return False
    '''
    def check_is_answer_right(self, task, gt_answer, pre_answer):
        label_answers = {ans[0]: ans[1] for ans in gt_answer}
        try:
            if self.mode == "pipeline":
                pre_answer = pre_answer[-1]["result"]
        except:
            pre_answer = ""
        pred_answer_names, pred_answers = extract_format(pre_answer)
        print(pred_answers)
        # Convert to lower case and clean up
        extracted_answers = {key.lower().strip(): value for key, value in zip(pred_answer_names, pred_answers)}

        gt_answer_lower = {
                key.lower().strip(): (
                    [v.strip("'\" ").strip() for v in value.split(',')]
                    if ',' in value else value.strip("'\" ").strip() 
                ) if isinstance(value, str) else value
                for key, value in label_answers.items()
            }
        print(gt_answer_lower)


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
                # Compare as individual elements
                correct_answers[ans_name] = is_equal(pred_value, gt_value)
        
        return all(correct_answers.values()) if correct_answers else False
    '''
    def load_task(self, task: Dict):
        prompt = ""
        if self.task_type.lower() == "dabench":
            import shutil
            question = task["question"]
            self.filename = task["file_name"]
            self.constraints = task["constraints"]
            format = task["format"]
            prompt = (
                f"Task Description:{question}\n",
                f"Task File: `{self.filename}`\n",
                f"Task Constraints: {self.constraints}\n",
                f"Answer Format: {format}\n"
            )
            prompt = ''.join(prompt)
            tgt_dir = Path(self.tmp_dir).resolve()
            table_dir = Path(self.toolkit_path).resolve().parent / "dataset" / "da-dev-tables"
            shutil.copy(table_dir / self.filename, tgt_dir)
            
            open(Path(self.tmp_dir) / "tmp0.py", 'w').close()
            open(Path(self.tmp_dir) / "temp_tools.py", "w").close()
            with open(Path(self.tmp_dir) / "tools.py", "w") as f:
                src = open(Path(self.toolkit_path) / "tools.py", 'r')
                f.write(src.read())
                src.close()
                
        return prompt
            