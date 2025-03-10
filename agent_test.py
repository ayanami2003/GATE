import argparse
import logging
import jsonlines
import os
import sys
from experiment.agent_exp import TestingFramework
from experiment.textcraft.exp import MinecraftFramework
from experiment.DABench.exp import DABenchFramework

class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        buf = buf.replace('^', '')
        if buf.strip():  
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())
        self.logger.handlers[0].flush()

    def flush(self):
        pass

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run LLM testing framework.")
    parser.add_argument('--task_type', type=str, required=True, help="Type of task (e.g., MATH, NLP).")
    parser.add_argument('--test_path', type=str, required=True, help="Path to the test dataset.")
    parser.add_argument('--output_dir', type=str, default="", help="Directory to save notebook output")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the LLM model.")
    parser.add_argument('--exp_name', type=str, required=True, help="Name of the experiment.")
    parser.add_argument('--toolkit_path', type=str, required=True, help="Path to the toolkit.")
    parser.add_argument('--solving_sys_prompt', type=str, required=True, help="Path to the system prompt file.")
    parser.add_argument('--temperature', type=float, default=0.3, help="Temperature for the LLM.")
    parser.add_argument('--max_tokens', type=int, default=2000, help="Max tokens for the LLM.")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-p sampling for the LLM.")
    parser.add_argument('--top_k_query', type=int, default=3, help="Top-k query for the framework.")
    parser.add_argument('--top_k_tool', type=int, default=3, help="Top-k tools for the framework.")
    parser.add_argument('--test_mode', type=str, default="standard", help="Test mode for the framework (e.g., standard, advanced).")
    parser.add_argument('--is_primitive', action='store_true', help="Use primitive tools.")
    parser.add_argument('--has_demo', action='store_true', help="Include a demo in the test.")
    parser.add_argument('--tool_mode', type=str, default="direct_tool", help="Tool mode for the framework.")
    parser.add_argument('--basic_tools', type=str, nargs='*', default=[], help="List of basic tools for the framework.")
    parser.add_argument('--train_output', type=str, required=True, help="Directory to save training output.")
    
    args = parser.parse_args()
    

    # Set up paths and logger
    os.makedirs(f'results/{args.task_type}/{args.model_name}-exp', exist_ok=True)
    os.makedirs(f'results/{args.task_type}/{args.model_name}-exp/results_final', exist_ok=True)

    log_path = f'results/{args.task_type}/{args.model_name}-exp/{args.exp_name}'
    os.makedirs(log_path, exist_ok=True)
    log_path = f'results/{args.task_type}/{args.model_name}-exp/{args.exp_name}/logs'
    os.makedirs(log_path, exist_ok=True)
    
    output_dir = f'results/{args.task_type}/{args.model_name}-exp/{args.exp_name}/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    output_path = f"results/{args.task_type}/{args.model_name}-exp/results_final/{args.exp_name}.jsonl"

    if not os.path.exists(output_path):
        open(output_path, 'w').close()
        
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    
    

    with jsonlines.open(args.test_path, 'r') as f:
        data = list(f)

    agent_config = {
        "agent_name": "SolvingAgent",
        "llm_config": {
            "model": args.model_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p
        },
        "available_action": ["NotebookBlock", "Terminate"],
        "system_prompt": args.solving_sys_prompt,
        "memory_window_size": 20
    }

    if args.task_type.lower() == "dabench":
        testagent = DABenchFramework(
            agent_cfg=agent_config,
            task_type=args.task_type,
            train_output=args.train_output,
            train_embed_path=f"Datasets/{args.task_type}/task",
            top_k_query=args.top_k_query,
            top_k_tool=args.top_k_tool,
            test_mode=args.test_mode,
            output_dir=output_dir,
            is_primitive=args.is_primitive,
            has_demo=args.has_demo,
            tool_mode=args.tool_mode,
            toolkit_path=args.toolkit_path,
            exp_name=args.exp_name,
            basic_tools=[]
        )
    elif args.task_type.lower() == "textcraft":
         testagent = MinecraftFramework(
            agent_cfg=agent_config,
            task_type=args.task_type,
            train_output=args.train_output,
            train_embed_path=f"Datasets/{args.task_type}/task",
            top_k_query=args.top_k_query,
            top_k_tool=args.top_k_tool,
            test_mode=args.test_mode,
            output_dir=output_dir,
            is_primitive=args.is_primitive,
            has_demo=args.has_demo,
            tool_mode=args.tool_mode,
            toolkit_path=args.toolkit_path,
            exp_name=args.exp_name,
            basic_tools=["check_inventory", "get_object", "craft_object"]
        )
         
    # Process each task
    for idx, task in enumerate(data):     
        id = f"{args.task_type}_{task['id']}"
        log_output = os.path.join(log_path, id + '.log')
        if os.path.exists(log_output):
            continue

        file_handler = logging.FileHandler(log_output, encoding='utf-8')
        formatter = logging.Formatter('%(levelname)s - %(message)s\n')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        data = testagent.test(task, output_path, solving_sys_prompt=args.solving_sys_prompt)
        logger.removeHandler(file_handler)
        file_handler.close()

if __name__ == "__main__":
    main()