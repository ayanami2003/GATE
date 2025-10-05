import argparse
import logging
import jsonlines
import os
import sys
from experiment.llm_exp import TestingFramework

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
    parser.add_argument('--model_name', type=str, required=True, help="Name of the LLM model.")
    parser.add_argument('--exp_name', type=str, required=True, help="Name of the experiment.")
    parser.add_argument('--toolkit_path', type=str, required=True, help="Path to the toolkit.")
    parser.add_argument('--solving_sys_prompt', type=str, required=True, help="Path to the system prompt file.")
    parser.add_argument('--temperature', type=float, default=0.3, help="Temperature for the LLM.")
    parser.add_argument('--max_tokens', type=int, default=1024, help="Max tokens for the LLM.")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-p sampling for the LLM.")
    parser.add_argument('--top_k_query', type=int, default=3, help="Top-k query for the framework.")
    parser.add_argument('--top_k_tool', type=int, default=3, help="Top-k tools for the framework.")
    parser.add_argument('--test_mode', type=str, default="standard", help="Test mode for the framework (e.g., standard, advanced).")
    parser.add_argument('--retry_times', type=int, default=5, help="Number of retries for a test case.")
    parser.add_argument('--debug_times', type=int, default=0, help="Number of debug retries.")
    parser.add_argument('--is_primitive', action='store_true', help="Use primitive tools.")
    parser.add_argument('--has_demo', action='store_true', help="Include a demo in the test.")
    parser.add_argument('--tool_mode', type=str, default="direct_tool", help="Tool mode for the framework.")
    parser.add_argument('--basic_tools', type=str, nargs='*', default=[], help="List of basic tools for the framework.")

    args = parser.parse_args()

    os.makedirs(f'results/{args.task_type}/{args.model_name}-exp', exist_ok=True)
    os.makedirs(f'results/{args.task_type}/{args.model_name}-exp/results_final', exist_ok=True)

    log_path = f'results/{args.task_type}/{args.model_name}-exp/{args.exp_name}'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    output_path = f"results/{args.task_type}/{args.model_name}-exp/results_final/{args.exp_name}.jsonl"

    if not os.path.exists(output_path):
        open(output_path, 'w').close()

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    os.makedirs(log_path, exist_ok=True)

    with jsonlines.open(args.test_path, 'r') as f:
        data = list(f)

    # LLM Configuration
    llm_config = {
        "model": args.model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p
    }

    testagent = TestingFramework(
        llm_cfg=llm_config,
        task_type=args.task_type,
        train_output=os.path.join(args.toolkit_path, "train.jsonl"),
        train_embed_path=f"Datasets/{args.task_type}/task",
        top_k_query=args.top_k_query,
        top_k_tool=args.top_k_tool,
        test_mode=args.test_mode,
        retry_times=args.retry_times,
        debug_times=args.debug_times,
        is_primitive=args.is_primitive,
        has_demo=args.has_demo,
        tool_mode=args.tool_mode,
        toolkit_path=args.toolkit_path,
        basic_tools=args.basic_tools
    )

    for idx, task in enumerate(data):     
        id = f"{args.task_type}_{idx + 1}"
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