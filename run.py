import argparse
import logging
import os
import sys
import time
import jsonlines
from agent import SkillAgent

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SkillAgent training with logging.")
    parser.add_argument("--task_type", type=str, required=True, help="Type of task (e.g., date, text, number, etc.)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset JSONL file.")
    parser.add_argument("--toolkit_path", type=str, required=True, help="Path to the toolkit directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output files.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store log files.")
    parser.add_argument("--resume", action='store_true', help="Resume training from the last checkpoint.")
    
    args = parser.parse_args()
    
    task_type = args.task_type
    log_path = os.path.join(args.log_dir, task_type)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    
    os.makedirs(log_path, exist_ok=True)
    
    skillagent = SkillAgent(
        cfg_path=os.path.join(args.toolkit_path, "prompt_lib/SkillAgent.json"), 
        toolkit_path=args.toolkit_path, 
        basic_tools=[], 
        resume=args.resume,
        task_type=task_type
    )
    
    with jsonlines.open(args.dataset_path, 'r') as f:
        data = list(f)
    
    for idx, task in enumerate(data):
        task_id = f"{task_type}_{task['id']}"
        log_output = os.path.join(log_path, task_id + '.log')
        if os.path.exists(log_output):
            continue
        
        file_handler = logging.FileHandler(log_output, encoding='utf-8')
        formatter = logging.Formatter('%(levelname)s - %(message)s\n')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        skillagent.train(
            task, 
            output_path=os.path.join(args.output_path, "train.jsonl"),
            save_dir=os.path.join(args.output_path, "output"),
            turn=idx
        )
        
        logger.removeHandler(file_handler)
        file_handler.close()
