from abc import ABC, abstractmethod
import logging,re, json
from typing import Dict, Union, Optional, List
from .config import Config
from .llm import OpenAILLM
from .action import *
from .memory import Memory
from copy import deepcopy
from . import action
import json5


class Agent:
    def __init__(self, config: Union[Config, Dict, str]):
        self.config = config if isinstance(config, Config)\
            else Config(config)
        self.agent_name = self.config.agent_name
        self.available_action = self.config.available_action
        self.system_prompt = self.config.system_prompt
        self.llm_config = self.config.llm_config
        self.memory = Memory(messages=[], window_size=self.config.memory_window_size)
        self.llm = OpenAILLM(self.llm_config) if self.llm_config \
            else None    
        self._post_init()
    
    def _post_init(self):
        try:
            with open(self.system_prompt, 'r', encoding='utf-8') as system:
                self.system_prompt = system.read()
        except FileExistsError:
            raise f"{self.system_prompt} does not exist"
        available_action_class = []
        for action_str in self.available_action:
            try:
                action_class = getattr(action, action_str)
                available_action_class.append(action_class)
            except AttributeError:
                raise ValueError(f"Class {action_str} not found in module 'action'")
        self.available_action = available_action_class

    def clear(self):
        try:
            with open(self.config.system_prompt, 'r', encoding='utf-8') as system:
                self.system_prompt = system.read()
        except FileExistsError:
            raise f"{self.system_prompt} does not exist"
        self.memory.clear_memory()
    
    def generate(self, task: Optional[str]=None, obs: Optional[str]=None, **kwargs) -> List:
        
        add_history = kwargs.get("add_history", True)    
        extra_msg = kwargs.get("extra_msg", {})
        system_prompt = self.system_prompt
        for key, content in extra_msg.items():
            system_prompt = system_prompt.replace(f"==={key}===", content)
        
        available_action = '\n'.join([specific_action.get_action_description()
        for specific_action in self.available_action])
        system_prompt = system_prompt.replace("===action===", available_action)
        system_prompt = system_prompt.replace("===task===", task)

        messages = [{
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                },
            ]
        }] + deepcopy(self.memory.get_latest_memory())
        
        messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Observation: {}\n".format(str(obs))
                    }
                ]
            })  
        
        for i in range(3):
            status, response = self.llm.get_response(messages)
            response = response.strip()
            if not status:
                if response.lower() in ["context_length_exceeded","rate_limit_exceeded","max_tokens"]:
                    messages.pop(-1)
                    messages = [messages[0]] + messages[2:]
            else:
                break
            
        try:
            action = self.parse_action(response)
        except ValueError as e:
            action = None
        message = {"obs": obs, "response": response,
            "name": self.agent_name} 
        if add_history:
            self.memory.extend_memory(messages=message)

        return response, action
    
    def parse_action(self, output: Optional[Union[List[str], str]] ) -> Action:
        
        if output is None or len(output) == 0:
            pass
        action_string = ""
        patterns = [ r'["\']?Action["\']?:? (.*?)Action', 
            r'["\']?Action["\']?:? (.*?)Observation',r'["\']?Action["\']?:? (.*?)Thought', 
            r'["\']?Action["\']?:? (.*?)$', r'^(.*?)Observation']
        for p in patterns:
            match = re.search(p, output, flags=re.DOTALL)
            if match:
                action_string = match.group(1).strip()
                break
        if action_string == "":
            action_string = output.strip()
      
        output_action = None
        for action_cls in self.available_action:
            action = action_cls.parse_action_from_text(action_string)
            if action is not None:
                output_action = action
                break
        if output_action is None:
            action_string = action_string.replace("\_", "_").replace("'''","```")
            for action_cls in self.available_action:
                action = action_cls.parse_action_from_text(action_string)
                if action is not None:
                    output_action = action
                    break
        
        return output_action
        
        

        
        

    
    
    
        
        
        
        
        