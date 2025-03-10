from typing import Dict, Union
import yaml, json, os
from abc import ABC, abstractmethod

class Config:
    def __init__(self, config: Union[str, Dict] = None):
        self.config = self.load_cfg(config)
        
    def load_cfg(self, config: Union[str, Dict] = None):
        if not config:
            return {}
        try:
            if isinstance(config, dict):
                return config
            elif isinstance(config, str):
                with open(config, 'r', encoding='utf-8') as cfg:
                    config_data = json.load(cfg)
                    return config_data
        except Exception as e:
            raise TypeError("Please provide a right config with dict or json file")       


class LLMConfig(Config):

    def __init__(self, config: Union[str, Dict] = None):
        super().__init__(config)
        self.LLM_type: str = self.config.get("LLM_type", "OpenAI")
        self.model: str = self.config.get("model", "gpt-4-turbo-2024-04-09")
        self.log_path: str = self.config.get("log_path", "logs/llm.log")
        self.temperature: float = self.config.get("temperature", 0.0)
        self.max_tokens: int = self.config.get("max_tokens", 1500)
        self.top_p: float = self.config.get("top_p", 0.9)
        self.API_KEY: str = self.config.get(
            "OPENAI_API_KEY", os.environ["OPENAI_API_KEY"]
        )
        self.API_BASE = self.config.get(
            "OPENAI_BASE_URL", os.environ.get("OPENAI_BASE_URL")
        )
            
class AgentConfig(Config):
    
    def __init__(self, config: Union[str, Dict] = None):
        super().__init__(config)
        self.agent_name = self.config.get("agent_name")
        self.available_action = self.config.get("available_action")
        self.llm_config = self.config.get("llm_config")
        self.system_prompt = self.config.get("system_prompt")
        self.memory_window_size = self.config.get("memory_window_size", 10)


class MultiAgentsConfig(Config):
    def __init__(self, config: Union[str, Dict] = None):
        super().__init__(config)
        self._load_configs()

    def _load_configs(self):
        configs = []
        for config in self.config:
            configs.append(AgentConfig(config))
        self.config = configs