from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from copy import deepcopy
from functools import wraps
from .utils import encode_memory
import json


class Memory:
    require_fields = ["obs", "thought", "action", "agent_name"]
    def __init__(self, messages: List=[], window_size: int=10):
        self._messages = messages
        self._window_size = window_size
        
    def __len__(self):
        return len(self._store)
    
    def __getitem__(self, index: Union[slice, int]):
        if isinstance(index, slice):
            return self._store[index.start:index.stop:index.step]
        elif isinstance(index, int):
            return self._store[index]
        
    @property
    def messages(self):
        return self._messages
    
    @property
    def window_size(self):
        return self._window_size
    
    @encode_memory
    def extend_memory(self, messages: List = []):
        return self._messages.extend(messages)
    
    def clear_memory(self):
        self._messages = []
    
    def get_latest_memory(self):
        if len(self._messages) < self.window_size:
            return self._messages
        else:
            return self._messages[-self.window_size:]

    def save_memory(self, path):
        with open(path, 'w') as json_file:
            json.dump(self._messages, json_file, indent=4)
        

        
        
        