import logging, os
from abc import ABC, abstractmethod
import numpy as np
from openai import OpenAI

from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return np.array(
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )
