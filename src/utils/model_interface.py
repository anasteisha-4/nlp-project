from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class Retriever(ABC):
    @abstractmethod
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        pass
    
    @abstractmethod
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        pass
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        return self.retrieve(query, top_k)


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def score(self, query: str, passages: List[str]) -> List[float]:
        pass
