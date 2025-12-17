from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_interface import Retriever, Reranker


class BasePipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Optional[Reranker] = None
    ):
        self.retriever = retriever
        self.reranker = reranker
    
    def search(
        self,
        query: str,
        top_k_retrieve: int = 100,
        top_k_final: int = 10
    ) -> List[Dict[str, Any]]:

        candidates = self.retriever.retrieve(query, top_k=top_k_retrieve)
        
        if self.reranker is None:
            return candidates[:top_k_final]
        reranked = self.reranker.rerank(query, candidates)
        
        return reranked[:top_k_final]
    
    def batch_search(
        self,
        queries: List[str],
        top_k_retrieve: int = 100,
        top_k_final: int = 10
    ) -> List[List[Dict[str, Any]]]:
        results = []
        for query in queries:
            result = self.search(query, top_k_retrieve, top_k_final)
            results.append(result)
        return results


if __name__ == "__main__":
    from retriever.baseline_retriever import BaselineRetriever
    from reranker.baseline_reranker import BaselineReranker
    
    retriever = BaselineRetriever()
    reranker = BaselineReranker()
    
    test_corpus = [
        {"doc_id": "1", "text": "Warszawa jest stolicą Polski i największym miastem w kraju."},
        {"doc_id": "2", "text": "Kraków to historyczne miasto w Polsce, dawna stolica."},
        {"doc_id": "3", "text": "Góry Tatry znajdują się na granicy Polski i Słowacji."},
        {"doc_id": "4", "text": "Wisła to najdłuższa rzeka w Polsce."},
        {"doc_id": "5", "text": "Polska ma dostęp do Morza Bałtyckiego."},
    ]
    
    retriever.index_corpus(test_corpus)
    
    pipeline = BasePipeline(retriever, reranker)
    
    queries = [
        "Jaka jest stolica Polski?",
        "Które góry są w Polsce?",
    ]
    
    for query in queries:
        results = pipeline.search(query, top_k_retrieve=3, top_k_final=2)
        for r in results:
            print(f"  [{r['doc_id']}] score={r['score']:.4f}: {r['text'][:60]}...")
