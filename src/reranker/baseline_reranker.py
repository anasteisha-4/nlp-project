import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_interface import Reranker


class BaselineReranker(Reranker):
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
    
    def score(self, query: str, passages: List[str]) -> List[float]:
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        passage_embeddings = self.model.encode(passages, convert_to_numpy=True)
        
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
        passage_embeddings = passage_embeddings / (norms + 1e-8)
        
        scores = np.dot(passage_embeddings, query_embedding)
        
        return scores.tolist()
    
    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [p["text"] for p in passages]
        scores = self.score(query, texts)
        for i, passage in enumerate(passages):
            passage["score"] = scores[i]

        sorted_passages = sorted(passages, key=lambda x: x["score"], reverse=True)

        return sorted_passages


if __name__ == "__main__":
    reranker = BaselineReranker()
    
    query = "Jaka jest stolica Polski?"
    passages = [
        {"doc_id": "1", "text": "Kraków to historyczne miasto w Polsce."},
        {"doc_id": "2", "text": "Warszawa jest stolicą Polski."},
        {"doc_id": "3", "text": "Góry Tatry są bardzo piękne."},
    ]
    
    results = reranker.rerank(query, passages)
    
    print(f"Query: {query}")
    print("Reranked results:")
    for r in results:
        print(f"  [{r['doc_id']}] score={r['score']:.4f}: {r['text']}")
