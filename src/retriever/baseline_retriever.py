import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_interface import Retriever


class BaselineRetriever(Retriever):
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.corpus: List[Dict[str, Any]] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        return self.model.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=len(queries) > 10
        )
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        return self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=len(documents) > 10
        )
    
    def index_corpus(self, corpus: List[Dict[str, Any]]) -> None:
        self.corpus = corpus
        texts = [doc["text"] for doc in corpus]
        self.corpus_embeddings = self.encode_documents(texts)
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        self.corpus_embeddings = self.corpus_embeddings / (norms + 1e-8)
        
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if self.corpus_embeddings is None:
            raise ValueError("Corpus not indexed. Call index_corpus() first.")

        query_embedding = self.encode_queries([query])[0]
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        scores = np.dot(self.corpus_embeddings, query_embedding)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "doc_id": self.corpus[idx]["doc_id"],
                "text": self.corpus[idx]["text"],
                "score": float(scores[idx])
            })
        
        return results


if __name__ == "__main__":
    retriever = BaselineRetriever()
    
    test_corpus = [
        {"doc_id": "1", "text": "Warszawa jest stolicą Polski."},
        {"doc_id": "2", "text": "Kraków to historyczne miasto w Polsce."},
        {"doc_id": "3", "text": "Góry Tatry znajdują się na granicy Polski i Słowacji."},
    ]
    
    retriever.index_corpus(test_corpus)
    
    query = "Jaka jest stolica Polski?"
    results = retriever.retrieve(query, top_k=2)
    
    print(f"Query: {query}")
    print("Results:")
    for r in results:
        print(f"  [{r['doc_id']}] score={r['score']:.4f}: {r['text']}")
