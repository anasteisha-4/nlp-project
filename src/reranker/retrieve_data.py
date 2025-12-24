# generate_retrieved_top100.py
import json
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

def main():
    corpus = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-corpus", split="corpus")
    queries = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-queries", split="test")
    train_qrels = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-qrels", split="test")

    corpus_dict = {d["_id"]: d["text"] for d in corpus if d.get("text")}
    queries_dict = {q["_id"]: q["text"] for q in queries if q.get("text")}
    train_query_ids = {qrel["query-id"] for qrel in train_qrels}

    train_queries = {qid: queries_dict[qid] for qid in train_query_ids if qid in queries_dict}

    model = SentenceTransformer("models/retriever")

    corpus_ids = list(corpus_dict.keys())
    corpus_texts = ["passage: " + corpus_dict[cid] for cid in corpus_ids]
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, batch_size=32, show_progress_bar=True)

    query_ids = list(train_queries.keys())
    query_texts = ["query: " + train_queries[qid] for qid in query_ids]
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, batch_size=32, show_progress_bar=True)

    results = semantic_search(query_embeddings, corpus_embeddings, top_k=100)

    output = []
    for i, qid in enumerate(query_ids):
        hits = results[i]
        candidates = [{"passage_id": corpus_ids[hit["corpus_id"]], "score": float(hit["score"])} for hit in hits]
        output.append({"query_id": qid, "candidates": candidates})

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/retrieved_top100_test.jsonl", "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()