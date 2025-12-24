# apply_reranker_custom_fp16.py
import json
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.cuda.amp import autocast

def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rerank_batch_size = 64

    corpus = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-corpus", split="corpus")
    queries = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-queries", split="test")

    corpus_dict = {d["_id"]: d["text"] for d in corpus if d.get("text")}
    queries_dict = {q["_id"]: q["text"] for q in queries if q.get("text")}

    query_ids = sorted(queries_dict.keys())
    test_queries = {qid: queries_dict[qid] for qid in query_ids}

    retriever = SentenceTransformer("models/retriever")

    corpus_items = sorted(corpus_dict.items())
    corpus_ids = [cid for cid, _ in corpus_items]
    corpus_texts = ["passage: " + text for _, text in corpus_items]
    query_texts = ["query: " + test_queries[qid] for qid in query_ids]

    corpus_embs = retriever.encode(corpus_texts, convert_to_tensor=True,
                                   batch_size=32, show_progress_bar=True)
    
    query_embs = retriever.encode(query_texts, convert_to_tensor=True,
                                  batch_size=32, show_progress_bar=True)

    retrieved_results = semantic_search(query_embs, corpus_embs, top_k=100)


    reranker_path = "model/reranker"
    tokenizer = AutoTokenizer.from_pretrained(reranker_path)
    reranker = AutoModelForSequenceClassification.from_pretrained(reranker_path).to(device)
    reranker.eval()

    final_results = []
    for i, qid in enumerate(tqdm(query_ids, desc="Reranking")):
        query = test_queries[qid]
        candidates = retrieved_results[i]

        passage_ids = [corpus_ids[hit["corpus_id"]] for hit in candidates]
        passages = [corpus_dict[pid] for pid in passage_ids]

        scored = []

        for passage_batch, pid_batch in zip( 
            batch_iter(passages, rerank_batch_size), batch_iter(passage_ids, rerank_batch_size)):
            queries_batch = [query] * len(passage_batch)

            inputs = tokenizer( queries_batch, passage_batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt").to(device)

            with torch.no_grad(), autocast():
                logits = reranker(**inputs).logits.view(-1)

            scored.extend(zip(pid_batch, logits.cpu().tolist()))

        scored.sort(key=lambda x: x[1], reverse=True)
        top10 = [{"passage_id": pid, "score": float(score)} for pid, score in scored[:10]]
        final_results.append({"query_id": qid, "passages": top10})

    os.makedirs("results", exist_ok=True)
    output_path = "top10.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
