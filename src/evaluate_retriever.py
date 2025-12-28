from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from metrics import recall_at_k, hit_rate_at_k, mrr, ndcg_at_k

USE_ARTIFICIAL_DATA = False

def main():
    if USE_ARTIFICIAL_DATA:
        print("Using artificial dataset for testing metrics...")

        corpus_texts = ["doc1 about cats", "doc2 about dogs", "doc3 about birds"]
        corpus_ids = ["d1", "d2", "d3"]

        query_texts = ["cats", "dogs"]
        query_ids = ["q1", "q2"]

        qrels = {
            "q1": {"d1"},
            "q2": {"d2"}
        }

    else:
        print("Loading WebFAQ (pol) dataset...")

        corpus_ds = load_dataset(
            "PaDaS-Lab/webfaq-retrieval",
            "pol-corpus",
            split="corpus"
        )

        queries_ds = load_dataset(
            "PaDaS-Lab/webfaq-retrieval",
            "pol-queries",
            split="test"
        )

        qrels_ds = load_dataset(
            "PaDaS-Lab/webfaq-retrieval",
            "pol-qrels",
            split="test"
        )

        corpus = {
            d["_id"]: d["text"]
            for d in corpus_ds
            if d.get("text")
        }

        queries = {
            q["_id"]: q["text"]
            for q in queries_ds
            if q.get("text")
        }

        qrels = {}
        for row in qrels_ds:
            qid = row["query-id"]
            pid = row["corpus-id"]
            qrels.setdefault(qid, set()).add(pid)

        corpus_ids = list(corpus.keys())
        corpus_texts = ["passage: " + corpus[cid] for cid in corpus_ids]

        query_ids = list(qrels.keys())
        query_texts = ["query: " + queries[qid] for qid in query_ids]

    print(f"Loaded {len(corpus_ids)} docs, {len(query_ids)} queries")

    print("Loading retriever...")
    model_path = "/Users/anastasia/Desktop/nlp-project/models/retriever"
    model = SentenceTransformer(model_path)
    print("Encoding corpus...")
    doc_embs = model.encode(
        corpus_texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True
    )

    print("Encoding queries...")
    query_embs = model.encode(
        query_texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True
    )

    print("Retrieving...")
    scores = query_embs @ doc_embs.T

    preds = {}
    for i, qid in enumerate(query_ids):
        ranking = np.argsort(scores[i])[::-1][:20]
        preds[qid] = [corpus_ids[j] for j in ranking]

    print("\n Metrics:")
    targets = [qrels[qid] for qid in query_ids]
    predictions = [preds[qid] for qid in query_ids]

    for k in [5, 10, 20]:
        print(f"Recall@{k}:", recall_at_k(predictions, targets, k))
        print(f"HitRate@{k}:", hit_rate_at_k(predictions, targets, k))

    print("MRR:", mrr(predictions, targets))
    print("nDCG@10:", ndcg_at_k(predictions, targets, k=10))


if __name__ == "__main__":
    main()
