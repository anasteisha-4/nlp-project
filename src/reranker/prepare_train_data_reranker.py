# prepare_reranker_pairs.py
import json
import os
from datasets import load_dataset
from tqdm import tqdm


def main():
    corpus = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-corpus", split="corpus")
    queries = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-queries", split="train")
    train_qrels = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-qrels", split="train")

    qrel_map = {q["query-id"]: q["corpus-id"] for q in train_qrels}
    corpus_dict = {d["_id"]: d["text"] for d in corpus if d.get("text")}
    queries_dict = {q["_id"]: q["text"] for q in queries if q.get("text")}

    retrieved = []

    with open("data/processed/retrieved_top100.jsonl", "r", encoding="utf-8") as f:
        for line in tqdm(f):
            retrieved.append(json.loads(line))

    pair_data = []
    max_pairs = 200000
    max_negatives = 1
    for item in tqdm(retrieved):
        if len(pair_data) >= max_pairs:
            break

        qid = item["query_id"]
        if qid not in queries_dict or qid not in qrel_map:
            continue

        query = queries_dict[qid]
        true_pid = qrel_map[qid]

        if true_pid not in corpus_dict:
            continue

        pos_text = corpus_dict[true_pid]
        hard_negs = 0
        for rank_idx in range(1, 5):
            if hard_negs >= max_negatives:
                break
            if rank_idx >= len(item["candidates"]):
                break

            cand = item["candidates"][rank_idx]
            pid = cand["passage_id"]

            if pid == true_pid or pid not in corpus_dict:
                continue

            neg_text = corpus_dict[pid]

            pair_data.append({
                "query": query,
                "pos": pos_text,
                "neg": neg_text
            })

            hard_negs += 1

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/reranker_train_pairs.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in pair_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
