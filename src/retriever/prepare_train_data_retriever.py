import json
import os
import random
from tqdm import tqdm
from datasets import load_dataset
from multiprocessing import Pool, cpu_count

CORPUS_DICT = None
QUERIES_DICT = None
ALL_PIDS = None

def process_qrel(qrel):
    qid = qrel["query-id"]
    pid = qrel["corpus-id"]

    if qid not in QUERIES_DICT or pid not in CORPUS_DICT:
        return None

    if pid not in ALL_PIDS:
        return None

    query = QUERIES_DICT[qid]
    pos_passage = CORPUS_DICT[pid]

    possible_neg_pids = [p for p in ALL_PIDS if p != pid]
    if not possible_neg_pids:
        return None

    if len(possible_neg_pids) < 7:
        neg_pids = random.choices(possible_neg_pids, k=7)
    else:
        neg_pids = random.sample(possible_neg_pids, k=7)

    neg_passages = [CORPUS_DICT[nid] for nid in neg_pids]

    return {
        "query": query,
        "positive_passage": pos_passage,
        "negative_passages": neg_passages
    }

def init_worker(corpus_dict, queries_dict, all_pids):
    global CORPUS_DICT, QUERIES_DICT, ALL_PIDS
    CORPUS_DICT = corpus_dict
    QUERIES_DICT = queries_dict
    ALL_PIDS = all_pids

def main():
    corpus_ds = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-corpus", split="corpus")
    queries_ds = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-queries", split="train")
    train_qrels_ds = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-qrels", split="train")


    corpus_dict = {
        d["_id"]: d["text"].strip()
        for d in tqdm(corpus_ds, desc="corpus")
        if d.get("text") and isinstance(d["text"], str) and d["text"].strip()
    }

    queries_dict = {
        q["_id"]: q["text"].strip()
        for q in tqdm(queries_ds, desc="queries")
        if q.get("text") and isinstance(q["text"], str) and q["text"].strip()
    }


    all_pids = list(corpus_dict.keys())
    print(f"corpus: {len(corpus_dict)}")
    print(f"queries: {len(queries_dict)}")
    print(f"qrels: {len(train_qrels_ds)}")

    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(corpus_dict, queries_dict, all_pids)) as pool:
        results = list(tqdm(
            pool.imap(process_qrel, train_qrels_ds),
            total=len(train_qrels_ds)
        ))

    train_data = [r for r in results if r is not None]

    output_path = "/home/sasha/nlp/data/processed/retriever_train.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()