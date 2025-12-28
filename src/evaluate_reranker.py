import json
from datasets import load_dataset
from metrics import recall_at_k, hit_rate_at_k, mrr, ndcg_at_k

def load_run(file_path, top_k=None):
    predictions = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item["query_id"]
            if "passages" in item:
                pids = [cand["passage_id"] for cand in item["passages"]]
            else:
                pids = [cand["passage_id"] for cand in item["candidates"]]
            if top_k:
                pids = pids[:top_k]
            predictions[qid] = pids
    return predictions

def main():
    qrels_ds = load_dataset("PaDaS-Lab/webfaq-retrieval", "pol-qrels", split="test")
    qrels = {}
    for row in qrels_ds:
        qid = row["query-id"]
        pid = row["corpus-id"]
        qrels.setdefault(qid, set()).add(pid)

    retriever_preds = load_run("src/pipeline/retrieved_top100_test.jsonl", top_k=10)
    common_qids_retr = [qid for qid in qrels if qid in retriever_preds]
    targets_retr = [qrels[qid] for qid in common_qids_retr]
    preds_retr = [retriever_preds[qid] for qid in common_qids_retr]

    retr_metrics = {
        "Recall@5": recall_at_k(preds_retr, targets_retr, 5),
        "Recall@10": recall_at_k(preds_retr, targets_retr, 10),
        "MRR": mrr(preds_retr, targets_retr),
        "nDCG@10": ndcg_at_k(preds_retr, targets_retr, k=10),
    }

    reranker_preds = load_run("src/pipeline/top10.jsonl")
    common_qids_rerank = [qid for qid in qrels if qid in reranker_preds]

    targets_rerank = [qrels[qid] for qid in common_qids_rerank]
    preds_rerank = [reranker_preds[qid] for qid in common_qids_rerank]

    rerank_metrics = {
        "Recall@5": recall_at_k(preds_rerank, targets_rerank, 5),
        "Recall@10": recall_at_k(preds_rerank, targets_rerank, 10),
        "MRR": mrr(preds_rerank, targets_rerank),
        "nDCG@10": ndcg_at_k(preds_rerank, targets_rerank, k=10),
    }

    print(f"{'Metric':<12} {'Retriever':<12} {'Reranker':<12} {'Δ':<8}")
    print("-"*60)

    for metric in ["Recall@5", "Recall@10", "MRR", "nDCG@10"]:
        retr_val = retr_metrics[metric] if retr_metrics else 0.0
        rerank_val = rerank_metrics[metric] if rerank_metrics else 0.0
        delta = rerank_val - retr_val
        delta_str = f"{delta:+.4f}" if retr_metrics else "N/A"
        print(f"{metric:<12} {retr_val:.4f}      {rerank_val:.4f}      {delta_str}")

    results_data = {
        "k": list(range(1, 11)),
        "retriever": {"Recall": [], "nDCG": []},
        "reranker": {"Recall": [], "nDCG": []},
        "mrr": {
            "retriever": retr_metrics["MRR"],
            "reranker": rerank_metrics["MRR"]
        }
    }

    for k in range(1, 11):
        results_data["retriever"]["Recall"].append(recall_at_k(preds_retr, targets_retr, k))
        results_data["retriever"]["nDCG"].append(ndcg_at_k(preds_retr, targets_retr, k))
        results_data["reranker"]["Recall"].append(recall_at_k(preds_rerank, targets_rerank, k))
        results_data["reranker"]["nDCG"].append(ndcg_at_k(preds_rerank, targets_rerank, k))

    with open("metrics_for_plots.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print("\nMetrics saved to metrics_for_plots.json")

if __name__ == "__main__":
    main()