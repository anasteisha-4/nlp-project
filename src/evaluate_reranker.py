from metrics import *
from tqdm import tqdm


def evaluate_reranker(
    queries,
    relevant_docs,
    retriever,
    reranker,
    retrieve_k=50,
    final_k=20
):
    """
    reranker.score(query, List[text]) -> List[float]
    """
    all_preds = []

    for q in tqdm(queries, desc="Evaluating Reranker"):
        candidates = retriever.search(q, top_k=retrieve_k)
        texts = [c["text"] for c in candidates]

        scores = reranker.score(q, texts)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        all_preds.append([
            c["doc_id"] for c, _ in ranked[:final_k]
        ])

    results = {}
    for k in [5, 10, 20]:
        results[f"Recall@{k}"] = recall_at_k(all_preds, relevant_docs, k)
        results[f"HitRate@{k}"] = hit_rate_at_k(all_preds, relevant_docs, k)

    results["MRR"] = mrr(all_preds, relevant_docs)
    results["nDCG@10"] = ndcg_at_k(all_preds, relevant_docs, 10)

    return results

