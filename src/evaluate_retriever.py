from metrics import *
from tqdm import tqdm


def evaluate_retriever(
    queries,
    relevant_docs,
    retriever,
    k_values=(5, 10, 20)
):
    """
    queries: List[str]
    relevant_docs: List[Set[doc_id]]
    retriever.search(query, top_k) -> List[{doc_id, text}]
    """
    all_preds = []

    for q in tqdm(queries, desc="Evaluating Retriever"):
        docs = retriever.search(q, top_k=max(k_values))
        all_preds.append([d["doc_id"] for d in docs])

    results = {}

    for k in k_values:
        results[f"Recall@{k}"] = recall_at_k(all_preds, relevant_docs, k)
        results[f"HitRate@{k}"] = hit_rate_at_k(all_preds, relevant_docs, k)

    results["MRR"] = mrr(all_preds, relevant_docs)
    results["nDCG@10"] = ndcg_at_k(all_preds, relevant_docs, 10)

    return results
