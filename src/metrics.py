import numpy as np


def recall_at_k(preds, targets, k):
    """
    preds: List[List[doc_id]]
    targets: List[Set[doc_id]]
    """
    recalls = []
    for p, t in zip(preds, targets):
        if len(t) == 0:
            continue
        recalls.append(len(set(p[:k]) & t) / len(t))
    return np.mean(recalls)


def hit_rate_at_k(preds, targets, k):
    hits = []
    for p, t in zip(preds, targets):
        hits.append(int(len(set(p[:k]) & t) > 0))
    return np.mean(hits)


def mrr(preds, targets):
    rr = []
    for p, t in zip(preds, targets):
        score = 0.0
        for i, doc_id in enumerate(p):
            if doc_id in t:
                score = 1 / (i + 1)
                break
        rr.append(score)
    return np.mean(rr)


def ndcg_at_k(preds, targets, k):
    def dcg(rel):
        return np.sum(
            [r / np.log2(i + 2) for i, r in enumerate(rel)]
        )

    scores = []
    for p, t in zip(preds, targets):
        rel = [1 if doc_id in t else 0 for doc_id in p[:k]]
        ideal = sorted(rel, reverse=True)
        scores.append(dcg(rel) / (dcg(ideal) + 1e-8))
    return np.mean(scores)
