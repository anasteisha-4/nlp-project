def search(
    query,
    retriever,
    reranker=None,
    top_k_retrieve=20,
    top_k_final=5,
    batch_size=8
):
    """
    Unified search pipeline:
    Retriever -> (optional) Reranker
    """

    # 1. Retrieve candidates
    candidates = retriever.search(query, top_k=top_k_retrieve)

    if reranker is None:
        return candidates[:top_k_final]

    # 2. Rerank
    texts = [c["text"] for c in candidates]
    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        scores.extend(reranker.score(query, batch))

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [c for c, _ in ranked[:top_k_final]]
