from rank_bm25 import BM25Okapi
import re


class HybridReranker:
    def __init__(self, alpha: float = 0.75):
        """
        alpha = weight for vector score
        (1-alpha) = weight for bm25 score
        """
        self.alpha = alpha

    def tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def minmax_normalize(self, scores):
        mn = min(scores)
        mx = max(scores)

        if mx - mn == 0:
            return [0.0 for _ in scores]

        return [(s - mn) / (mx - mn) for s in scores]

    def rerank(self, query: str, matches: list[dict], top_k: int = 3):
        if not matches:
            return []

        docs = [m["metadata"]["text"] for m in matches]
        tokenized_docs = [self.tokenize(d) for d in docs]

        bm25 = BM25Okapi(tokenized_docs)
        query_tokens = self.tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)

        # normalize bm25 scores
        bm25_norm = self.minmax_normalize(bm25_scores)

        # normalize vector scores too (optional but recommended)
        vector_scores = [m["score"] for m in matches]
        vector_norm = self.minmax_normalize(vector_scores)

        # attach final hybrid score
        for i, m in enumerate(matches):
            m["bm25_score"] = float(bm25_scores[i])
            m["bm25_norm"] = float(bm25_norm[i])
            m["vector_norm"] = float(vector_norm[i])

            m["hybrid_score"] = (
                self.alpha * m["vector_norm"]
                + (1 - self.alpha) * m["bm25_norm"]
            )

        matches_sorted = sorted(
            matches,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return matches_sorted[:top_k]
