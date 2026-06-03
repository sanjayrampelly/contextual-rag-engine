from rank_bm25 import BM25Okapi
import re


class BM25Reranker:
    def __init__(self):
        pass

    def tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def rerank(self, query: str, matches: list[dict], top_k: int = 3):
        if not matches:
            return []

        # Extract chunk texts from matches
        docs = [m["metadata"]["text"] for m in matches]

        tokenized_docs = [self.tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized_docs)

        query_tokens = self.tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)

        # attach bm25_score to each match
        for i, m in enumerate(matches):
            m["bm25_score"] = float(bm25_scores[i])

        # sort by bm25_score first, then pinecone vector score
        matches_sorted = sorted(
            matches,
            key=lambda x: (x["bm25_score"], x["score"]),
            reverse=True,
        )

        return matches_sorted[:top_k]
