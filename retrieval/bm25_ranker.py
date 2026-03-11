# import math
# import re
# from collections import Counter


# class BM25Reranker:
#     """
#     BM25 reranker for list[dict] matches from Pinecone.

#     Each match must look like:
#     {
#         "id": "...",
#         "score": <vector_score>,
#         "metadata": {"text": "...", ...}
#     }
#     """

#     def __init__(self, k1: float = 1.5, b: float = 0.75):
#         self.k1 = k1
#         self.b = b

#         # small stopword list (you can expand later)
#         self.stopwords = {
#             "the", "is", "a", "an", "and", "or", "to", "of", "in", "on",
#             "for", "with", "as", "at", "by", "from", "that", "this", "it",
#             "are", "be", "was", "were", "will", "would", "can", "could",
#             "should", "into", "during", "only"
#         }

#     def tokenize(self, text: str) -> list[str]:
#         text = text.lower()
#         text = re.sub(r"[^a-z0-9\s]", " ", text)
#         tokens = text.split()

#         # remove stopwords + very short tokens
#         tokens = [t for t in tokens if t not in self.stopwords and len(t) > 2]
#         return tokens

#     def compute_idf(self, docs_tokens: list[list[str]]) -> dict[str, float]:
#         """
#         IDF(t) = log( 1 + (N - df + 0.5) / (df + 0.5) )
#         """
#         N = len(docs_tokens)
#         df = Counter()

#         for tokens in docs_tokens:
#             unique_tokens = set(tokens)
#             for t in unique_tokens:
#                 df[t] += 1

#         idf = {}
#         for t, freq in df.items():
#             idf[t] = math.log(1 + (N - freq + 0.5) / (freq + 0.5))

#         return idf

#     def bm25_score(
#         self,
#         query_tokens: list[str],
#         doc_tokens: list[str],
#         idf: dict[str, float],
#         avgdl: float
#     ) -> float:
#         doc_len = len(doc_tokens)
#         tf = Counter(doc_tokens)

#         score = 0.0
#         for t in query_tokens:
#             if t not in tf:
#                 continue

#             term_freq = tf[t]

#             numerator = term_freq * (self.k1 + 1)
#             denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))

#             score += idf.get(t, 0.0) * (numerator / denominator)

#         return score

#     def rerank(self, query: str, matches: list[dict]) -> list[dict]:
#         if not matches:
#             return []

#         query_tokens = self.tokenize(query)
#         if not query_tokens:
#             return matches

#         docs_tokens = [self.tokenize(m["metadata"]["text"]) for m in matches]
#         avgdl = sum(len(t) for t in docs_tokens) / len(docs_tokens)
#         idf = self.compute_idf(docs_tokens)

#         for m, doc_tokens in zip(matches, docs_tokens):
#             m["bm25_score"] = self.bm25_score(query_tokens, doc_tokens, idf, avgdl)

#         # sort by BM25 first, then Pinecone vector score
#         reranked = sorted(
#             matches,
#             key=lambda x: (x["bm25_score"], x["score"]),
#             reverse=True
#         )

#         return reranked

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
            reverse=True
        )

        return matches_sorted[:top_k]
